"""Unit tests for NavSession: pose+goal -> motion pulses + Pi event handling."""
import math

import pytest

from slam.nav_session import NavCommand, NavConfig, NavSession, State


def _sess(goal=(2.0, 0.0), **cfg):
    return NavSession(goal_xy=goal, goal_id=1, goal_label='chair',
                      config=NavConfig(**cfg))


def test_arrives_when_inside_radius():
    s = _sess(arrived_radius_m=0.4)
    cmd = s.tick(robot_pose=(2.0, 0.0, 0.0), now=0.0)
    assert s.state is State.ARRIVED
    assert cmd.kind == 'stop'


def test_rotates_when_heading_off():
    s = _sess(heading_tol_rad=0.35)
    cmd = s.tick(robot_pose=(0.0, 0.0, math.pi / 2), now=0.0)  # goal to the right
    assert cmd.kind == 'rotate'
    assert cmd.wz < 0                                          # turn right
    assert cmd.vx == 0.0


def test_drives_forward_when_heading_ok():
    s = _sess()
    cmd = s.tick(robot_pose=(0.0, 0.0, 0.0), now=0.0)
    assert cmd.kind == 'forward'
    assert cmd.vx > 0
    assert cmd.wz == 0.0


def test_no_command_while_waiting_for_pulse():
    s = _sess()
    s.tick((0.0, 0.0, 0.0), 0.0)                                # one pulse issued
    assert s.tick((0.0, 0.0, 0.0), 0.1) is None                 # no overlap


def test_pulse_done_unblocks_next_tick():
    s = _sess()
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('pulse_done')
    cmd = s.tick((0.05, 0.0, 0.0), 0.5)
    assert cmd is not None
    assert cmd.kind == 'forward'


def test_blocked_queues_strafe_then_forward():
    s = _sess(strafe_speed=0.15)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:left')
    cmd1 = s.tick((0.0, 0.0, 0.0), 0.1)
    cmd2 = s.tick((0.0, 0.0, 0.0), 0.2)                         # next after pulse_done
    s.on_pi_event('pulse_done')
    cmd2 = s.tick((0.0, 0.0, 0.0), 0.3)
    assert cmd1.kind == 'strafe'
    assert cmd1.vy != 0
    assert cmd2.kind == 'forward'


def test_three_blocks_fail():
    s = _sess(block_retries=3)
    for _ in range(3):
        s.tick((0.0, 0.0, 0.0), 0.0)
        s.on_pi_event('blocked:front')
    assert s.state is State.FAILED
    assert s.fail_reason == 'blocked'


def test_block_count_resets_on_clean_pulse():
    s = _sess(block_retries=3)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:front')
    s.tick((0.0, 0.0, 0.0), 0.1)                                # strafe queued
    s.on_pi_event('pulse_done')
    s.tick((0.0, 0.0, 0.0), 0.2)                                # forward of strafe seq
    s.on_pi_event('pulse_done')
    assert s.block_count == 0


def test_scan_when_pose_stale_and_no_tag():
    s = _sess(scan_max_rotations=4)
    s.on_tag_fix(t=0.0)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('pulse_done')
    # 10 s later, no further tag fix -> trigger scan
    cmd = s.tick(robot_pose=(0.0, 0.0, 0.0), now=10.0, pose_stale=True)
    assert cmd.kind == 'scan_rotate'


def test_scan_exhausted_fails():
    s = _sess(scan_max_rotations=2)
    s.on_tag_fix(t=0.0)
    for i in range(2):
        s.tick((0.0, 0.0, 0.0), 10.0 + i, pose_stale=True)
        s.on_pi_event('pulse_done')
    s.tick((0.0, 0.0, 0.0), 20.0, pose_stale=True)
    assert s.state is State.FAILED
    assert s.fail_reason == 'lost'


def test_tag_fix_during_scan_resumes_drive():
    s = _sess(scan_max_rotations=4)
    s.on_tag_fix(t=0.0)
    s.tick((0.0, 0.0, 0.0), 10.0, pose_stale=True)               # scan pulse
    s.on_pi_event('pulse_done')
    s.on_tag_fix(t=11.0)                                         # recovered
    cmd = s.tick((0.1, 0.0, 0.0), 11.5, pose_stale=False)
    assert cmd.kind in ('rotate', 'forward')


def test_scan_count_clears_after_tag_recovery():
    """After scan recovers via on_tag_fix, a later stale period must restart
    scanning from scan_count=0, not from where the previous scan left off."""
    s = _sess(scan_max_rotations=3)
    s.on_tag_fix(t=0.0)
    # first lost period: 2 scan pulses then tag found
    s.tick((0.0, 0.0, 0.0), 10.0, pose_stale=True)
    s.on_pi_event('pulse_done')
    s.tick((0.0, 0.0, 0.0), 11.0, pose_stale=True)
    s.on_pi_event('pulse_done')
    assert s.scan_count == 2
    s.on_tag_fix(t=12.0)
    assert s.scan_count == 0
    # second lost period: should get its full budget of 3 pulses again
    for i in range(3):
        s.tick((0.0, 0.0, 0.0), 20.0 + i, pose_stale=True)
        s.on_pi_event('pulse_done')
    # 4th tick exhausts -> FAILED('lost')
    s.tick((0.0, 0.0, 0.0), 25.0, pose_stale=True)
    assert s.state is State.FAILED
    assert s.fail_reason == 'lost'
