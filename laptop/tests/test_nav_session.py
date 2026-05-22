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
    assert cmd.wz == 0.0                          # dead-on -> no steering


def test_forward_pulse_blends_steering_when_slightly_off():
    """A small heading error (within heading_tol) drives forward but blends
    in proportional steering toward the goal — no drift-then-snap."""
    s = _sess(goal=(2.0, 0.0), heading_tol_rad=0.15)
    # goal dead ahead in +x, robot yawed +0.1 rad: heading_err = -0.1 (<tol)
    cmd = s.tick(robot_pose=(0.0, 0.0, 0.1), now=0.0)
    assert cmd.kind == 'forward'
    assert cmd.vx > 0
    assert cmd.wz < 0                             # steers back toward the line
    assert abs(cmd.wz) <= 0.25                    # capped by forward_steer_max


def test_no_command_while_waiting_for_pulse():
    s = _sess()
    s.tick((0.0, 0.0, 0.0), 0.0)                                # one pulse issued
    assert s.tick((0.0, 0.0, 0.0), 0.1) is None                 # no overlap


def test_pulse_done_unblocks_next_tick():
    s = _sess()
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('pulse_done')
    # advance well past the post-pulse dwell so the next pulse is issued
    cmd = s.tick((0.05, 0.0, 0.0), 30.0)
    assert cmd is not None
    assert cmd.kind == 'forward'


def test_dwell_holds_between_drive_pulses():
    """After a drive pulse the robot dwells (nav_dwell_sec) before the next —
    a tick inside the dwell window issues no command."""
    s = _sess(nav_dwell_sec=2.0)
    cmd = s.tick((0.0, 0.0, 0.0), 0.0)
    assert cmd.kind == 'forward'
    s.on_pi_event('pulse_done')
    # tick shortly after pulse completion, still inside the dwell -> None
    assert s.tick((0.1, 0.0, 0.0), cmd.seconds + 0.5) is None
    # past the dwell -> next pulse issued
    assert s.tick((0.1, 0.0, 0.0), cmd.seconds + 2.5) is not None


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


def test_blocked_strafes_away_per_side():
    """+vy = strafe LEFT (matches agent_node._dir_to_twist); strafe must move
    AWAY from the blocked side, never into it."""
    # left blocked -> strafe right -> negative vy
    s = _sess(strafe_speed=0.15)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:left')
    cmd = s.tick((0.0, 0.0, 0.0), 0.1)
    assert cmd.kind == 'strafe'
    assert cmd.vy < 0, f'blocked:left should strafe right (vy<0), got vy={cmd.vy}'

    # right blocked -> strafe left -> positive vy
    s = _sess(strafe_speed=0.15)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:right')
    cmd = s.tick((0.0, 0.0, 0.0), 0.1)
    assert cmd.kind == 'strafe'
    assert cmd.vy > 0, f'blocked:right should strafe left (vy>0), got vy={cmd.vy}'


def test_blocked_front_near_goal_is_arrival():
    """blocked:front within blocked_arrival_dist of the goal is the goal
    object itself — ARRIVED, not an obstacle. The robot is past the arrival
    radius (tick did not already arrive) but close enough."""
    s = _sess(goal=(2.0, 0.0), arrived_radius_m=0.4,
              blocked_arrival_dist_m=1.0)
    cmd = s.tick((1.3, 0.0, 0.0), 0.0)       # 0.7 m out: not arrived, drives
    assert cmd.kind == 'forward'
    s.on_pi_event('blocked:front')           # ultrasonic trips on the object
    assert s.state is State.ARRIVED


def test_blocked_front_with_target_ahead_is_arrival():
    """blocked:front while the camera sees the goal class close ahead is an
    arrival even if the map distance is still large."""
    s = _sess(goal=(5.0, 0.0))
    s.tick((0.0, 0.0, 0.0), 0.0, target_ahead=True)
    s.on_pi_event('blocked:front')
    assert s.state is State.ARRIVED


def test_blocked_front_far_from_goal_is_obstacle():
    """blocked:front far from the goal with no target in view -> obstacle ->
    avoidance side-step, not arrival."""
    s = _sess(goal=(5.0, 0.0), blocked_arrival_dist_m=1.0)
    s.tick((0.0, 0.0, 0.0), 0.0, target_ahead=False)
    s.on_pi_event('blocked:front')
    assert s.state is not State.ARRIVED
    cmd = s.tick((0.0, 0.0, 0.0), 0.1)
    assert cmd.kind == 'strafe'


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


def test_scan_when_no_pose_at_start():
    """NAV started with no localization at all (robot_pose is None) must
    rotate-scan to acquire a tag, not hang in INIT."""
    s = _sess(scan_max_rotations=12)
    cmd = s.tick(robot_pose=None, now=0.0)
    assert cmd is not None
    assert cmd.kind == 'scan_rotate'


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
    t = 10.0
    for _ in range(2):
        cmd = s.tick((0.0, 0.0, 0.0), t, pose_stale=True)
        assert cmd.kind == 'scan_rotate'
        s.on_pi_event('pulse_done')
        t += 10.0                          # advance well past the scan dwell
    s.tick((0.0, 0.0, 0.0), t, pose_stale=True)
    assert s.state is State.FAILED
    assert s.fail_reason == 'lost'


def test_scan_dwell_holds_between_rotations():
    """A scan rotation is followed by a still dwell — a tick inside the dwell
    window issues no new pulse so the detector can grab clean frames."""
    s = _sess(scan_max_rotations=12)
    s.on_tag_fix(t=0.0)
    cmd = s.tick((0.0, 0.0, 0.0), 10.0, pose_stale=True)
    assert cmd.kind == 'scan_rotate'
    s.on_pi_event('pulse_done')
    # a tick shortly after the pulse, still inside the dwell -> no command
    assert s.tick((0.0, 0.0, 0.0), 10.5, pose_stale=True) is None
    assert s.scan_count == 1


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
    # first lost period: 2 scan pulses then tag found (advance past each dwell)
    s.tick((0.0, 0.0, 0.0), 10.0, pose_stale=True)
    s.on_pi_event('pulse_done')
    s.tick((0.0, 0.0, 0.0), 20.0, pose_stale=True)
    s.on_pi_event('pulse_done')
    assert s.scan_count == 2
    s.on_tag_fix(t=22.0)
    assert s.scan_count == 0
    # second lost period: should get its full budget of 3 pulses again
    t = 30.0
    for _ in range(3):
        cmd = s.tick((0.0, 0.0, 0.0), t, pose_stale=True)
        assert cmd.kind == 'scan_rotate'
        s.on_pi_event('pulse_done')
        t += 10.0
    # next tick exhausts -> FAILED('lost')
    s.tick((0.0, 0.0, 0.0), t, pose_stale=True)
    assert s.state is State.FAILED
    assert s.fail_reason == 'lost'
