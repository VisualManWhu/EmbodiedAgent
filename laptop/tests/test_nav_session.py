"""Unit tests for NavSession: pose+goal -> motion pulses + Pi event handling."""
import math

import pytest

from slam.nav_session import NavCommand, NavConfig, NavSession, State


def _sess(goal=(2.0, 0.0), fixed=True, **cfg):
    """Build a NavSession. ``fixed`` simulates an initial confirmed
    localization (a real tag fix) so the session may drive — set False to
    test the pre-first-fix scan-to-localize behaviour."""
    s = NavSession(goal_xy=goal, goal_id=1, goal_label='chair',
                   config=NavConfig(**cfg))
    if fixed:
        s.on_tag_fix(0.0)
    return s


def test_scans_before_first_fix():
    """A goto issued with no confirmed localization yet must rotate-scan to
    acquire a tag before driving — even if a (dead-reckoned) pose exists."""
    s = _sess(fixed=False)
    cmd = s.tick(robot_pose=(0.0, 0.0, 0.0), now=0.0)   # plausible pose, no fix
    assert cmd.kind == 'scan_rotate'


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


def _drain_queue(s, t0=0.1):
    """Pop all queued avoidance pulses, advancing time + pulse_done each."""
    cmds = []
    t = t0
    while True:
        cmd = s.tick((0.0, 0.0, 0.0), t)
        if cmd is None:
            break
        cmds.append(cmd)
        s.on_pi_event('pulse_done')
        t += 1.0
    return cmds


def test_blocked_queues_backup_strafe_forward():
    s = _sess(strafe_speed=0.15)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:left')
    cmds = _drain_queue(s)
    kinds = [c.kind for c in cmds[:3]]
    assert kinds == ['backward', 'strafe', 'forward']
    assert cmds[0].vx < 0                                       # backs up first
    assert cmds[1].vy != 0


def test_blocked_strafes_away_per_side():
    """+vy = strafe LEFT (matches agent_node._dir_to_twist); strafe must move
    AWAY from the blocked side, never into it."""
    # left blocked -> strafe right -> negative vy
    s = _sess(strafe_speed=0.15)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:left')
    strafe = [c for c in _drain_queue(s) if c.kind == 'strafe'][0]
    assert strafe.vy < 0, f'blocked:left should strafe right (vy<0), got {strafe.vy}'

    # right blocked -> strafe left -> positive vy
    s = _sess(strafe_speed=0.15)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:right')
    strafe = [c for c in _drain_queue(s) if c.kind == 'strafe'][0]
    assert strafe.vy > 0, f'blocked:right should strafe left (vy>0), got {strafe.vy}'


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
    # obstacle -> avoidance maneuver queued (back up, strafe away, forward)
    kinds = [c.kind for c in _drain_queue(s)]
    assert 'strafe' in kinds and 'backward' in kinds


def test_blocks_trigger_backup_recovery_not_fail():
    """block_retries side-steps failing does NOT fail outright — it queues a
    back-up recovery and keeps navigating."""
    s = _sess(goal=(5.0, 0.0), block_retries=3, max_recovery_cycles=5)
    for _ in range(3):
        s.tick((0.0, 0.0, 0.0), 0.0)
        s.on_pi_event('blocked:front')
    assert s.state is not State.FAILED
    assert s.recovery_count == 1
    # next drive tick issues the back-up pulse (reverse)
    cmd = s.tick((0.0, 0.0, 0.0), 0.1)
    assert cmd.kind == 'backward'
    assert cmd.vx < 0


def test_recovery_cycles_exhausted_finally_fails():
    """After max_recovery_cycles back-ups the session does finally FAIL."""
    s = _sess(goal=(5.0, 0.0), block_retries=3, max_recovery_cycles=2)
    # each recovery cycle = block_retries blocked events
    for _ in range(3 * 3):
        s.tick((0.0, 0.0, 0.0), 0.0)
        s.on_pi_event('blocked:front')
    assert s.state is State.FAILED
    assert s.fail_reason == 'blocked'


def test_block_count_resets_on_clean_pulse():
    s = _sess(goal=(5.0, 0.0), block_retries=3)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:front')
    # drain the whole backup+strafe+forward avoidance queue; block_count
    # clears only once the queue is empty on a clean pulse_done.
    _drain_queue(s)
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


def _front_strafe(s):
    """Queue an avoidance for a blocked:front and return its strafe pulse."""
    s.queued.clear()
    s.on_pi_event('blocked:front')
    return [c for c in s.queued if c.kind == 'strafe'][0]


def test_repeated_front_blocks_strafe_further_each_time():
    """Field bug: a fixed 9 cm side-step three times never cleared a real
    obstacle. A second block in a row must step sideways further."""
    s = _sess(goal=(5.0, 0.0), block_retries=3, strafe_seconds=1.0)
    s.tick((0.0, 0.0, 0.0), 0.0)
    first = _front_strafe(s)
    second = _front_strafe(s)
    assert second.seconds > first.seconds
    assert second.vy * first.vy > 0                 # same side while escalating


def test_strafe_duration_is_capped():
    s = _sess(goal=(5.0, 0.0), block_retries=10, strafe_seconds=1.0,
              strafe_max_seconds=2.5)
    s.tick((0.0, 0.0, 0.0), 0.0)
    for _ in range(6):
        last = _front_strafe(s)
    assert last.seconds == 2.5


def test_after_a_failed_recovery_cycle_the_other_side_is_tried():
    s = _sess(goal=(5.0, 0.0), block_retries=3, max_recovery_cycles=5)
    s.tick((0.0, 0.0, 0.0), 0.0)
    first = _front_strafe(s)
    _front_strafe(s)
    s.on_pi_event('blocked:front')                  # 3rd -> recovery cycle
    assert s.recovery_count == 1
    after = _front_strafe(s)
    assert after.vy * first.vy < 0                  # opposite side now


# ---- final heading (tag goals: stop in front of the tag, facing it) -------

def _facing_sess(final_yaw, **cfg):
    s = NavSession(goal_xy=(2.0, 0.0), goal_id=None, goal_label='tag0',
                   config=NavConfig(**cfg), final_yaw=final_yaw)
    s.on_tag_fix(0.0)
    return s


def test_at_goal_but_facing_away_rotates_to_final_heading_before_arriving():
    s = _facing_sess(final_yaw=math.pi / 2)
    cmd = s.tick(robot_pose=(2.0, 0.0, 0.0), now=0.0)
    assert cmd.kind == 'rotate' and cmd.wz > 0       # turn left toward +90 deg
    assert cmd.vx == 0.0
    assert s.state is State.WAITING_PULSE


def test_at_goal_and_facing_right_way_arrives():
    s = _facing_sess(final_yaw=math.pi / 2)
    cmd = s.tick(robot_pose=(2.0, 0.0, math.pi / 2 + 0.1), now=0.0)
    assert s.state is State.ARRIVED and cmd.kind == 'stop'


def test_facing_turn_waits_for_dwell_then_arrives_once_aligned():
    s = _facing_sess(final_yaw=math.pi / 2, nav_dwell_sec=2.0)
    first = s.tick(robot_pose=(2.0, 0.0, 0.0), now=0.0)
    s.on_pi_event('pulse_done')
    # still settling after the turn: no new command, not arrived yet
    assert s.tick(robot_pose=(2.0, 0.0, 1.0), now=first.seconds + 0.5) is None
    cmd = s.tick(robot_pose=(2.0, 0.0, math.pi / 2), now=first.seconds + 2.5)
    assert s.state is State.ARRIVED and cmd.kind == 'stop'


def test_facing_does_not_drive_away_if_the_turn_nudges_it_out_of_radius():
    s = _facing_sess(final_yaw=math.pi / 2, arrived_radius_m=0.4)
    first = s.tick(robot_pose=(2.0, 0.0, 0.0), now=0.0)
    s.on_pi_event('pulse_done')
    cmd = s.tick(robot_pose=(1.5, 0.0, math.pi / 2), now=first.seconds + 3.0)
    assert s.state is State.ARRIVED and cmd.kind == 'stop'


def test_facing_gives_up_after_a_few_attempts_and_still_arrives():
    s = _facing_sess(final_yaw=math.pi / 2, final_face_max=2, nav_dwell_sec=0.0)
    t = 0.0
    for _ in range(2):
        cmd = s.tick(robot_pose=(2.0, 0.0, 0.0), now=t)
        assert cmd.kind == 'rotate'
        s.on_pi_event('pulse_done')
        t += 10.0
    cmd = s.tick(robot_pose=(2.0, 0.0, 0.0), now=t)   # never converged
    assert s.state is State.ARRIVED and cmd.kind == 'stop'


def test_no_final_heading_arrives_immediately_as_before():
    s = _sess(arrived_radius_m=0.4)
    cmd = s.tick(robot_pose=(2.0, 0.0, 3.0), now=0.0)
    assert s.state is State.ARRIVED and cmd.kind == 'stop'


def test_final_heading_is_not_confirmed_by_dead_reckoning_alone():
    """Field bug: the facing turn overshot by 32 deg, but dead reckoning
    assumes a perfect turn, so the robot 'arrived' looking away from the tag
    it had been sent to face."""
    s = _facing_sess(final_yaw=math.pi / 2, nav_dwell_sec=0.0,
                     final_fix_wait_sec=12.0)
    first = s.tick((2.0, 0.0, 0.0), 0.0)
    assert first.kind == 'rotate'
    s.on_pi_event('pulse_done')
    aligned = (2.0, 0.0, math.pi / 2)
    # dead reckoning says the turn landed perfectly: not good enough
    assert s.tick(aligned, 3.0, pose_fresh=False) is None
    assert s.state is not State.ARRIVED
    # a real fix says it overshot -> corrective turn
    cmd = s.tick((2.0, 0.0, math.pi / 2 + 0.55), 5.0, pose_fresh=True)
    assert cmd.kind == 'rotate' and cmd.wz < 0
    s.on_pi_event('pulse_done')
    cmd = s.tick(aligned, 9.0, pose_fresh=True)          # fix confirms it
    assert s.state is State.ARRIVED and cmd.kind == 'stop'


def test_final_heading_settles_for_dead_reckoning_after_waiting_for_a_fix():
    s = _facing_sess(final_yaw=math.pi / 2, nav_dwell_sec=0.0,
                     final_fix_wait_sec=12.0)
    aligned = (2.0, 0.0, math.pi / 2)
    assert s.tick(aligned, 0.0, pose_fresh=False) is None
    assert s.tick(aligned, 11.0, pose_fresh=False) is None
    cmd = s.tick(aligned, 12.5, pose_fresh=False)
    assert s.state is State.ARRIVED and cmd.kind == 'stop'


def test_waiting_for_a_fix_does_not_drive_away_from_the_goal():
    s = _facing_sess(final_yaw=math.pi / 2, arrived_radius_m=0.4)
    assert s.tick((2.0, 0.0, math.pi / 2), 0.0, pose_fresh=False) is None
    # the dead-reckoned position wanders out of the radius meanwhile
    assert s.tick((1.5, 0.0, math.pi / 2), 2.0, pose_fresh=False) is None
