import math

import pytest

from slam.tag_aimer import AimConfig, TagAimer

# A wall along y = 0 with three tags facing +y (into the room), like the
# field layout, plus one on the far wall facing -y.
TAGS = {
    2: {'x': 1.4, 'y': 0.0, 'yaw_deg': 90.0},
    3: {'x': 3.1, 'y': 0.0, 'yaw_deg': 90.0},
    4: {'x': 4.3, 'y': 0.0, 'yaw_deg': 90.0},
    9: {'x': 3.1, 'y': 4.0, 'yaw_deg': 270.0},
}
DEG = math.radians
FACING_WALL = (3.1, 0.9, DEG(-90))          # 0.9 m from tag 3, looking at it


def _aimer(**cfg):
    return TagAimer(TAGS, AimConfig(**cfg))


def test_prefers_an_oblique_tag_over_the_one_straight_ahead():
    """Tag 3 is dead ahead (face-on -> poor heading); tags 2 and 4 are off to
    the sides and seen at an angle. Look at one of those."""
    a = _aimer()
    pan = a.update(0.0, FACING_WALL, robot_still=True)
    assert a.target_id in (2, 4)
    assert abs(pan) > DEG(30)
    assert a.pan_cmd == pan


def test_pan_sign_left_is_positive():
    a = _aimer()
    # facing -y: tag 4 (larger x) is on the robot's LEFT -> positive pan
    s4, pan4 = a.score(4, FACING_WALL)
    s2, pan2 = a.score(2, FACING_WALL)
    assert pan4 > 0 and pan2 < 0
    assert s4 > 0
    assert s2 == 0.0            # 62 deg to the right: past the 60 deg travel
    assert a.score(2, (2.6, 1.2, DEG(-90)))[0] > 0      # from here it is not


def test_tags_behind_out_of_range_or_out_of_travel_score_zero():
    a = _aimer()
    assert a.score(9, FACING_WALL)[0] == 0.0             # behind the robot
    assert a.score(3, (3.1, 6.0, DEG(-90)))[0] == 0.0    # 6 m away
    assert a.score(3, (3.1, 0.2, DEG(-90)))[0] == 0.0    # closer than min
    # standing behind the wall the tag faces away
    assert a.score(3, (3.1, -1.0, DEG(90)))[0] == 0.0


def test_does_not_move_while_the_robot_is_moving_or_the_servo_is_settling():
    a = _aimer(settle_sec=0.6)
    assert a.update(0.0, FACING_WALL, robot_still=False) is None
    assert a.update(1.0, FACING_WALL, robot_still=True) is not None
    assert a.settling(1.3)
    assert a.update(1.3, (0.0, 0.0, 0.0), robot_still=True) is None
    assert not a.settling(1.7)


def test_small_pose_changes_do_not_jitter_the_servo():
    a = _aimer()
    a.update(0.0, FACING_WALL, robot_still=True)
    a.note_seen([a.target_id], 4.8)
    nudged = (FACING_WALL[0] + 0.03, FACING_WALL[1], FACING_WALL[2] + 0.02)
    assert a.update(5.0, nudged, robot_still=True) is None


def test_sticks_with_the_current_tag_unless_another_is_clearly_better():
    a = _aimer()
    a.update(0.0, FACING_WALL, robot_still=True)
    first = a.target_id
    a.note_seen([first], 1.0)
    # mirror-symmetric spot: the other side tag is now marginally better
    other = (3.1 + (0.05 if first == 2 else -0.05), 0.9, DEG(-90))
    a.note_seen([first], 4.9)
    a.update(5.0, other, robot_still=True)
    assert a.target_id == first


def test_gives_up_on_a_tag_it_cannot_see_and_tries_another():
    a = _aimer(look_timeout_sec=2.0, fail_cooldown_sec=10.0)
    a.update(0.0, FACING_WALL, robot_still=True)
    first = a.target_id
    # aimed, settled, 3 s later still never detected -> occluded
    pan = a.update(3.0, FACING_WALL, robot_still=True)
    assert a.target_id is not None and a.target_id != first
    assert pan is not None
    # ...and it is not retried during the cool-down
    a.note_seen([a.target_id], 3.5)
    a.update(8.0, FACING_WALL, robot_still=True)
    assert a.target_id != first


def test_seeing_the_target_keeps_it():
    a = _aimer(look_timeout_sec=2.0)
    a.update(0.0, FACING_WALL, robot_still=True)
    first = a.target_id
    for t in (1.0, 2.0, 3.0, 4.0):
        a.note_seen([first], t)
        a.update(t + 0.1, FACING_WALL, robot_still=True)
    assert a.target_id == first


def test_lost_pose_sweeps_the_camera_then_reports_exhausted():
    a = _aimer(settle_sec=0.5, sweep_dwell_sec=1.0,
               sweep_rad=(0.0, DEG(40), DEG(-40)))
    t, issued = 0.0, []
    for _ in range(40):
        pan = a.update(t, None, robot_still=True)
        if pan is not None:
            issued.append(round(math.degrees(pan)))
        t += 0.2
    assert issued == [0, 40, -40]
    assert a.sweep_exhausted is True
    assert a.update(t, None, robot_still=True) is None
    a.reset_sweep()                              # the chassis turned
    assert a.update(t, None, robot_still=True) == 0.0


def test_sweep_dwells_long_enough_at_each_angle():
    a = _aimer(settle_sec=0.5, sweep_dwell_sec=1.0,
               sweep_rad=(0.0, DEG(40)))
    assert a.update(0.0, None, robot_still=True) == 0.0
    assert a.update(1.4, None, robot_still=True) is None     # < 0.5 + 1.0
    assert a.update(1.6, None, robot_still=True) == pytest.approx(DEG(40))


def test_regaining_the_pose_ends_the_sweep_and_aims_at_a_tag():
    a = _aimer()
    a.update(0.0, None, robot_still=True)
    pan = a.update(2.0, FACING_WALL, robot_still=True)
    assert a.target_id in (2, 4)
    assert pan is not None
    assert a.sweep_exhausted is False


def test_no_usable_tag_looks_straight_ahead():
    a = _aimer()
    a.pan_cmd = DEG(40)
    pan = a.update(0.0, (10.0, 10.0, 0.0), robot_still=True)   # far from all
    assert pan == 0.0 and a.target_id is None


def test_reset_recentres_bookkeeping():
    a = _aimer()
    a.update(0.0, FACING_WALL, robot_still=True)
    a.reset()
    assert a.pan_cmd == 0.0 and a.target_id is None
    assert not a.settling(0.1)
