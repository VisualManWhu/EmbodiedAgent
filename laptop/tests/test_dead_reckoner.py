"""Unit tests for dead_reckoner: integrate cmd_vel pulses between tag fixes."""
import math

import pytest

from slam.dead_reckoner import DeadReckoner


def test_no_pose_yet_returns_none():
    dr = DeadReckoner()
    pose, stale = dr.pose_at(now=10.0)
    assert pose is None
    assert stale is True


def test_forward_pulse_translates_along_yaw():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.15, vy=0.0, wz=0.0, seconds=1.0, started_at=0.0)
    pose, stale = dr.pose_at(now=1.0)
    x, y, yaw = pose
    assert x == pytest.approx(0.15, abs=1e-3)
    assert y == pytest.approx(0.0, abs=1e-3)
    assert yaw == pytest.approx(0.0, abs=1e-3)
    assert stale is False


def test_pure_rotation():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.0, vy=0.0, wz=1.0, seconds=1.0, started_at=0.0)
    pose, _ = dr.pose_at(now=1.0)
    _, _, yaw = pose
    assert yaw == pytest.approx(1.0, abs=1e-3)


def test_strafe_pulse_translates_sideways():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.0, vy=0.15, wz=0.0, seconds=1.0, started_at=0.0)
    pose, _ = dr.pose_at(now=1.0)
    x, y, _ = pose
    assert x == pytest.approx(0.0, abs=1e-3)
    assert y == pytest.approx(0.15, abs=1e-3)


def test_partial_pulse_only_counts_elapsed_portion():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.2, vy=0.0, wz=0.0, seconds=2.0, started_at=0.0)
    pose, _ = dr.pose_at(now=0.5)         # only 0.5 s elapsed
    x, _, _ = pose
    assert x == pytest.approx(0.10, abs=5e-3)


def test_time_cap_marks_stale():
    dr = DeadReckoner(time_limit_sec=1.0)
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.05, vy=0.0, wz=0.0, seconds=1.0, started_at=0.0)
    _, stale = dr.pose_at(now=5.0)
    assert stale is True


def test_distance_cap_marks_stale():
    dr = DeadReckoner(distance_limit_m=0.1)
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.5, vy=0.0, wz=0.0, seconds=1.0, started_at=0.0)
    _, stale = dr.pose_at(now=1.0)
    assert stale is True


def test_reset_clears_pulses():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.15, vy=0.0, wz=0.0, seconds=1.0, started_at=0.0)
    dr.reset((1.0, 2.0, 0.5), t=10.0)
    pose, stale = dr.pose_at(now=10.0)
    assert pose == (1.0, 2.0, 0.5)
    assert stale is False


def test_tag_fix_mid_pulse_keeps_the_rest_of_the_pulse():
    """Field bug: a rotate pulse is sent, and a moment later a (still
    tag-in-view) frame re-localizes the robot. That fix must not erase the
    pulse — the robot keeps turning after it. Forgetting the pulse made nav
    think it had not turned at all and issue the same rotate again."""
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.0, vy=0.0, wz=1.0, seconds=1.5, started_at=0.0)
    # fix 0.3 s into the turn: the tag says we have turned 0.3 rad so far
    dr.reset((0.0, 0.0, 0.3), t=0.3)
    pose, stale = dr.pose_at(now=2.0)               # pulse long finished
    assert pose[2] == pytest.approx(1.5, abs=1e-2)  # 0.3 + remaining 1.2 s
    assert stale is False


def test_tag_fix_mid_forward_pulse_keeps_the_remaining_distance():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.15, vy=0.0, wz=0.0, seconds=2.0, started_at=0.0)
    dr.reset((0.15, 0.0, 0.0), t=1.0)
    pose, _ = dr.pose_at(now=3.0)
    assert pose[0] == pytest.approx(0.30, abs=1e-3)


def test_tag_fix_before_a_queued_pulse_starts_keeps_it_whole():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.0, vy=0.0, wz=1.0, seconds=1.0, started_at=0.5)
    dr.reset((0.0, 0.0, 0.0), t=0.2)
    pose, _ = dr.pose_at(now=2.0)
    assert pose[2] == pytest.approx(1.0, abs=1e-2)


# ---- yaw-rate model: what the motors really do with a commanded twist ----

from slam.dead_reckoner import modeled_yaw_rate          # noqa: E402

W_CMD, W_EFF = 0.5, 1.12        # in-place: commanded 0.5 -> measured 1.12 rad/s


def test_in_place_rotation_matches_the_hand_calibration():
    assert modeled_yaw_rate(0.0, 0.0, 0.5, W_CMD, W_EFF) == pytest.approx(1.12)
    assert modeled_yaw_rate(0.0, 0.0, -0.5, W_CMD, W_EFF) == pytest.approx(-1.12)


def test_steering_while_driving_turns_far_less_than_in_place():
    """Field bug: the reckoner scaled forward-pulse steering by the in-place
    factor (x2.24), believing a 2.5 s forward pulse turned the robot ~80 deg.
    In reality the duty floor leaves only a small left/right difference, so
    nav 'corrected' a turn that never happened."""
    rate = modeled_yaw_rate(0.15, 0.0, 0.25, W_CMD, W_EFF)
    assert 0.10 < rate < 0.20                      # ~0.15 rad/s, not 0.56
    assert modeled_yaw_rate(0.15, 0.0, -0.25, W_CMD, W_EFF) == pytest.approx(-rate)


def test_tiny_steering_is_swallowed_by_the_duty_floor():
    assert abs(modeled_yaw_rate(0.15, 0.0, 0.06, W_CMD, W_EFF)) < 0.06


def test_straight_and_strafe_do_not_turn():
    assert modeled_yaw_rate(0.15, 0.0, 0.0, W_CMD, W_EFF) == pytest.approx(0.0)
    assert modeled_yaw_rate(0.0, 0.15, 0.0, W_CMD, W_EFF) == pytest.approx(0.0)
    assert modeled_yaw_rate(-0.15, 0.0, 0.0, W_CMD, W_EFF) == pytest.approx(0.0)


def test_cancelled_pulse_is_not_integrated():
    """A pulse during which the robot was stuck must not move the pose."""
    dr = DeadReckoner()
    dr.reset((1.0, 1.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.25, vy=0.0, wz=0.0, seconds=1.8, started_at=0.5)
    assert dr.pose_at(now=3.0)[0][0] == pytest.approx(1.45, abs=1e-3)
    dr.cancel_pulse(0.5)
    assert dr.pose_at(now=3.0)[0] == (1.0, 1.0, 0.0)


def test_pulse_cut_short_by_a_block_only_counts_up_to_the_block():
    dr = DeadReckoner()
    dr.reset((1.0, 1.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.25, vy=0.0, wz=0.0, seconds=1.6, started_at=1.0)
    dr.truncate_pulses(now=1.8)                     # blocked 0.8 s in
    pose, _ = dr.pose_at(now=5.0)
    assert pose[0] == pytest.approx(1.0 + 0.25 * 0.8, abs=1e-3)
    # a pulse that had already finished is left as it was
    dr.reset((0.0, 0.0, 0.0), t=10.0)
    dr.record_pulse(vx=0.25, vy=0.0, wz=0.0, seconds=1.0, started_at=10.0)
    dr.truncate_pulses(now=13.0)
    assert dr.pose_at(now=14.0)[0][0] == pytest.approx(0.25, abs=1e-3)
