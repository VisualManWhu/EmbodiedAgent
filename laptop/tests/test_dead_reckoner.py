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
