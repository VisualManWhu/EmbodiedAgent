"""pantilt_check analysis: does the pan-tilt model agree with what the camera
actually does? Synthetic poses only — no robot, no camera."""
import math

import numpy as np
import pytest

from slam import transforms
from slam.pantilt_check import analyze, camera_yaw_elev, fit_linear

MOUNT = {'forward_m': 0.0, 'lateral_m': 0.0, 'height_m': 0.15}
BASE = transforms.make_transform(transforms.rot_z(0.7), [1.2, 0.4, 0.0])
PANS = [math.radians(d) for d in range(-60, 61, 20)]


def _samples(pan_gain=1.0, pan_offset=0.0, tilt_gain=1.0, pans=PANS,
             tilts=(0.0,), lens_forward_m=0.0, yaw_noise=None):
    """What the camera would really report if the servo turned
    ``gain * commanded + offset`` while the robot base stayed put.
    ``lens_forward_m`` puts the lens ahead of the pan axis, so it swings
    sideways as the head pans (the model has no such term)."""
    out = []
    for i, pan in enumerate(pans):
        for tilt in tilts:
            true_pan = pan_gain * pan + pan_offset
            T_base_cam = transforms.camera_pose_in_base(
                true_pan, tilt_gain * tilt, MOUNT)
            T_base_cam[:3, 3] += transforms.rot_z(true_pan) @ np.array(
                [lens_forward_m, 0.0, 0.0])
            T = BASE @ T_base_cam
            if yaw_noise is not None:
                T = transforms.make_transform(
                    transforms.rot_z(yaw_noise[i]), [0, 0, 0]) @ T
            out.append({'pan_cmd': pan, 'tilt_cmd': tilt, 'T_map_cam': T,
                        'tag_ids': [0]})
    return out


def test_camera_yaw_and_elevation_follow_the_optical_axis():
    T = BASE @ transforms.camera_pose_in_base(0.3, 0.0, MOUNT)
    yaw, elev = camera_yaw_elev(T)
    assert yaw == pytest.approx(0.7 + 0.3)
    assert elev == pytest.approx(0.0, abs=1e-9)
    # the model's positive tilt pitches the camera DOWN
    _, elev = camera_yaw_elev(BASE @ transforms.camera_pose_in_base(
        0.0, 0.2, MOUNT))
    assert elev == pytest.approx(-0.2)


def test_fit_linear_recovers_gain_and_offset():
    xs = [-1.0, 0.0, 1.0, 2.0]
    gain, offset, resid = fit_linear(xs, [0.8 * x + 0.05 for x in xs])
    assert gain == pytest.approx(0.8)
    assert offset == pytest.approx(0.05)
    assert resid == pytest.approx(0.0, abs=1e-9)
    assert fit_linear([0.3], [0.3]) is None         # one point: no fit


def test_perfect_servo_gives_constant_base_pose_and_ok_verdict():
    r = analyze(_samples(), MOUNT)
    assert r['pan_gain'] == pytest.approx(1.0, abs=1e-6)
    assert r['base_pos_spread_m'] < 1e-6
    assert r['base_yaw_spread_rad'] < 1e-6
    assert r['verdict'] == 'ok'


def test_servo_turning_the_wrong_way_is_reported_as_inverted():
    r = analyze(_samples(pan_gain=-1.0), MOUNT)
    assert r['pan_gain'] == pytest.approx(-1.0, abs=1e-6)
    assert r['pan_inverted'] is True
    assert r['base_yaw_spread_rad'] > math.radians(60)      # model is wrong
    assert r['corrected_yaw_spread_rad'] < 1e-6             # but fixable
    assert r['verdict'] == 'calibrate'


def test_servo_gain_error_is_measured_and_fixable():
    r = analyze(_samples(pan_gain=0.85, pan_offset=0.04), MOUNT)
    assert r['pan_gain'] == pytest.approx(0.85, abs=1e-6)
    assert r['pan_inverted'] is False
    assert r['base_yaw_spread_rad'] > math.radians(5)
    assert r['corrected_yaw_spread_rad'] < 1e-6
    assert r['verdict'] == 'calibrate'


def test_erratic_servo_cannot_be_calibrated_away():
    noise = [0.0, 0.20, -0.18, 0.0, 0.22, -0.2, 0.15]
    r = analyze(_samples(yaw_noise=noise), MOUNT)
    assert r['corrected_yaw_spread_rad'] > math.radians(3)
    assert r['verdict'] == 'unreliable'


def test_camera_off_the_pan_axis_shows_up_as_position_spread():
    """A lens mounted ahead of the pan axis swings sideways as it pans. The
    model assumes the lens sits on the axis, so the base appears to move."""
    r = analyze(_samples(lens_forward_m=0.06), MOUNT)
    assert r['pan_gain'] == pytest.approx(1.0, abs=1e-6)
    assert 0.03 < r['base_pos_spread_m'] < 0.08
    assert r['base_yaw_spread_rad'] < 1e-6


def test_tilt_response_sign_is_measured():
    tilts = [0.0, 0.1, 0.2, 0.3]
    r = analyze(_samples(pans=[0.0], tilts=tilts), MOUNT)
    assert r['tilt_gain'] == pytest.approx(1.0, abs=1e-6)
    r = analyze(_samples(pans=[0.0], tilts=tilts, tilt_gain=-1.0), MOUNT)
    assert r['tilt_gain'] == pytest.approx(-1.0, abs=1e-6)
    assert r['tilt_inverted'] is True


def test_too_few_fixes_is_inconclusive():
    r = analyze(_samples(pans=[0.0]), MOUNT)
    assert r['verdict'] == 'inconclusive'
    assert analyze([], MOUNT)['verdict'] == 'inconclusive'


def test_jump_only_when_the_tag_changes_blames_the_tag_map_not_the_servo():
    """Field result: gain 1.01, but the base pose moved 22 cm between tags 2,
    3 and 4 while staying put within each. That is the tag map."""
    samples = _samples()
    shift = transforms.make_transform(transforms.rot_z(0.10), [0.0, 0.2, 0.0])
    for i, s in enumerate(samples):
        if i >= 4:                                  # these angles saw tag 4
            s['tag_ids'] = [4]
            s['T_map_cam'] = shift @ s['T_map_cam']
        else:
            s['tag_ids'] = [3]
    r = analyze(samples, MOUNT)
    assert r['base_pos_spread_m'] > 0.15
    assert r['within_tag_pos_spread_m'] < 1e-6
    assert r['within_tag_yaw_spread_rad'] < 1e-6
    assert r['verdict'] == 'tag_map'
    assert set(r['per_tag']) == {(3,), (4,)}


def test_tilt_fit_pools_pan_angles_when_straight_ahead_loses_the_tag():
    samples = [s for s in _samples(pans=[-0.8, 0.0, 0.8],
                                   tilts=[0.0, 0.1, 0.2])
               if not (s['pan_cmd'] == 0.0 and s['tilt_cmd'] > 0)]
    r = analyze(samples, MOUNT)
    assert r['tilt_gain'] == pytest.approx(1.0, abs=1e-6)


def test_heading_noise_from_a_face_on_tag_is_not_blamed_on_the_servo():
    """Field result: tag 3 straight ahead gave headings scattered over 10 deg
    while its position was fine. That is the tag's viewing angle, not the
    pan servo, so it must not turn the verdict to 'unreliable'."""
    samples = _samples()
    for i, s in enumerate(samples):
        s['incidence_rad'] = math.radians(55.0)
    for i, wobble in ((2, 0.12), (3, -0.10), (4, 0.09)):     # the middle angles
        s = samples[i]
        s['incidence_rad'] = math.radians(5.0)
        # heading wrong, position untouched: spin the camera about itself
        T = np.array(s['T_map_cam'])
        T[:3, :3] = transforms.rot_z(wobble) @ T[:3, :3]
        s['T_map_cam'] = T
        s['tag_ids'] = [3]
    r = analyze(samples, MOUNT)
    assert r['n_yaw_reliable'] == len(samples) - 3
    assert r['base_yaw_spread_rad'] < 1e-6
    assert r['base_pos_spread_m'] < 1e-6
    assert r['verdict'] == 'ok'
