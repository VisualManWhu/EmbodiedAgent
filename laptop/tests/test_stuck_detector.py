import numpy as np
import pytest

from slam.stuck_detector import StuckCheck, frame_signature, similarity


def _scene(seed=0):
    """A textured view: carpet-like noise plus a few large shapes."""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 180, (480, 640), dtype=np.uint8)
    img = np.repeat(np.repeat(img[::4, ::4], 4, axis=0), 4, axis=1)
    img[100:220, 200:330] = 230
    img[300:480, :] = (img[300:480, :] * 0.6).astype(np.uint8)
    return img


def _noisy(img, sigma=4.0, seed=1):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32)
                   + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _shift(img, dx, dy):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def test_same_view_with_sensor_noise_correlates_and_a_moved_one_does_not():
    base = _scene()
    same, texture = similarity(frame_signature(base),
                               frame_signature(_noisy(base)))
    moved, _ = similarity(frame_signature(base),
                          frame_signature(_shift(_noisy(base), 14, 6)))
    assert texture > StuckCheck.MIN_TEXTURE
    assert same > 0.8
    assert moved < 0.4
    assert same >= StuckCheck.SAME_VIEW_NCC > moved


def test_bland_view_reports_no_texture():
    flat = np.full((480, 640), 128, np.uint8)
    ncc, texture = similarity(frame_signature(flat), frame_signature(flat))
    assert texture < StuckCheck.MIN_TEXTURE


def _check(pre_fix=None, expected=0.45):
    return StuckCheck(100.0, expected, (1.0, 1.0, 0.0),
                      frame_signature(_scene()), 0.0, pre_fix=pre_fix)


def test_unchanged_picture_after_the_pulse_means_stuck():
    """Field: wedged on a chair leg, the ultrasonic saw nothing; the pulse
    ran its 2.5 s and the robot had moved 9 cm."""
    c = _check()
    assert c.evaluate(101.0, None, 0.0, True) is None       # pulse still going
    c.on_done(101.8)
    after = frame_signature(_noisy(_scene()))
    assert c.evaluate(101.9, after, 0.0, True) is None      # let it settle
    assert c.evaluate(102.2, after, 0.0, True) == 'stuck'
    assert c.how == 'image' and c.ncc > 0.8


def test_changed_picture_means_it_moved():
    c = _check()
    c.on_done(101.8)
    after = frame_signature(_shift(_noisy(_scene()), 14, 6))
    assert c.evaluate(102.2, after, 0.0, True) == 'moved'


def test_a_tag_fix_overrules_the_picture():
    # picture unchanged (say, facing a blank far wall) but the fix shows the
    # robot did cover the distance
    c = _check(pre_fix=(1.0, 1.0, 0.0))
    c.on_done(101.8)
    same = frame_signature(_noisy(_scene()))
    assert c.evaluate(102.2, same, 0.0, True) is None       # waits for a fix
    assert c.evaluate(102.6, same, 0.0, True, fix=(1.42, 1.0, 0.0)) == 'moved'
    assert c.how == 'fix' and c.moved_m == pytest.approx(0.42)


def test_a_tag_fix_showing_no_progress_means_stuck():
    c = _check(pre_fix=(1.0, 1.0, 0.0))
    c.on_done(101.8)
    assert c.evaluate(102.5, None, 0.0, False,
                      fix=(1.08, 1.0, 0.0)) == 'stuck'
    assert c.how == 'fix' and c.moved_m == pytest.approx(0.08)


def test_picture_alone_decides_if_no_fix_turns_up_in_time():
    c = _check(pre_fix=(1.0, 1.0, 0.0))
    c.on_done(101.8)
    same = frame_signature(_noisy(_scene()))
    assert c.evaluate(102.2, same, 0.0, True) is None
    assert c.evaluate(103.5, None, 0.5, True) == 'stuck'    # 1.7 s, no fix
    assert c.how == 'image'


def test_displacement_is_measured_along_the_heading_at_the_start():
    c = StuckCheck(100.0, 0.45, (1.0, 1.0, np.pi / 2),
                   frame_signature(_scene()), 0.0,
                   pre_fix=(1.0, 1.0, np.pi / 2))               # facing +y
    c.on_done(101.8)
    # shoved 0.4 m sideways, no forward progress: still stuck
    assert c.evaluate(102.5, None, 0.0, False,
                      fix=(1.4, 1.02, np.pi / 2)) == 'stuck'


def test_no_usable_evidence_assumes_it_moved():
    """Never call it stuck on no evidence: that paints safe floor red."""
    flat = np.full((480, 640), 128, np.uint8)
    c = StuckCheck(100.0, 0.45, (1.0, 1.0, 0.0), frame_signature(flat), 0.0)
    c.on_done(101.8)
    assert c.evaluate(102.2, frame_signature(flat), 0.0, True) == 'moved'
    assert c.how == 'unknown'
    # camera was panned away before the 'after' picture could be taken
    c2 = _check()
    c2.on_done(101.8)
    assert c2.evaluate(102.2, frame_signature(_scene()), 0.7, True) is None
    assert c2.evaluate(105.0, frame_signature(_scene()), 0.7, True) == 'moved'


def test_camera_is_held_until_the_after_picture_is_taken():
    c = _check()
    assert c.wants_camera_still(101.0) is False     # pulse not finished
    c.on_done(101.8)
    assert c.wants_camera_still(101.9) is True
    c.evaluate(102.2, frame_signature(_noisy(_scene())), 0.0, True)
    assert c.wants_camera_still(102.3) is False
