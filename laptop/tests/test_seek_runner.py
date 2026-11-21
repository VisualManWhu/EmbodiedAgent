"""SlamRunner <-> SeekSession glue, with the Pi and the map stubbed out."""
import threading
import types

import pytest

pytest.importorskip('ultralytics')
pytest.importorskip('cv2')

from laptop_detector import SlamRunner                      # noqa: E402
from slam.nav_session import NavConfig, State               # noqa: E402
from slam.semantic_map import CONFIRMED                     # noqa: E402


class _Cursor:
    def take(self, status):
        return status or []

    def sync(self, status):
        pass


def _runner(landmarks=()):
    r = object.__new__(SlamRunner)
    r.session_nav = None
    r.seek = None
    r._class_hits = {}
    r._class_sight = {}
    r.aimer = None
    r._pan_mismatch_since = None
    r._pan_ignored = 0
    r._last_pan = 0.0
    r._deferred_done = None
    r.tag_map = {}
    r._nav_done_lock = threading.Lock()
    r._nav_done_event = None
    r.last_robot = (0.0, 0.0, 0.0)
    r.landmark_conf_min = 0.7
    r.nav_config = NavConfig()
    r.pi_events = _Cursor()
    r.smap = types.SimpleNamespace(landmarks={
        lm['id']: types.SimpleNamespace(
            id=lm['id'], label=lm['label'], state=CONFIRMED, confidence=1.0,
            position=(lm['x'], lm['y'], 0.0))
        for lm in landmarks})
    r.sent = []
    r._post_pi = r.sent.append
    r._get_pi_status = lambda: []
    return r


def _see(r, cls, now, n=3):
    r._record_hits([{'cls': cls}] * n, now)


def test_seek_with_map_landmark_starts_navigation_not_a_spin():
    r = _runner([{'id': 5, 'label': 'chair', 'x': 2.0, 'y': 0.0}])
    r.start_seek('chair')
    r._tick_seek(r.seek, [], 100.0)
    assert r.session_nav is not None
    assert r.session_nav.goal_id == 5
    assert r.sent == []                             # no `find` sent to the Pi


def test_sighting_during_navigation_hands_over_to_the_pi_visual_approach():
    r = _runner([{'id': 5, 'label': 'chair', 'x': 2.0, 'y': 0.0}])
    r.start_seek('chair')
    r._tick_seek(r.seek, [], 100.0)
    _see(r, 'chair', 104.9)
    r._tick_seek(r.seek, [], 105.0)
    assert r.session_nav is None                    # navigation cancelled
    assert r.sent == [{'action': 'find', 'target': 'chair'}]
    r._tick_seek(r.seek, ['arrived:chair'], 110.0)
    assert r.seek is None
    ev = r._nav_done_event
    assert ev['state'] == 'ARRIVED' and ev['via'] == 'vision'
    assert ev['goal_label'] == 'chair'


def test_map_arrival_without_sighting_looks_around_instead_of_reporting():
    r = _runner([{'id': 5, 'label': 'chair', 'x': 2.0, 'y': 0.0}])
    r.start_seek('chair')
    r._tick_seek(r.seek, [], 100.0)
    r.session_nav.state = State.ARRIVED
    r._publish_nav_done(130.0)
    assert r._nav_done_event is None                # nothing told to the bot
    assert r.sent == [{'action': 'find', 'target': 'chair'}]
    assert r.seek.unseen_ids == [5]


def test_seek_failure_stops_the_pi_and_reports_stale_landmark():
    r = _runner([{'id': 5, 'label': 'chair', 'x': 2.0, 'y': 0.0}])
    r.start_seek('chair')
    r._tick_seek(r.seek, [], 100.0)
    r.session_nav.state = State.ARRIVED
    r._publish_nav_done(130.0)
    r._tick_seek(r.seek, [], 181.0)                 # recheck spin exhausted
    assert r.sent[-1] == {'action': 'stop'}
    ev = r._nav_done_event
    assert ev['state'] == 'FAILED' and ev['reason'] == 'not_found'
    assert ev['unseen_ids'] == [5]
    assert r.seek is None


def test_plain_id_goto_reports_whether_the_object_was_seen_on_arrival():
    r = _runner([{'id': 5, 'label': 'chair', 'x': 2.0, 'y': 0.0}])
    r.start_goto((2.0, 0.0), 5, 'chair')
    r.session_nav.state = State.ARRIVED
    r._publish_nav_done(130.0)
    assert r._nav_done_event['seen'] is False
    assert r.sent == []                             # id goto never hands over

    r.start_goto((2.0, 0.0), 5, 'chair')
    _see(r, 'chair', 139.0, n=1)
    r.session_nav.state = State.ARRIVED
    r._publish_nav_done(140.0)
    assert r._nav_done_event['seen'] is True


def test_sightings_record_size_and_frame_edge_clipping():
    r = _runner()
    shape = (480, 640, 3)
    r._record_hits([{'cls': 'chair', 'cx': 320, 'cy': 300, 'w': 80, 'h': 120},
                    {'cls': 'chair', 'cx': 100, 'cy': 150, 'w': 200,
                     'h': 300}], 50.0, shape)
    s = r._class_sight['chair']                     # the biggest box wins
    assert s['t'] == 50.0 and abs(s['h_frac'] - 300 / 480) < 1e-9
    assert s['clipped'] is True                     # top edge at y = 0
    r._record_hits([{'cls': 'cup', 'cx': 320, 'cy': 240, 'w': 40, 'h': 60}],
                   51.0, shape)
    assert r._class_sight['cup']['clipped'] is False


def test_close_loss_arrival_stops_the_pi_and_reports_vision_arrival():
    r = _runner()
    shape = (480, 640, 3)
    big = [{'cls': 'chair', 'cx': 320, 'cy': 240, 'w': 300, 'h': 330}]
    for t in (100.0, 100.2, 100.4):
        r._record_hits(big, t, shape)
    r.start_seek('chair')
    r._tick_seek(r.seek, [], 100.5)                 # in view -> visual
    assert r.sent == [{'action': 'find', 'target': 'chair'}]
    r._tick_seek(r.seek, [], 104.0)                 # blind for 3.6 s, was close
    assert r.sent[-1] == {'action': 'stop'}
    assert r._nav_done_event['state'] == 'ARRIVED'
    assert r._nav_done_event['via'] == 'vision'


def test_stop_clears_an_active_seek():
    r = _runner()
    r.http = types.SimpleNamespace(post=lambda *a, **k: None)
    r.cmd_url = ''
    r.start_seek('chair')
    r._tick_seek(r.seek, [], 100.0)
    r.stop_goto()
    assert r.seek is None and r.session_nav is None


# ---- pan-tilt tag aiming -------------------------------------------------

AIM_TAGS = {2: {'x': 1.4, 'y': 0.0, 'yaw_deg': 90.0},
            3: {'x': 3.1, 'y': 0.0, 'yaw_deg': 90.0},
            4: {'x': 4.3, 'y': 0.0, 'yaw_deg': 90.0}}
NEAR_WALL = (3.1, 0.9, -1.5708)


def _aim_runner():
    from slam.tag_aimer import TagAimer
    r = _runner()
    r.tag_map = AIM_TAGS
    r.aimer = TagAimer(AIM_TAGS)
    r.start_goto((0.0, 1.0), None, 'tag0')
    return r


def _pans(r):
    return [c['yaw'] for c in r.sent if c.get('action') == 'pantilt']


def test_lost_pose_sweeps_the_camera_and_holds_the_chassis():
    r = _aim_runner()
    assert r._tick_aimer(100.0, None, True) is True          # hold
    assert _pans(r) == [0.0]                                 # sweep started
    assert r._tick_aimer(100.3, None, True) is True          # still sweeping
    r.aimer.sweep_exhausted = True
    r.aimer._cmd_t = -1e9
    assert r._tick_aimer(120.0, None, True) is False         # now turn chassis


def test_known_pose_aims_at_an_oblique_tag_then_releases_the_chassis():
    r = _aim_runner()
    r.session_nav.on_tag_fix(99.0)
    assert r._tick_aimer(100.0, NEAR_WALL, False) is True    # settling: hold
    assert len(_pans(r)) == 1 and abs(_pans(r)[0]) > 0.5
    assert r.aimer.target_id in (2, 4)
    assert r._tick_aimer(100.7, NEAR_WALL, False) is False   # settled: go


def test_camera_is_left_alone_while_a_pulse_is_in_flight():
    r = _aim_runner()
    r.session_nav.on_tag_fix(99.0)
    r.session_nav.state = State.WAITING_PULSE
    assert r._tick_aimer(100.0, NEAR_WALL, False) is False
    assert _pans(r) == []


def test_frames_taken_while_the_camera_swings_are_rejected():
    r = _aim_runner()
    r.session_nav.on_tag_fix(99.0)
    r._tick_aimer(100.0, NEAR_WALL, False)
    pan = r.aimer.pan_cmd
    assert r._frame_pan_valid(0.0, [], 100.1) is False       # old frame
    assert r._frame_pan_valid(pan, [], 100.3) is False       # servo moving
    assert r._frame_pan_valid(pan, [{'id': 4}], 100.8) is True
    assert r.aimer._seen_t == {4: 100.8}


def test_pi_that_ignores_pan_commands_disables_aiming_instead_of_stalling():
    r = _aim_runner()
    t = 100.0
    for _ in range(3):
        r.aimer.pan_cmd = 0.6                # commanded, never acknowledged
        r._frame_pan_valid(0.0, [], t)
        r._frame_pan_valid(0.0, [], t + 2.5)
        t += 10.0
    assert r.aimer is None
    assert r._frame_pan_valid(0.0, [], t) is True


def test_navigation_end_recentres_the_camera():
    r = _aim_runner()
    r.aimer.pan_cmd = 0.7
    r.session_nav.state = State.ARRIVED
    r._publish_nav_done(130.0)
    assert r.sent[-1] == {'action': 'pantilt', 'yaw': 0.0, 'pitch': 0.0}
    assert r.aimer.pan_cmd == 0.0
    # a failed goto sends `stop`, which re-centres on the Pi by itself
    r.start_goto((0.0, 1.0), None, 'tag0')
    r.aimer.pan_cmd = 0.7
    r.session_nav.state = State.FAILED
    r._publish_nav_done(140.0)
    assert r.sent[-1] == {'action': 'stop'}
    assert r.aimer.pan_cmd == 0.0


def test_result_is_held_back_until_the_camera_is_centred_again():
    """The bot photographs as soon as it sees the result; that photo must be
    taken looking forward, not at the side tag the camera was tracking."""
    r = _aim_runner()
    r.aimer.pan_cmd = 0.9
    r._last_pan = 0.9                       # camera still off to the side
    r.session_nav.state = State.ARRIVED
    r._publish_nav_done(130.0)
    assert r._nav_done_event is None
    r._last_pan = 0.9
    r._release_deferred_done(130.3)         # still swinging back
    assert r._nav_done_event is None
    r._last_pan = 0.0
    r._release_deferred_done(130.5)         # centred, but only just
    assert r._nav_done_event is None
    r._release_deferred_done(131.0)
    assert r._nav_done_event['state'] == 'ARRIVED'


def test_result_is_never_held_forever():
    r = _aim_runner()
    r._last_pan = 0.9
    r.session_nav.state = State.ARRIVED
    r._publish_nav_done(130.0)
    r._release_deferred_done(134.5)         # camera never reported centred
    assert r._nav_done_event is not None
