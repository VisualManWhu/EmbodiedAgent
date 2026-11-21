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
