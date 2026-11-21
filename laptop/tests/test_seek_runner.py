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
    r.grid = None
    r._prev_fix = None
    r._last_block = None
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


# ---- traversability grid ---------------------------------------------------

ROOM = {0: {'x': 0.0, 'y': 0.0}, 1: {'x': 4.0, 'y': 0.0},
        2: {'x': 4.0, 'y': 3.0}, 3: {'x': 0.0, 'y': 3.0}}


def _grid_runner(goal=(3.5, 1.5)):
    from slam.traverse_grid import TraverseGrid
    r = _runner()
    r.tag_map = ROOM
    r.grid = TraverseGrid.from_tags(ROOM, spacing_m=0.25)
    r.nav_config.replan_on_block = True
    r.start_goto(goal, None, 'spot')
    r.session_nav.on_tag_fix(99.0)
    return r


def test_fixes_teach_the_grid_where_the_robot_has_been():
    from slam.traverse_grid import SAFE, UNKNOWN
    r = _grid_runner()
    r._grid_note_fix(0.5, 1.5, 100.0)
    r._grid_note_fix(1.0, 1.5, 103.0)               # 0.5 m on, 3 s later
    assert r.grid.state_at((0.75, 1.5)) == SAFE     # the stretch between
    r._grid_note_fix(3.0, 2.5, 104.0)               # a jump: not a drive
    assert r.grid.state_at((3.0, 2.5)) == SAFE
    assert r.grid.state_at((2.0, 2.0)) == UNKNOWN


def test_goto_is_planned_once_the_pose_is_known_and_follows_the_route():
    r = _grid_runner()
    x = 2.0
    y = 0.75
    while y <= 2.25:                                # danger across the middle
        r.grid.mark_danger_at((x, y))
        y += 0.25
    r._plan_route_if_needed(None, True)             # no pose yet: no plan
    assert r.session_nav.route_ready is False
    r._plan_route_if_needed((0.5, 1.5, 0.0), False)
    assert r.session_nav.route_ready is True
    assert len(r.session_nav.waypoints) >= 1        # it goes round
    wps = list(r.session_nav.waypoints)
    r._plan_route_if_needed((0.5, 1.5, 0.0), False) # not re-planned every tick
    assert r.session_nav.waypoints == wps


def test_block_marks_the_obstacle_and_replans_around_it():
    from slam.traverse_grid import DANGER
    r = _grid_runner()
    r._plan_route_if_needed((0.5, 1.5, 0.0), False)
    # nothing known yet: one straight leg to a node within reach of the goal
    assert len(r.session_nav.waypoints) == 1
    assert abs(r.session_nav.waypoints[0][1] - 1.5) < 1e-9
    r._nav_pose_dbg = ((1.0, 1.5, 0.0), False, False)
    r._feed_nav_events(['blocked:front'], 110.0)
    assert r.grid.state_at((1.5, 1.5)) == DANGER    # the node ahead
    assert r.session_nav.needs_replan is True
    r._plan_route_if_needed((0.9, 1.5, 0.0), False)
    assert r.session_nav.needs_replan is False
    assert len(r.session_nav.waypoints) >= 2        # now it goes round


def test_a_few_blocks_mark_obstacles_but_not_the_floor_the_robot_is_on():
    """Three blocks used to paint the robot's own patch danger. Feeling its
    way along a wall near the goal tripped that immediately."""
    from slam.traverse_grid import DANGER
    r = _grid_runner()
    r._plan_route_if_needed((0.5, 1.5, 0.0), False)
    for i in range(3):
        r._nav_pose_dbg = ((1.0, 1.5, 0.0), False, False)
        r._feed_nav_events(['blocked:front'], 110.0 + i)
    assert r.grid.state_at((1.5, 1.5)) == DANGER    # the obstacle ahead
    assert r.grid.state_at((1.0, 1.5)) != DANGER    # not where it stands


def test_goto_that_fails_on_blocks_marks_the_patch_as_a_trap():
    from slam.traverse_grid import DANGER
    r = _grid_runner()
    r.session_nav.config.max_blocks = 2
    r._plan_route_if_needed((0.5, 1.5, 0.0), False)
    for i in range(3):
        if r.session_nav is None:
            break
        r._nav_pose_dbg = ((1.0, 1.5, 0.0), False, False)
        r._feed_nav_events(['blocked:front'], 110.0 + i)
    assert r.session_nav is None                    # the goto gave up
    assert r._nav_done_event['reason'] == 'blocked'
    assert r.grid.state_at((1.0, 1.5)) == DANGER    # stuck here


def test_no_route_fails_the_goto_with_a_clear_reason():
    r = _grid_runner()
    y = 0.0
    while y <= 3.0:
        r.grid.mark_danger_at((2.0, y))             # wall to wall
        y += 0.25
    r._plan_route_if_needed((0.5, 1.5, 0.0), False)
    assert r.session_nav.state is State.FAILED
    assert r.session_nav.fail_reason == 'no_path'
    r._publish_nav_done(120.0)
    assert r._nav_done_event['reason'] == 'no_path'


def test_operator_stop_right_after_a_block_marks_the_area_danger():
    import time as _time
    from slam.traverse_grid import DANGER, UNKNOWN
    r = _grid_runner()
    r._last_block = (_time.time() - 5.0, (2.0, 2.0, 0.0))
    r.stop_goto()
    assert r.grid.state_at((2.0, 2.0)) == DANGER
    # ...but a stop long after any block marks nothing
    r2 = _grid_runner()
    r2._last_block = (_time.time() - 300.0, (2.0, 2.0, 0.0))
    r2.stop_goto()
    assert r2.grid.state_at((2.0, 2.0)) == UNKNOWN


def test_arriving_at_the_goal_by_touching_it_is_not_recorded_as_an_obstacle():
    from slam.traverse_grid import DANGER
    r = _grid_runner(goal=(1.5, 1.5))
    r._plan_route_if_needed((0.5, 1.5, 0.0), False)
    r.session_nav.tick((1.0, 1.5, 0.0), 100.0)      # 0.5 m from the goal
    r._nav_pose_dbg = ((1.0, 1.5, 0.0), False, False)
    r._feed_nav_events(['blocked:front'], 110.0)
    assert r._nav_done_event['state'] == 'ARRIVED'
    assert DANGER not in {n['state'] for n in r.grid.snapshot()}


def test_clean_forward_pulse_marks_the_dead_reckoned_stretch_safe():
    from slam.traverse_grid import SAFE, UNKNOWN
    r = _grid_runner()
    r._grid_note_fix(0.5, 1.5, 100.0)               # last tag fix
    r._last_nav_kind = 'forward'
    r._nav_pose_dbg = ((0.85, 1.5, 0.0), False, False)   # dead-reckoned
    r._feed_nav_events(['pulse_done'], 103.0)
    assert r.grid.state_at((0.75, 1.5)) == SAFE
    # a rotate pulse, or a stale pose, teaches nothing
    r._last_nav_kind = 'rotate'
    r._nav_pose_dbg = ((1.5, 1.5, 0.0), False, False)
    r._feed_nav_events(['pulse_done'], 104.0)
    r._last_nav_kind = 'forward'
    r._nav_pose_dbg = ((1.5, 1.5, 0.0), False, True)
    r._feed_nav_events(['pulse_done'], 105.0)
    assert r.grid.state_at((1.5, 1.5)) == UNKNOWN


def test_block_with_an_untrusted_pose_marks_nothing():
    from slam.traverse_grid import DANGER
    r = _grid_runner()
    r._plan_route_if_needed((0.5, 1.5, 0.0), False)
    r._nav_pose_dbg = ((1.0, 1.5, 0.0), False, True)     # stale pose
    r._feed_nav_events(['blocked:front'], 110.0)
    assert DANGER not in {n['state'] for n in r.grid.snapshot()}
    assert r.session_nav.needs_replan is True            # still re-plans
