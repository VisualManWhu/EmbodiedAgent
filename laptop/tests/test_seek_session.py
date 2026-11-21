from slam.seek_session import SeekConfig, SeekSession, SeekState

CHAIR_A = {'id': 7, 'x': 1.0, 'y': 0.0}
CHAIR_B = {'id': 9, 'x': 3.0, 'y': 0.0}


def _confirmed_hits(now):
    return [now - 0.6, now - 0.3, now - 0.1]


def test_target_in_view_goes_straight_to_visual_even_if_map_has_it():
    s = SeekSession('chair')
    actions = s.tick(100.0, _confirmed_hits(100.0), [CHAIR_A])
    assert actions == [('find', 'chair')]
    assert s.state is SeekState.VISUAL


def test_not_in_view_and_map_has_it_navigates_directly():
    s = SeekSession('chair')
    actions = s.tick(100.0, [], [CHAIR_A, CHAIR_B])
    assert actions == [('goto', CHAIR_A)]           # nearest first, no spin
    assert s.state is SeekState.NAV
    assert s.goal_id == 7


def test_not_in_view_and_map_empty_spin_searches():
    s = SeekSession('chair')
    assert s.tick(100.0, [], []) == [('find', 'chair')]
    assert s.state is SeekState.VISUAL


def test_vision_overrides_navigation_once_confirmed():
    s = SeekSession('chair')
    s.tick(100.0, [], [CHAIR_A])
    actions = s.tick(105.0, _confirmed_hits(105.0), [CHAIR_A])
    assert actions == [('nav_cancel',), ('find', 'chair')]
    assert s.state is SeekState.VISUAL


def test_single_stray_detection_does_not_hijack_navigation():
    s = SeekSession('chair')
    s.tick(100.0, [], [CHAIR_A])
    assert s.tick(105.0, [104.9], [CHAIR_A]) == []
    assert s.tick(109.0, [104.9, 108.9], [CHAIR_A]) == []   # too far apart
    assert s.state is SeekState.NAV


def test_visual_arrival_event_finishes_the_seek():
    s = SeekSession('chair')
    s.tick(100.0, _confirmed_hits(100.0), [])
    s.on_pi_event('arrived:bottle')                 # some other class
    assert not s.done
    s.on_pi_event('arrived:chair')
    assert s.state is SeekState.ARRIVED
    assert s.via == 'vision'


def test_arrival_event_ignored_while_navigating():
    s = SeekSession('chair')
    s.tick(100.0, [], [CHAIR_A])
    s.on_pi_event('arrived:chair')
    assert s.state is SeekState.NAV


def test_vision_lost_hands_back_to_map_with_hysteresis():
    cfg = SeekConfig(visual_lost_sec=12.0)
    s = SeekSession('chair', cfg)
    s.tick(100.0, _confirmed_hits(100.0), [CHAIR_A])
    # still inside the lost window -> stay visual
    assert s.tick(108.0, [99.9], [CHAIR_A]) == []
    assert s.state is SeekState.VISUAL
    # a new sighting extends the window
    assert s.tick(115.0, [110.0], [CHAIR_A]) == []
    # 12 s after the last sighting -> back to the map
    assert s.tick(122.5, [110.0], [CHAIR_A]) == [('goto', CHAIR_A)]
    assert s.state is SeekState.NAV


def test_map_arrival_with_target_in_sight_is_success():
    s = SeekSession('chair')
    s.tick(100.0, [], [CHAIR_A])
    assert s.on_nav_done('ARRIVED', 130.0, [129.0]) == []
    assert s.state is SeekState.ARRIVED
    assert s.via == 'map'
    assert s.goal_id == 7


def test_map_arrival_without_sighting_is_not_success_looks_around_then_next():
    cfg = SeekConfig(recheck_sec=50.0)
    s = SeekSession('chair', cfg)
    s.tick(100.0, [], [CHAIR_A, CHAIR_B])
    actions = s.on_nav_done('ARRIVED', 130.0, [])
    assert actions == [('find', 'chair')]           # look around here first
    assert s.state is SeekState.VISUAL
    assert s.unseen_ids == [7]
    assert s.tick(160.0, [], [CHAIR_A, CHAIR_B]) == []
    # nothing found here -> the next untried landmark, never id 7 again
    assert s.tick(181.0, [], [CHAIR_A, CHAIR_B]) == [('goto', CHAIR_B)]
    assert s.goal_id == 9


def test_nav_failure_falls_back_to_visual_search():
    s = SeekSession('chair')
    s.tick(100.0, [], [CHAIR_A])
    assert s.on_nav_done('FAILED', 140.0, []) == [('find', 'chair')]
    assert s.state is SeekState.VISUAL
    assert s.unseen_ids == []


def test_everything_exhausted_fails_and_stops_the_pi_search():
    cfg = SeekConfig(search_sec=90.0)
    s = SeekSession('chair', cfg)
    s.tick(100.0, [], [])
    assert s.tick(189.0, [], []) == []
    assert s.tick(191.0, [], []) == [('stop',)]
    assert s.state is SeekState.FAILED
    assert s.fail_reason == 'not_found'
    assert s.tick(200.0, [], []) == []              # terminal: no more actions


def test_blind_search_extends_on_sighting_then_reports_arrival():
    s = SeekSession('chair')
    s.tick(100.0, [], [])
    assert s.tick(188.0, [187.5], []) == []
    assert s.tick(195.0, [187.5], []) == []         # 187.5 + 12 > 195
    s.on_pi_event('arrived:chair')
    assert s.state is SeekState.ARRIVED


def test_total_timeout_cancels_navigation():
    cfg = SeekConfig(total_timeout_sec=60.0)
    s = SeekSession('chair', cfg)
    s.tick(100.0, [], [CHAIR_A])
    assert s.tick(161.0, [], [CHAIR_A]) == [('nav_cancel',), ('stop',)]
    assert s.fail_reason == 'timeout'


def test_late_nav_done_after_vision_took_over_is_ignored():
    s = SeekSession('chair')
    s.tick(100.0, [], [CHAIR_A])
    s.tick(105.0, _confirmed_hits(105.0), [CHAIR_A])
    assert s.on_nav_done('FAILED', 106.0, []) == []
    assert s.state is SeekState.VISUAL
    assert 7 not in s.tried


def _sight(t, h_frac, clipped=False):
    return {'t': t, 'h_frac': h_frac, 'clipped': clipped}


def test_going_blind_right_next_to_the_target_counts_as_arrival():
    """The camera is too low to recognise a chair from 30 cm: detection drops
    out at the end of the approach and the Pi never reports arrival."""
    s = SeekSession('chair')
    s.tick(100.0, _confirmed_hits(100.0), [], _sight(99.9, 0.30))
    assert s.tick(104.0, [103.9], [], _sight(103.9, 0.58)) == []
    assert s.tick(105.0, [103.9], [], _sight(103.9, 0.58)) == []   # 1.1 s
    assert s.tick(107.0, [103.9], [], _sight(103.9, 0.58)) == [('stop',)]
    assert s.trail[-1] == 'arrived:close'
    assert s.state is SeekState.ARRIVED
    assert s.via == 'vision'


def test_frame_edge_clipping_alone_is_not_closeness():
    """Field result: h=0.46 clipped was a chair 1.3 m away, not an arrival."""
    s = SeekSession('chair')
    s.tick(100.0, _confirmed_hits(100.0), [], _sight(99.9, 0.40, True))
    assert s.tick(107.0, [103.9], [], _sight(103.9, 0.46, True)) == []
    assert s.state is SeekState.VISUAL


def test_live_sighting_big_enough_is_an_arrival_without_waiting_for_the_pi():
    s = SeekSession('chair')
    s.tick(100.0, _confirmed_hits(100.0), [], _sight(99.9, 0.30))
    assert s.tick(103.0, [102.9], [], _sight(102.9, 0.60)) == []
    assert s.tick(104.0, [103.9], [], _sight(103.9, 0.64)) == [('stop',)]
    assert s.state is SeekState.ARRIVED
    assert s.trail[-1] == 'arrived:near'


def test_losing_a_distant_target_is_not_an_arrival():
    s = SeekSession('chair')
    s.tick(100.0, _confirmed_hits(100.0), [CHAIR_A], _sight(99.9, 0.20))
    assert s.tick(107.0, [99.9], [CHAIR_A], _sight(99.9, 0.20)) == []
    assert s.state is SeekState.VISUAL


def test_old_close_sighting_does_not_end_a_fresh_blind_search():
    s = SeekSession('chair')
    old = _sight(40.0, 0.80)                        # from a minute ago
    assert s.tick(100.0, [], [], old) == [('find', 'chair')]
    assert s.tick(110.0, [], [], old) == []
    assert s.state is SeekState.VISUAL


def test_map_arrival_seen_window_tolerates_close_range_blindness():
    s = SeekSession('chair')
    s.tick(100.0, [], [CHAIR_A])
    s.on_nav_done('ARRIVED', 130.0, [122.0])        # last seen 8 s ago
    assert s.state is SeekState.ARRIVED and s.via == 'map'
