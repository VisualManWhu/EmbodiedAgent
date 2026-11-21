import math

import pytest

from slam.traverse_grid import (DANGER, NEAR, SAFE, UNKNOWN, TraverseGrid,
                                convex_hull)

# 4 m x 3 m room; tags on the walls, one in the interior
TAGS = {0: {'x': 0.0, 'y': 0.0}, 1: {'x': 4.0, 'y': 0.0},
        2: {'x': 4.0, 'y': 3.0}, 3: {'x': 0.0, 'y': 3.0},
        4: {'x': 2.0, 'y': 1.0}}


def _grid(**kw):
    return TraverseGrid.from_tags(TAGS, spacing_m=0.25, **kw)


def _block_wall(g, x, y0, y1):
    """Mark a vertical run of nodes dangerous (an obstacle across the room)."""
    y = y0
    while y <= y1 + 1e-9:
        g.mark_danger_at((x, y))
        y += g.spacing


def _len(path):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def test_convex_hull_drops_interior_points():
    hull = convex_hull([(0, 0), (4, 0), (4, 3), (0, 3), (2, 1)])
    assert sorted(hull) == [(0, 0), (0, 3), (4, 0), (4, 3)]


def test_nodes_cover_the_tag_bounding_rectangle():
    g = _grid()
    xs = [n['x'] for n in g.snapshot()]
    ys = [n['y'] for n in g.snapshot()]
    assert min(xs) >= 0.0 and max(xs) <= 4.0
    assert min(ys) >= 0.0 and max(ys) <= 3.0
    assert len(g.snapshot()) == 17 * 13                 # 0.25 m spacing


def test_rectangle_keeps_the_corner_a_missing_tag_would_cut_off():
    """Field layout: no tag in one corner of the room. The convex hull drops
    that whole corner — which is where the impassable area was."""
    tags = {0: {'x': 0, 'y': 0}, 4: {'x': 4.3, 'y': -0.5},
            5: {'x': 5.3, 'y': 1.2}, 6: {'x': 0.2, 'y': 2.8}}
    corner = (4.5, 2.5)
    assert TraverseGrid.from_tags(tags).state_at(corner) == UNKNOWN
    assert TraverseGrid.from_tags(tags, shape='hull').state_at(corner) is None


def test_everything_starts_unknown():
    g = _grid()
    assert {n['state'] for n in g.snapshot()} == {UNKNOWN}
    assert g.state_at((1.0, 1.0)) == UNKNOWN
    assert g.state_at((9.0, 9.0)) is None               # off the grid


def test_driven_segment_marks_every_node_along_it_safe():
    g = _grid()
    g.mark_path((0.5, 1.0), (1.5, 1.0))
    for x in (0.5, 0.75, 1.0, 1.25, 1.5):
        assert g.state_at((x, 1.0)) == SAFE
    assert g.state_at((1.75, 1.0)) == UNKNOWN
    assert g.state_at((1.0, 1.25)) == UNKNOWN


def test_blocked_marks_the_node_ahead_not_the_one_the_robot_is_on():
    g = _grid()
    g.mark_path((1.0, 1.0), (1.0, 1.0))
    g.mark_blocked((1.0, 1.0, 0.0), ahead_m=0.35)       # facing +x
    assert g.state_at((1.0, 1.0)) == NEAR               # passable, but close
    assert g.state_at((1.25, 1.0)) == DANGER
    # default reach: sensor offset + trigger range, measured from the centre
    g2 = _grid()
    assert g2.mark_blocked((1.0, 1.0, 0.0)) == (1.5, 1.0)
    assert g.state_at((1.5, 1.0)) == NEAR               # next to danger
    assert g.state_at((1.0, 1.25)) == NEAR
    assert g.state_at((2.0, 1.0)) == UNKNOWN


def test_trapped_marks_the_whole_patch_including_where_the_robot_stands():
    g = _grid()
    g.mark_trapped((2.0, 2.0, 0.0), radius_m=0.4)
    assert g.state_at((2.0, 2.0)) == DANGER
    assert g.state_at((2.25, 2.0)) == DANGER
    assert g.state_at((2.25, 2.25)) == DANGER
    assert g.state_at((2.5, 2.0)) == NEAR
    assert g.state_at((3.0, 2.0)) == UNKNOWN


def test_danger_clears_after_the_robot_drives_through_twice():
    """Furniture moves. A node that has since been crossed is not dangerous."""
    g = _grid()
    g.mark_danger_at((1.0, 1.0))
    g.mark_path((1.0, 1.0), (1.0, 1.0))
    assert g.state_at((1.0, 1.0)) == DANGER             # once is not enough
    g.mark_path((0.5, 1.0), (1.0, 1.0))
    assert g.state_at((1.0, 1.0)) == SAFE


def test_standing_still_does_not_count_as_repeated_passes():
    g = _grid()
    g.mark_danger_at((1.0, 1.0))
    for _ in range(10):
        g.mark_path((1.0, 1.0), (1.0, 1.0))
    assert g.state_at((1.0, 1.0)) == DANGER


def test_plan_in_unknown_space_is_a_straight_line():
    g = _grid()
    path = g.plan((0.5, 1.5), (3.5, 1.5))
    assert path[-1] == (3.5, 1.5)
    assert len(path) == 1                               # nothing to go round
    assert g.last_plan['unknown_frac'] == pytest.approx(1.0)


def test_plan_goes_round_a_danger_zone():
    g = _grid()
    _block_wall(g, 2.0, 0.75, 2.25)                     # wall across the middle
    path = g.plan((0.5, 1.5), (3.5, 1.5))
    assert path is not None and path[-1] == (3.5, 1.5)
    full = [(0.5, 1.5)] + path
    # never touches a danger node
    for a, b in zip(full, full[1:]):
        for k in range(21):
            p = (a[0] + (b[0] - a[0]) * k / 20, a[1] + (b[1] - a[1]) * k / 20)
            assert g.state_at(p) != DANGER
    assert _len(full) > 3.0                             # it had to detour


def test_plan_prefers_a_longer_known_safe_route_over_a_short_unknown_one():
    g = _grid()
    # a safe dog-leg: along y = 0.5
    g.mark_path((0.5, 1.5), (0.5, 0.5))
    g.mark_path((0.5, 0.5), (3.5, 0.5))
    g.mark_path((3.5, 0.5), (3.5, 1.5))
    path = [(0.5, 1.5)] + g.plan((0.5, 1.5), (3.5, 1.5))
    assert min(p[1] for p in path) == pytest.approx(0.5)    # took the dog-leg
    assert g.last_plan['unknown_frac'] < 0.1
    # ...but not at any price: with unknown barely penalised it cuts across
    cheap = _grid(unknown_weight=1.05)
    cheap.mark_path((0.5, 1.5), (0.5, 0.5))
    cheap.mark_path((0.5, 0.5), (3.5, 0.5))
    cheap.mark_path((3.5, 0.5), (3.5, 1.5))
    assert len(cheap.plan((0.5, 1.5), (3.5, 1.5))) == 1


def test_plan_gives_danger_a_wide_berth_when_there_is_room():
    g = _grid()
    g.mark_danger_at((2.0, 1.5))
    path = [(0.5, 1.5)] + g.plan((0.5, 1.5), (3.5, 1.5))
    closest = min(
        math.dist((a[0] + (b[0] - a[0]) * k / 40,
                   a[1] + (b[1] - a[1]) * k / 40), (2.0, 1.5))
        for a, b in zip(path, path[1:]) for k in range(41))
    assert closest >= 0.3                   # more than one node away
    # with no penalty for skirting danger it shaves the corner closer
    tight = _grid(near_penalty=0.0)
    tight.mark_danger_at((2.0, 1.5))
    tpath = [(0.5, 1.5)] + tight.plan((0.5, 1.5), (3.5, 1.5))
    assert _len(tpath) <= _len(path) + 1e-9


def test_no_route_except_through_danger_returns_none():
    g = _grid()
    _block_wall(g, 2.0, 0.0, 3.0)                       # wall to wall
    assert g.plan((0.5, 1.5), (3.5, 1.5)) is None


def test_goal_inside_danger_is_unreachable_but_start_inside_can_leave():
    g = _grid()
    g.mark_trapped((2.0, 1.5, 0.0), radius_m=0.4)
    assert g.plan((0.5, 1.5), (2.0, 1.5)) is None
    out = g.plan((2.0, 1.5), (0.5, 1.5))                # the way out
    assert out is not None and out[-1] == (0.5, 1.5)
    # leaving its own patch is allowed; crossing a DIFFERENT one is not
    _block_wall(g, 1.0, 0.0, 3.0)
    assert g.plan((2.0, 1.5), (0.5, 1.5)) is None


def test_escape_route_follows_the_way_the_robot_came_in():
    """Trapped: everything around is danger except the nodes it drove in on."""
    g = _grid()
    g.mark_path((0.5, 1.5), (2.0, 1.5))                 # drove in along y=1.5
    g.mark_trapped((2.0, 1.5, 0.0), radius_m=0.6)
    # the entry corridor was driven: passes after the mark clear it
    g.mark_path((2.0, 1.5), (1.5, 1.5))
    g.mark_path((1.5, 1.5), (2.0, 1.5))
    g.mark_path((2.0, 1.5), (1.5, 1.5))
    path = g.plan((2.0, 1.5), (0.5, 1.5))
    assert path is not None
    # it leaves through the corridor it re-drove, i.e. westwards along y=1.5
    assert path[0][0] < 2.0 and abs(path[0][1] - 1.5) < 0.3


def test_state_survives_a_save_and_load(tmp_path):
    g = _grid()
    g.mark_path((0.5, 1.0), (1.5, 1.0))
    g.mark_blocked((1.5, 1.0, 0.0), ahead_m=0.35)
    f = tmp_path / 'grid.json'
    g.save(str(f))
    g2 = TraverseGrid.from_tags(TAGS, spacing_m=0.25)
    assert g2.load(str(f)) is True
    assert g2.state_at((1.0, 1.0)) == SAFE
    assert g2.state_at((1.75, 1.0)) == DANGER
    assert g2.state_at((3.0, 2.0)) == UNKNOWN


def test_grid_saved_for_a_different_layout_is_ignored(tmp_path):
    g = _grid()
    g.mark_path((0.5, 1.0), (1.5, 1.0))
    f = tmp_path / 'grid.json'
    g.save(str(f))
    other = TraverseGrid.from_tags(TAGS, spacing_m=0.5)
    assert other.load(str(f)) is False
    assert other.state_at((1.0, 1.0)) == UNKNOWN
    assert other.load(str(tmp_path / 'missing.json')) is False


def test_counts_summarise_the_grid():
    g = _grid()
    g.mark_path((0.5, 1.0), (1.0, 1.0))
    g.mark_danger_at((3.0, 2.0))
    c = g.counts()
    assert c[SAFE] == 3 and c[DANGER] == 1 and c[NEAR] == 8
    assert sum(c.values()) == 17 * 13


def test_way_out_of_a_stuck_patch_never_goes_through_a_sensed_obstacle():
    """Field bug: blocked three times by one obstacle, the robot's own patch
    was marked, merged with the obstacle nodes, and the escape route was
    planned straight through them."""
    g = _grid()
    for y in (1.0, 1.25, 1.5, 1.75, 2.0):
        g.mark_danger_at((1.5, y))                      # obstacle, west side
    g.mark_trapped((2.0, 1.5, 0.0), radius_m=0.5)       # robot stuck beside it
    assert g.nodes[g._key((1.5, 1.5))]['kind'] == 'obstacle'    # not relabelled
    path = [(2.0, 1.5)] + g.plan((2.0, 1.5), (0.5, 1.5))
    for a, b in zip(path, path[1:]):
        for k in range(41):
            p = (a[0] + (b[0] - a[0]) * k / 40, a[1] + (b[1] - a[1]) * k / 40)
            key = g._key(p)
            assert g.nodes[key]['kind'] != 'obstacle', p
    assert _len(path) > 1.6                             # had to go round


def test_route_keeps_a_node_clear_of_an_obstacle_in_unknown_space():
    """Field result: the re-planned route hugged the node next to the one
    just marked, and the robot hit the same obstacle again."""
    g = _grid()
    g.mark_danger_at((2.0, 1.5))
    path = [(0.5, 1.5)] + g.plan((0.5, 1.5), (3.5, 1.5))
    for a, b in zip(path, path[1:]):
        for k in range(41):
            p = (a[0] + (b[0] - a[0]) * k / 40, a[1] + (b[1] - a[1]) * k / 40)
            assert g.state_at(p) not in (DANGER, NEAR), p


def test_goal_in_a_wall_corner_is_reached_by_getting_close_enough():
    """Field bug: the tag-0 standoff is 0.1 m from the side wall. Its nearest
    node lies on the wall line, so every route ran along the wall and the
    robot fought it for four minutes."""
    g = _grid()
    for y in (0.5, 0.75, 1.0, 1.25, 1.5):
        g.mark_danger_at((0.0, y))                      # the wall itself
    goal = (0.1, 1.05)
    assert g.plan((2.0, 1.0), goal) is None             # exact node: in the wall
    route = g.plan((2.0, 1.0), goal, goal_radius_m=0.32)
    assert route is not None
    end = route[-1]
    assert math.dist(end, goal) <= 0.32 + 1e-9
    assert g.state_at(end) != DANGER
    assert end[0] >= 0.25                               # stops short of the wall


def test_goal_radius_fails_only_when_nothing_nearby_is_reachable():
    g = _grid()
    g.mark_trapped((2.0, 1.5, 0.0), radius_m=0.6)
    assert g.plan((0.5, 1.5), (2.0, 1.5), goal_radius_m=0.3) is None
    assert g.plan((0.5, 1.5), (2.0, 1.5), goal_radius_m=0.9) is not None


def test_already_within_the_goal_radius_plans_no_detour():
    g = _grid()
    route = g.plan((1.0, 1.0), (1.1, 1.05), goal_radius_m=0.32)
    assert route is not None and len(route) == 1
    assert math.dist(route[0], (1.1, 1.05)) <= 0.32


def test_one_block_report_does_not_overturn_ground_the_robot_has_driven():
    """Field bug: near the goal, blocks reported with a drifting pose painted
    floor the robot had just driven across as danger."""
    g = _grid()
    g.mark_path((1.0, 1.0), (2.0, 1.0))                 # driven
    assert g.mark_blocked((1.05, 1.0, 0.0)) is None     # points at (1.5, 1.0)
    assert g.state_at((1.5, 1.0)) == SAFE
    assert g.mark_blocked((1.05, 1.0, 0.0)) == (1.5, 1.0)    # twice: believed
    assert g.state_at((1.5, 1.0)) == DANGER
    # ground never driven is marked at the first report
    assert g.mark_blocked((1.0, 2.0, 0.0)) == (1.5, 2.0)
