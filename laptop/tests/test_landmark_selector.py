"""Unit tests for landmark_selector: class -> ranked candidates."""
import numpy as np
import pytest

from slam.landmark_selector import (by_class, by_id, by_tag,
                                    nearest_by_class)
from slam.semantic_map import CONFIRMED, TENTATIVE, Landmark, SemanticMap


def _add(smap, lid, label, x, y, conf=1.0, state=CONFIRMED):
    lm = Landmark(id=lid)
    lm.class_votes = {label: conf}
    lm.position = np.array([x, y, 0.0])
    lm.state = state
    smap.landmarks[lid] = lm
    return lm


def test_by_id_returns_landmark():
    smap = SemanticMap()
    _add(smap, 1, 'chair', 1.0, 0.0)
    assert by_id(smap, 1).id == 1
    assert by_id(smap, 99) is None


def test_by_class_filters_state_and_confidence():
    smap = SemanticMap()
    _add(smap, 1, 'chair', 1.0, 0.0, conf=0.9)
    # landmark 2: conf=0.5 means 0.5 votes for 'chair' + 0.5 for another class
    lm2 = Landmark(id=2)
    lm2.class_votes = {'chair': 0.5, 'table': 0.5}
    lm2.position = np.array([2.0, 0.0, 0.0])
    lm2.state = CONFIRMED
    smap.landmarks[2] = lm2
    _add(smap, 3, 'chair', 3.0, 0.0, conf=0.9, state=TENTATIVE)
    _add(smap, 4, 'couch', 1.0, 1.0, conf=0.9)              # wrong class
    out = by_class(smap, 'chair', robot_xy=(0.0, 0.0),
                   conf_min=0.7, state=CONFIRMED)
    assert [lm.id for lm in out] == [1]


def test_by_class_orders_by_distance_then_id():
    smap = SemanticMap()
    _add(smap, 5, 'chair', 3.0, 0.0)
    _add(smap, 6, 'chair', 1.0, 0.0)
    _add(smap, 7, 'chair', 1.0, 0.0)                         # same dist as 6
    out = by_class(smap, 'chair', robot_xy=(0.0, 0.0))
    assert [lm.id for lm in out] == [6, 7, 5]


def test_nearest_by_class_returns_first_or_none():
    smap = SemanticMap()
    assert nearest_by_class(smap, 'chair', (0.0, 0.0)) is None
    _add(smap, 1, 'chair', 1.0, 0.0)
    assert nearest_by_class(smap, 'chair', (0.0, 0.0)).id == 1


def test_by_tag_returns_xy_or_none():
    tag_map = {0: {'x': 0.0, 'y': 0.0, 'z': 0.3, 'yaw_deg': 0.0, 'size_m': 0.1}}
    assert by_tag(tag_map, 0) == (0.0, 0.0)
    assert by_tag(tag_map, 9) is None


def test_tag_standoff_is_in_front_of_the_tag_facing_it():
    import math
    from slam.landmark_selector import tag_standoff
    tag_map = {0: {'x': 0.0, 'y': 0.0, 'yaw_deg': 90.0},     # faces +y
               5: {'x': 5.0, 'y': 1.0, 'yaw_deg': 180.0}}    # faces -x
    xy, yaw = tag_standoff(tag_map, 0, 1.0)
    assert xy == pytest.approx((0.0, 1.0), abs=1e-9)
    assert yaw == pytest.approx(-math.pi / 2)                # looking at -y
    xy, yaw = tag_standoff(tag_map, 5, 0.8)
    assert xy == pytest.approx((4.2, 1.0), abs=1e-9)
    assert abs(yaw) == pytest.approx(0.0, abs=1e-9)          # looking at +x
    assert tag_standoff(tag_map, 99, 1.0) is None


def test_standoff_stretches_for_a_tag_mounted_too_high_to_see_up_close():
    from slam.landmark_selector import min_view_distance, tag_standoff
    tag_map = {0: {'x': 0.0, 'y': 0.0, 'z': 0.30, 'size_m': 0.11,
                   'yaw_deg': 90.0},
               5: {'x': 5.0, 'y': 1.0, 'z': 0.54, 'size_m': 0.11,
                   'yaw_deg': 180.0}}
    # The stop point keeps a margin beyond the closest distance the tag is
    # visible from: the robot arrives anywhere within its arrival radius of
    # the point, and stopping short used to put the tag out of view.
    xy, _ = tag_standoff(tag_map, 0, 1.0, cam_height_m=0.15)
    need0 = min_view_distance(tag_map[0], 0.15)
    assert xy[1] == pytest.approx(need0 + 0.35)
    assert 1.25 < xy[1] < 1.4
    # high tag: must back off much further
    xy, yaw = tag_standoff(tag_map, 5, 1.0, cam_height_m=0.15)
    need = min_view_distance(tag_map[5], 0.15)
    assert 2.0 < need < 2.2
    assert 5.0 - xy[0] == pytest.approx(need + 0.35)
    # a low tag keeps the plain distance
    low = {9: {'x': 0.0, 'y': 0.0, 'z': 0.2, 'size_m': 0.11, 'yaw_deg': 0.0}}
    xy, _ = tag_standoff(low, 9, 1.0, cam_height_m=0.15)
    assert xy[0] == pytest.approx(1.0)
    assert abs(yaw) == pytest.approx(0.0, abs=1e-9)      # still facing it
    # without a camera height the plain distance is used, as before
    xy, _ = tag_standoff(tag_map, 5, 1.0)
    assert xy == pytest.approx((4.0, 1.0), abs=1e-9)
    # a tag at or below camera height needs no extra distance
    assert min_view_distance({'z': 0.10, 'size_m': 0.08}, 0.15) == 0.0


def test_standoff_swings_round_a_corner_tag_instead_of_hugging_the_wall():
    """Field bug: tag 0 sits in a corner. Straight in front of it is 15 cm
    from the side wall; the sonar stops the robot 35 cm short of a wall, so
    it was blocked three times and never got within reach."""
    import math
    from slam.landmark_selector import tag_bounds, tag_standoff
    tag_map = {0: {'x': 0.0, 'y': 0.0, 'z': 0.2, 'yaw_deg': 83.3},
               4: {'x': 4.3, 'y': -0.5, 'z': 0.2, 'yaw_deg': 83.3},
               5: {'x': 5.3, 'y': 1.2, 'z': 0.2, 'yaw_deg': 173.3},
               6: {'x': 0.2, 'y': 2.8, 'z': 0.2, 'yaw_deg': -6.7}}
    bounds = tag_bounds(tag_map)
    assert bounds == (0.0, -0.5, 5.3, 2.8)
    plain, _ = tag_standoff(tag_map, 0, 1.3)
    assert plain[0] < 0.2                                   # against the wall
    (x, y), yaw = tag_standoff(tag_map, 0, 1.3, bounds=bounds)
    assert x >= 0.5                                         # clear of it now
    assert math.hypot(x, y) == pytest.approx(1.3)           # same distance
    # still looking straight at the tag
    assert yaw == pytest.approx(math.atan2(-y, -x))
    # the smallest swing that clears, towards the open side of the room
    off = math.degrees(math.atan2(y, x)) - 83.3
    assert off == pytest.approx(-30.0)


def test_standoff_with_room_in_front_stays_on_the_normal():
    from slam.landmark_selector import tag_bounds, tag_standoff
    tag_map = {0: {'x': 0.0, 'y': 0.0, 'yaw_deg': 90.0},
               3: {'x': 2.5, 'y': 0.0, 'yaw_deg': 90.0},
               5: {'x': 5.0, 'y': 3.0, 'yaw_deg': 180.0}}
    xy, _ = tag_standoff(tag_map, 3, 1.0, bounds=tag_bounds(tag_map))
    assert xy == pytest.approx((2.5, 1.0))


def test_standoff_with_no_clear_spot_takes_the_roomiest():
    from slam.landmark_selector import tag_standoff
    tag_map = {0: {'x': 0.0, 'y': 0.0, 'yaw_deg': 90.0}}
    # a room too small for any candidate to be 0.5 m from every side
    (x, y), _ = tag_standoff(tag_map, 0, 1.0, bounds=(-0.3, 0.0, 0.3, 1.2))
    assert abs(x) < 1e-6 and y == pytest.approx(1.0)


def test_standoff_skips_a_spot_the_grid_knows_as_danger():
    """Field: the swung-round viewpoint for tag 6 landed exactly on a node
    marked as an obstacle on an earlier run — an unreachable goal."""
    import math
    from slam.landmark_selector import tag_standoff
    tag_map = {3: {'x': 2.5, 'y': 0.0, 'yaw_deg': 90.0}}
    bounds = (0.0, 0.0, 5.0, 3.0)
    plain, _ = tag_standoff(tag_map, 3, 1.0, bounds=bounds)
    assert plain == pytest.approx((2.5, 1.0))
    danger = lambda p: math.dist(p, (2.5, 1.0)) < 0.15
    (x, y), yaw = tag_standoff(tag_map, 3, 1.0, bounds=bounds, avoid=danger)
    assert math.dist((x, y), (2.5, 1.0)) > 0.2            # moved off it
    assert math.hypot(x - 2.5, y) == pytest.approx(1.0)   # same distance
    assert yaw == pytest.approx(math.atan2(-y, 2.5 - x))  # still facing it
