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
    # low tag: 1 m is (just) enough
    xy, _ = tag_standoff(tag_map, 0, 1.0, cam_height_m=0.15)
    assert 1.0 <= xy[1] < 1.15
    # high tag: must back off to about 2.3 m
    xy, yaw = tag_standoff(tag_map, 5, 1.0, cam_height_m=0.15)
    need = min_view_distance(tag_map[5], 0.15)
    assert 2.0 < need < 2.2
    assert 5.0 - xy[0] == pytest.approx(1.1 * need)
    assert abs(yaw) == pytest.approx(0.0, abs=1e-9)      # still facing it
    # without a camera height the plain distance is used, as before
    xy, _ = tag_standoff(tag_map, 5, 1.0)
    assert xy == pytest.approx((4.0, 1.0), abs=1e-9)
    # a tag at or below camera height needs no extra distance
    assert min_view_distance({'z': 0.10, 'size_m': 0.08}, 0.15) == 0.0
