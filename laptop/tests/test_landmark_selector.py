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
