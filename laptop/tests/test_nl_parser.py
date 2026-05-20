import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nl_parser import parse


def test_move_forward():
    assert parse('前进') == {'action': 'move', 'dir': 'forward'}
    assert parse('往前') == {'action': 'move', 'dir': 'forward'}


def test_move_backward():
    assert parse('后退') == {'action': 'move', 'dir': 'backward'}


def test_rotate_vs_strafe():
    assert parse('左转') == {'action': 'move', 'dir': 'rotate_left'}
    assert parse('右转') == {'action': 'move', 'dir': 'rotate_right'}
    assert parse('左移') == {'action': 'move', 'dir': 'left'}
    assert parse('右移') == {'action': 'move', 'dir': 'right'}


def test_stop():
    assert parse('停') == {'action': 'stop'}
    assert parse('停下来') == {'action': 'stop'}


def test_photo():
    assert parse('拍照') == {'action': 'photo'}
    assert parse('给我看看') == {'action': 'photo'}


def test_rotate_photo_not_photo():
    assert parse('旋转拍照') == {'action': 'rotate_photo'}
    assert parse('环拍') == {'action': 'rotate_photo'}


def test_find_known_target():
    assert parse('去找瓶子') == {'action': 'find', 'target': 'bottle'}
    assert parse('找椅子') == {'action': 'find', 'target': 'chair'}
    assert parse('靠近那个人') == {'action': 'find', 'target': 'person'}


def test_find_english_target():
    assert parse('find bottle') == {'action': 'find', 'target': 'bottle'}


def test_find_unknown_target():
    assert parse('去找独角兽') == {'action': 'find', 'target': None}


def test_map():
    assert parse('地图') == {'action': 'map'}
    assert parse('给我看语义地图') == {'action': 'map'}
    assert parse('map') == {'action': 'map'}


def test_move_with_duration():
    assert parse('前进3秒') == {'action': 'move', 'dir': 'forward', 'seconds': 3.0}
    assert parse('后退 2 秒') == {'action': 'move', 'dir': 'backward', 'seconds': 2.0}
    assert parse('左转1.5秒') == {'action': 'move', 'dir': 'rotate_left', 'seconds': 1.5}
    assert parse('forward 4s') == {'action': 'move', 'dir': 'forward', 'seconds': 4.0}
    # no duration -> no 'seconds' key (agent_node falls back to manual_step_sec)
    assert parse('前进') == {'action': 'move', 'dir': 'forward'}


def test_move_duration_capped():
    # safety cap so a typo cannot drive the car for minutes
    assert parse('前进300秒')['seconds'] == 30.0
    assert parse('前进0秒')['seconds'] == 0.2


def test_unparseable():
    assert parse('帮我把灯关了') is None
    assert parse('') is None
