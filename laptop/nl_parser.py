"""Keyword parser: Chinese / English text -> structured command dict.

parse(text) returns one of:
  {'action': 'move', 'dir': 'forward'|'backward'|'left'|'right'|'rotate_left'|'rotate_right',
                     'seconds': <optional float>}   # duration if user said "...3秒" / "...5s"
  {'action': 'stop'}
  {'action': 'photo'}
  {'action': 'rotate_photo'}
  {'action': 'map'}                               # request the semantic map
  {'action': 'goto', 'landmark_id': <int>}        # nav to a specific id
  {'action': 'goto', 'origin': True}              # nav to tag 0
  {'action': 'goto', 'target_class': '<coco_class>'}  # nav to nearest of class
  {'action': 'patrol'}                            # visit each CONFIRMED landmark
  {'action': 'room_tour'}                         # visit each tag by id
  {'action': 'find', 'target': '<coco_class>'}   # target None if unrecognised
  None  -- unparseable

Duration is capped at MAX_DURATION_SEC as a safety stop so a fat-fingered
"前进300秒" cannot run the car away unattended.

Targets cover the full COCO-80 class set. Aliases are matched longest-first so
short aliases (e.g. '车') never shadow longer ones (e.g. '自行车').
"""

import re

MAX_DURATION_SEC = 30.0
MIN_DURATION_SEC = 0.2

_DURATION_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:秒钟|秒|sec(?:onds?)?|s\b)', re.IGNORECASE)


def _extract_duration(text):
    """Return a duration in seconds parsed from ``text``, or None."""
    m = _DURATION_RE.search(text)
    if not m:
        return None
    s = float(m.group(1))
    return max(MIN_DURATION_SEC, min(MAX_DURATION_SEC, s))


def _move(direction, text):
    cmd = {'action': 'move', 'dir': direction}
    sec = _extract_duration(text)
    if sec is not None:
        cmd['seconds'] = sec
    return cmd


# Chinese alias -> COCO class name (full COCO-80 coverage)
TARGET_MAP = {
    # people / vehicles
    '人': 'person',
    '自行车': 'bicycle', '单车': 'bicycle',
    '汽车': 'car', '小汽车': 'car', '车': 'car',
    '摩托车': 'motorcycle', '摩托': 'motorcycle',
    '飞机': 'airplane',
    '公交车': 'bus', '巴士': 'bus', '公交': 'bus',
    '火车': 'train',
    '卡车': 'truck', '货车': 'truck',
    '船': 'boat',
    '红绿灯': 'traffic light', '交通灯': 'traffic light',
    '消防栓': 'fire hydrant',
    '停车标志': 'stop sign',
    '停车计时器': 'parking meter',
    '长椅': 'bench', '长凳': 'bench',
    # animals
    '鸟': 'bird',
    '猫': 'cat',
    '狗': 'dog',
    '马': 'horse',
    '羊': 'sheep',
    '牛': 'cow',
    '大象': 'elephant',
    '熊': 'bear',
    '斑马': 'zebra',
    '长颈鹿': 'giraffe',
    # accessories
    '背包': 'backpack',
    '雨伞': 'umbrella', '伞': 'umbrella',
    '手提包': 'handbag', '包': 'backpack',
    '领带': 'tie',
    '行李箱': 'suitcase', '手提箱': 'suitcase',
    # sports
    '飞盘': 'frisbee',
    '滑雪板': 'skis',
    '单板滑雪板': 'snowboard',
    '运动球': 'sports ball', '球': 'sports ball',
    '风筝': 'kite',
    '棒球棒': 'baseball bat',
    '棒球手套': 'baseball glove',
    '滑板': 'skateboard',
    '冲浪板': 'surfboard',
    '网球拍': 'tennis racket',
    # kitchen / food
    '瓶子': 'bottle', '水瓶': 'bottle', '瓶': 'bottle',
    '红酒杯': 'wine glass', '酒杯': 'wine glass',
    '杯子': 'cup', '杯': 'cup',
    '叉子': 'fork', '叉': 'fork',
    '刀': 'knife',
    '勺子': 'spoon', '汤匙': 'spoon', '勺': 'spoon',
    '碗': 'bowl',
    '香蕉': 'banana',
    '苹果': 'apple',
    '三明治': 'sandwich',
    '橙子': 'orange', '橘子': 'orange',
    '西兰花': 'broccoli',
    '胡萝卜': 'carrot',
    '热狗': 'hot dog',
    '披萨': 'pizza', '比萨': 'pizza',
    '甜甜圈': 'donut',
    '蛋糕': 'cake',
    # furniture
    '椅子': 'chair', '椅': 'chair',
    '沙发': 'couch',
    '盆栽': 'potted plant', '盆栽植物': 'potted plant',
    '床': 'bed',
    '餐桌': 'dining table',
    '马桶': 'toilet',
    # electronics
    '电视': 'tv', '电视机': 'tv',
    '笔记本电脑': 'laptop', '笔记本': 'laptop', '电脑': 'laptop',
    '鼠标': 'mouse',
    '遥控器': 'remote',
    '键盘': 'keyboard',
    '手机': 'cell phone',
    '微波炉': 'microwave',
    '烤箱': 'oven',
    '烤面包机': 'toaster',
    '水槽': 'sink',
    '冰箱': 'refrigerator',
    # misc
    '书': 'book',
    '时钟': 'clock', '钟': 'clock',
    '花瓶': 'vase',
    '剪刀': 'scissors',
    '泰迪熊': 'teddy bear', '玩具熊': 'teddy bear',
    '吹风机': 'hair drier',
    '牙刷': 'toothbrush',
}

# All 80 COCO class names — accepted directly if typed in English
COCO_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
}

SUPPORTED_TARGETS_CN = ('支持 COCO 80 类，常用：瓶子 / 椅子 / 人 / 杯子 / 手机 / '
                        '笔记本 / 书 / 背包 / 沙发 / 电视 / 猫 / 狗 / 碗 / 键盘 / 鼠标')


def _match_target(text):
    low = text.lower()
    # longest Chinese alias first (so '自行车' beats '车', '水瓶' beats '瓶')
    for cn in sorted(TARGET_MAP, key=len, reverse=True):
        if cn in text:
            return TARGET_MAP[cn]
    for c in sorted(COCO_CLASSES, key=len, reverse=True):
        if c in low:
            return c
    return None


def parse(text):
    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    # rotate_photo MUST be checked before photo ('旋转拍照' contains '拍照')
    if any(k in t for k in ('旋转拍照', '环拍', '转一圈拍照', '转圈拍照')):
        return {'action': 'rotate_photo'}

    if any(k in t for k in ('地图', '语义地图', 'map')):
        return {'action': 'map'}

    if any(k in t for k in ('停', 'stop')):
        return {'action': 'stop'}

    if any(k in t for k in ('拍照', '照片', '看看', 'photo')):
        return {'action': 'photo'}

    # goto id N -- must match before plain '左/前/后' direction matchers
    m = re.search(r'(?:去|goto)\s*id\s*(\d+)', t, flags=re.IGNORECASE)
    if m:
        return {'action': 'goto', 'landmark_id': int(m.group(1))}

    if any(k in t for k in ('回原点', '回到原点', 'go home', 'home')):
        return {'action': 'goto', 'origin': True}

    if any(k in t for k in ('巡逻', 'patrol')):
        return {'action': 'patrol'}

    if any(k in t for k in ('绕室', 'room tour')):
        return {'action': 'room_tour'}

    # 'goto <class>' / '去<class>' (but not '去找') -- requires a target match
    if '去找' not in t and ('去' in t or t.lower().startswith('goto')):
        target = _match_target(t)
        if target:
            return {'action': 'goto', 'target_class': target}

    if any(k in t for k in ('找', '寻找', '靠近', 'find')):
        return {'action': 'find', 'target': _match_target(t)}

    # rotation before plain strafe ('左转' contains '左')
    if any(k in t for k in ('左转', 'turn left', 'rotate left')):
        return _move('rotate_left', t)
    if any(k in t for k in ('右转', 'turn right', 'rotate right')):
        return _move('rotate_right', t)
    if any(k in t for k in ('左移', '向左', '左')):
        return _move('left', t)
    if any(k in t for k in ('右移', '向右', '右')):
        return _move('right', t)
    if any(k in t for k in ('前进', '往前', '向前', '前', 'forward')):
        return _move('forward', t)
    if any(k in t for k in ('后退', '倒退', '后', 'backward', 'back')):
        return _move('backward', t)

    return None
