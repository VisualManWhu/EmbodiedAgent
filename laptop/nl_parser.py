"""Keyword parser: Chinese / English text -> structured command dict.

parse(text) returns one of:
  {'action': 'move', 'dir': 'forward'|'backward'|'left'|'right'|'rotate_left'|'rotate_right'}
  {'action': 'stop'}
  {'action': 'photo'}
  {'action': 'rotate_photo'}
  {'action': 'find', 'target': '<coco_class>'}   # target None if unrecognised
  None  -- unparseable
"""

# Chinese alias -> COCO class name
TARGET_MAP = {
    '瓶子': 'bottle', '水瓶': 'bottle', '瓶': 'bottle',
    '椅子': 'chair', '椅': 'chair',
    '人': 'person',
    '杯子': 'cup', '杯': 'cup',
    '手机': 'cell phone',
    '笔记本': 'laptop', '电脑': 'laptop',
    '书': 'book',
    '键盘': 'keyboard',
    '鼠标': 'mouse',
    '背包': 'backpack', '包': 'backpack',
    '沙发': 'couch',
    '电视': 'tv',
    '钟': 'clock', '时钟': 'clock',
}

# English COCO class names accepted directly
COCO_CLASSES = {
    'person', 'bottle', 'cup', 'bowl', 'chair', 'couch', 'bed', 'tv',
    'laptop', 'mouse', 'keyboard', 'cell phone', 'book', 'clock', 'vase',
    'backpack', 'umbrella', 'handbag', 'remote', 'scissors', 'dining table',
}

SUPPORTED_TARGETS_CN = '瓶子 / 椅子 / 人 / 杯子 / 手机 / 笔记本 / 书 / 背包 / 沙发 / 电视'


def _match_target(text):
    low = text.lower()
    # longest Chinese alias first (so '水瓶' beats '瓶')
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

    if any(k in t for k in ('停', 'stop')):
        return {'action': 'stop'}

    if any(k in t for k in ('拍照', '照片', '看看', 'photo')):
        return {'action': 'photo'}

    if any(k in t for k in ('找', '寻找', '靠近', 'find')):
        return {'action': 'find', 'target': _match_target(t)}

    # rotation before plain strafe ('左转' contains '左')
    if any(k in t for k in ('左转', 'turn left', 'rotate left')):
        return {'action': 'move', 'dir': 'rotate_left'}
    if any(k in t for k in ('右转', 'turn right', 'rotate right')):
        return {'action': 'move', 'dir': 'rotate_right'}
    if any(k in t for k in ('左移', '向左', '左')):
        return {'action': 'move', 'dir': 'left'}
    if any(k in t for k in ('右移', '向右', '右')):
        return {'action': 'move', 'dir': 'right'}
    if any(k in t for k in ('前进', '往前', '向前', '前', 'forward')):
        return {'action': 'move', 'dir': 'forward'}
    if any(k in t for k in ('后退', '倒退', '后', 'backward', 'back')):
        return {'action': 'move', 'dir': 'backward'}

    return None
