"""Telegram bot — phone natural-language control for the search robot.

Receives messages, keyword-parses them (nl_parser), POSTs commands to the Pi
agent_node, fetches camera snapshots, and polls agent status to relay arrival
and rotate-photo events back to the phone.

Setup:
    pip install python-telegram-bot requests
    cp laptop/telegram_bot_config.example.txt laptop/telegram_bot_config.txt
    # edit telegram_bot_config.txt — BOT_TOKEN, PI_IP, AUTHORIZED_IDS
Run:    python telegram_bot.py
"""
import io
import sys
import threading
import time

import requests
from telegram import (InlineKeyboardButton, InlineKeyboardMarkup, Update)
from telegram.ext import (ApplicationBuilder, CallbackQueryHandler,
                          ContextTypes, MessageHandler, filters)

from nl_parser import parse, SUPPORTED_TARGETS_CN
from secrets_loader import load_bot_secrets

try:
    BOT_TOKEN, PI_IP, AUTHORIZED_IDS = load_bot_secrets()
except (FileNotFoundError, ValueError) as _e:
    print(f'config error: {_e}', file=sys.stderr)
    sys.exit(1)

CMD_URL = f'http://{PI_IP}:9091/command'
STATUS_URL = f'http://{PI_IP}:9091/status'
SNAPSHOT_URL = f'http://{PI_IP}:8080/snapshot'
# laptop_detector serves detection-boxed frames here (same laptop as the bot)
DETECTOR_URL = 'http://127.0.0.1:8090/annotated'
# laptop_detector --slam serves the top-down semantic map here
MAP_URL = 'http://127.0.0.1:8091/map.png'
NAV_API = 'http://127.0.0.1:8092'

HELP = ('支持指令：前进/后退/左移/右移/左转/右转/停/拍照/旋转拍照/地图/'
        '去找<目标>。可加时长：前进3秒、后退2秒(最长30秒)。'
        '目标：' + SUPPORTED_TARGETS_CN)

PHOTO_LABELS = {1: '前', 2: '右', 3: '后', 4: '左'}

_session = requests.Session()


def post_command(cmd):
    """POST a command to the Pi. Returns True on success."""
    try:
        _session.post(CMD_URL, json=cmd, timeout=2.0)
        return True
    except requests.RequestException:
        return False


def fetch_photo():
    """Return JPEG bytes of the current view. Prefer the detector's
    detection-boxed frame; fall back to the Pi's raw camera snapshot if the
    detector is not running."""
    for url in (DETECTOR_URL, SNAPSHOT_URL):
        try:
            r = _session.get(url, timeout=3.0)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
    return None


def fetch_map():
    """Return PNG bytes of the top-down semantic map, or None if the SLAM
    mapper (laptop_detector --slam) is not running."""
    try:
        r = _session.get(MAP_URL, timeout=3.0)
        if r.status_code == 200:
            return r.content
    except requests.RequestException:
        pass
    return None


def get_status():
    try:
        return _session.get(STATUS_URL, timeout=2.0).json()
    except requests.RequestException:
        return None


def fetch_candidates(cls):
    try:
        r = _session.post(f'{NAV_API}/nav/candidates',
                          json={'target_class': cls}, timeout=3.0)
        return r.json().get('candidates', [])
    except requests.RequestException:
        return []


def start_goto_id(lid):
    try:
        r = _session.post(f'{NAV_API}/nav/goto',
                          json={'landmark_id': lid}, timeout=3.0)
        return r.status_code == 200
    except requests.RequestException:
        return False


def start_goto_tag(tid):
    try:
        r = _session.post(f'{NAV_API}/nav/goto',
                          json={'tag_id': tid}, timeout=3.0)
        return r.status_code == 200
    except requests.RequestException:
        return False


def start_goto_xy(x, y, label):
    try:
        r = _session.post(f'{NAV_API}/nav/goto',
                          json={'xy': [x, y], 'label': label}, timeout=3.0)
        return r.status_code == 200
    except requests.RequestException:
        return False


def poll_nav_done():
    try:
        r = _session.get(f'{NAV_API}/nav/done', timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            return data if data else None
    except requests.RequestException:
        pass
    return None


def _poll_nav(loop, context, chat_id, stop_after_sec=300):
    import asyncio
    deadline = time.time() + stop_after_sec
    while time.time() < deadline:
        time.sleep(0.7)
        ev = poll_nav_done()
        if ev is None:
            continue
        state = ev.get('state')
        label = ev.get('goal_label') or '?'
        gid = ev.get('goal_id')
        reason = ev.get('reason')
        if state == 'ARRIVED':
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id,
                    f'已到达 {label}' + (f' (id {gid})' if gid is not None else '')),
                loop)
        else:
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id,
                    f'导航失败 ({reason or state})'), loop)
        return


def _run_patrol(loop, context, chat_id):
    pass


def _run_room_tour(loop, context, chat_id):
    pass


async def _send_photo(context, chat_id, caption):
    jpg = fetch_photo()
    if jpg is None:
        await context.bot.send_message(chat_id, '取图失败')
        return
    await context.bot.send_photo(chat_id, photo=io.BytesIO(jpg), caption=caption)


def _poll_events(loop, context, chat_id, stop_after_sec=120):
    """Background thread: poll Pi status, relay arrival / rotate-photo events."""
    import asyncio
    deadline = time.time() + stop_after_sec
    while time.time() < deadline:
        time.sleep(0.7)
        st = get_status()
        if st is None:
            continue
        event = st.get('event', '')
        if not event:
            continue
        if event.startswith('arrived:'):
            target = event.split(':', 1)[1]
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id, f'已到达 {target}'), loop)
            asyncio.run_coroutine_threadsafe(
                _send_photo(context, chat_id, f'到达 {target}'), loop)
            return
        if event.startswith('photo_ready:'):
            n = int(event.split(':', 1)[1])
            label = PHOTO_LABELS.get(n, str(n))
            asyncio.run_coroutine_threadsafe(
                _send_photo(context, chat_id, f'旋转拍照 {n}/4 ({label})'), loop)
        elif event == 'rotate_photo_done':
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id, '旋转拍照完成'), loop)
            return


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in AUTHORIZED_IDS:
        return  # ignore unauthorized users
    chat_id = update.effective_chat.id
    text = update.message.text or ''
    cmd = parse(text)

    if cmd is None:
        await context.bot.send_message(chat_id, '没听懂。' + HELP)
        return

    if cmd['action'] == 'photo':
        await _send_photo(context, chat_id, '当前画面')
        return

    if cmd['action'] == 'map':
        png = fetch_map()
        if png is None:
            await context.bot.send_message(
                chat_id, '语义地图未就绪（需 laptop_detector --slam 运行）')
        else:
            await context.bot.send_photo(chat_id, photo=io.BytesIO(png),
                                         caption='语义地图')
        return

    if cmd['action'] == 'goto':
        loop = __import__('asyncio').get_running_loop()
        if cmd.get('origin'):
            ok = start_goto_tag(0)
            if not ok:
                await context.bot.send_message(chat_id, '原点 (tag 0) 未在地图')
            else:
                await context.bot.send_message(chat_id, '前往原点 ...')
                threading.Thread(target=_poll_nav,
                                 args=(loop, context, chat_id),
                                 daemon=True).start()
            return
        if 'landmark_id' in cmd:
            ok = start_goto_id(cmd['landmark_id'])
            if not ok:
                await context.bot.send_message(
                    chat_id, f"id {cmd['landmark_id']} 未在地图")
            else:
                await context.bot.send_message(
                    chat_id, f"前往 landmark id {cmd['landmark_id']} ...")
                threading.Thread(target=_poll_nav,
                                 args=(loop, context, chat_id),
                                 daemon=True).start()
            return
        if 'target_class' in cmd:
            cands = fetch_candidates(cmd['target_class'])
            if not cands:
                await context.bot.send_message(
                    chat_id, f"未在地图中找到 {cmd['target_class']}")
                return
            if len(cands) == 1:
                lid = cands[0]['id']
                start_goto_id(lid)
                await context.bot.send_message(
                    chat_id, f"前往 {cmd['target_class']} id {lid} ...")
                threading.Thread(target=_poll_nav,
                                 args=(loop, context, chat_id),
                                 daemon=True).start()
                return
            buttons = [[InlineKeyboardButton(
                f"{c['label']} id {c['id']} ({c['dist_m']:.1f}m)",
                callback_data=f"goto:{c['id']}")] for c in cands]
            await context.bot.send_message(
                chat_id, f"找到 {len(cands)} 个 {cmd['target_class']}",
                reply_markup=InlineKeyboardMarkup(buttons))
            return

    if cmd['action'] == 'patrol':
        await context.bot.send_message(chat_id, '开始巡逻 ...')
        threading.Thread(target=_run_patrol, args=(
            __import__('asyncio').get_running_loop(), context, chat_id),
            daemon=True).start()
        return

    if cmd['action'] == 'room_tour':
        await context.bot.send_message(chat_id, '开始绕室 ...')
        threading.Thread(target=_run_room_tour, args=(
            __import__('asyncio').get_running_loop(), context, chat_id),
            daemon=True).start()
        return

    if cmd['action'] == 'find' and cmd['target'] is None:
        await context.bot.send_message(chat_id, '不认识该目标。' + HELP)
        return

    if not post_command(cmd):
        await context.bot.send_message(chat_id, '小车失联')
        return

    if cmd['action'] == 'find':
        await context.bot.send_message(chat_id, f"前往寻找 {cmd['target']} ...")
        loop = __import__('asyncio').get_running_loop()
        threading.Thread(target=_poll_events, args=(loop, context, chat_id),
                         daemon=True).start()
    elif cmd['action'] == 'rotate_photo':
        await context.bot.send_message(chat_id, '开始旋转拍照 ...')
        loop = __import__('asyncio').get_running_loop()
        threading.Thread(target=_poll_events, args=(loop, context, chat_id),
                         daemon=True).start()
    else:
        await context.bot.send_message(chat_id, f"执行：{text}")


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.from_user.id not in AUTHORIZED_IDS:
        return
    data = q.data or ''
    if data.startswith('goto:'):
        lid = int(data.split(':', 1)[1])
        if start_goto_id(lid):
            await q.edit_message_text(f'前往 landmark id {lid} ...')
            threading.Thread(target=_poll_nav, args=(
                __import__('asyncio').get_running_loop(), context,
                q.message.chat.id), daemon=True).start()
        else:
            await q.edit_message_text('启动导航失败')


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(CallbackQueryHandler(on_callback))
    print('telegram bot running — message your bot from the phone')
    app.run_polling()


if __name__ == '__main__':
    main()
