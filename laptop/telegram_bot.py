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
from pi_events import EventCursor
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
        '去找<目标>/去<目标>/去 id N/回原点/巡逻/绕室。'
        '可加时长：前进3秒、后退2秒(最长30秒)。'
        '目标：' + SUPPORTED_TARGETS_CN)

PHOTO_LABELS = {1: '前', 2: '右', 3: '后', 4: '左'}

_session = requests.Session()

# Each robot command supersedes the watcher thread of the previous one, so a
# stale watcher can't report someone else's arrival or — worse — start its
# map-navigation fallback after the operator has said `停`.
_watch_lock = threading.Lock()
_watch_gen = 0


def _new_watch():
    global _watch_gen
    with _watch_lock:
        _watch_gen += 1
        return _watch_gen


def _watch_alive(gen):
    return gen == _watch_gen


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


def start_seek(cls):
    """Ask the detector to find ``cls``: map navigation and visual search run
    as one behaviour there (vision wins whenever the object is in view).
    False if the detector / SLAM is not running."""
    try:
        r = _session.post(f'{NAV_API}/nav/seek',
                          json={'target_class': cls}, timeout=3.0)
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


def stop_nav():
    """Tell the laptop to abort any active NavSession. Best effort."""
    try:
        _session.post(f'{NAV_API}/nav/stop', timeout=2.0)
    except requests.RequestException:
        pass


def _nav_result_text(ev):
    """Phone message for a finished goto / seek event from the detector."""
    state = ev.get('state')
    label = ev.get('goal_label') or '?'
    gid = ev.get('goal_id')
    reason = ev.get('reason')
    unseen = ev.get('unseen_ids') or []
    stale = ('（地图记录的 ' + '、'.join(f'id {i}' for i in unseen)
             + ' 位置已无此物）') if unseen else ''
    if state == 'ARRIVED':
        if ev.get('seen') is False:
            # reached the map position, but vision does not confirm it
            return f'已到 {label} id {gid} 的记录位置，但没看到它'
        if ev.get('via') == 'vision':
            return f'已到达 {label}（视觉确认）' + stale
        return f'已到达 {label}' + (f' (id {gid})' if gid is not None else '')
    if reason == 'not_found':
        return f'没找到 {label}' + stale
    if reason == 'timeout':
        return f'寻找 {label} 超时' + stale
    return f'导航失败 ({reason or state})'


def _poll_nav(loop, context, chat_id, gen=None, stop_after_sec=450):
    """Wait for the detector's goto / seek result and relay it. ``gen`` ends
    the wait silently once a newer command has superseded this one."""
    import asyncio
    deadline = time.time() + stop_after_sec
    while time.time() < deadline:
        time.sleep(0.7)
        if gen is not None and not _watch_alive(gen):
            return
        ev = poll_nav_done()
        if ev is None:
            continue
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id, _nav_result_text(ev)), loop)
        if ev.get('state') == 'ARRIVED':
            asyncio.run_coroutine_threadsafe(
                _send_photo(context, chat_id,
                            f"到达 {ev.get('goal_label') or ''}"), loop)
        return
    # loop exited without a nav_done event — tell the user, don't fail silently
    asyncio.run_coroutine_threadsafe(
        context.bot.send_message(chat_id,
            f'导航超时 ({stop_after_sec:.0f}s 无结果)'), loop)


def _fetch_list(path):
    try:
        r = _session.get(f'{NAV_API}/{path}', timeout=3.0)
        return r.json()
    except requests.RequestException:
        return {}


def _wait_for_nav_done(timeout_sec):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        time.sleep(0.7)
        ev = poll_nav_done()
        if ev:
            return ev
    return None


def _run_patrol(loop, context, chat_id):
    import asyncio
    data = _fetch_list('nav/landmarks')
    landmarks = data.get('landmarks', [])
    if not landmarks:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id, '地图中没有 CONFIRMED 物体'), loop)
        return
    for lm in landmarks:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id,
                f"前往 {lm['label']} id {lm['id']} ..."), loop)
        if not start_goto_id(lm['id']):
            continue
        ev = _wait_for_nav_done(180.0)
        if ev is None or ev.get('state') != 'ARRIVED':
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id,
                    f"跳过 {lm['label']} id {lm['id']} ({ev.get('reason') if ev else 'timeout'})"),
                loop)
            continue
        asyncio.run_coroutine_threadsafe(
            _send_photo(context, chat_id,
                        f"{lm['label']} id {lm['id']}"), loop)
    asyncio.run_coroutine_threadsafe(
        context.bot.send_message(chat_id, '巡逻完成'), loop)


def _run_room_tour(loop, context, chat_id):
    import asyncio
    data = _fetch_list('nav/tags')
    tags = sorted(data.get('tags', []), key=lambda t: t['id'])
    if not tags:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id, 'tag_map 为空'), loop)
        return
    for t in tags:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id, f"前往 tag {t['id']} ..."), loop)
        if not start_goto_tag(t['id']):
            continue
        ev = _wait_for_nav_done(180.0)
        if ev is None or ev.get('state') != 'ARRIVED':
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id,
                    f"跳过 tag {t['id']} ({ev.get('reason') if ev else 'timeout'})"),
                loop)
            continue
    asyncio.run_coroutine_threadsafe(
        context.bot.send_message(chat_id, '绕室完成'), loop)


async def _send_photo(context, chat_id, caption):
    jpg = fetch_photo()
    if jpg is None:
        await context.bot.send_message(chat_id, '取图失败')
        return
    await context.bot.send_photo(chat_id, photo=io.BytesIO(jpg), caption=caption)


def _poll_events(loop, context, chat_id, cursor, gen, stop_after_sec=120):
    """Background thread: poll Pi status, relay arrival / rotate-photo events.

    ``cursor`` was synced before the command was sent, so only events emitted
    in response to it are seen; ``gen`` ends the thread once a newer command
    has superseded this one."""
    import asyncio
    deadline = time.time() + stop_after_sec
    while time.time() < deadline and _watch_alive(gen):
        time.sleep(0.7)
        for event in cursor.take(get_status()):
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
                    _send_photo(context, chat_id,
                                f'旋转拍照 {n}/4 ({label})'), loop)
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

    # Any command that drives the robot supersedes earlier watcher threads.
    gen = _new_watch()

    if cmd['action'] == 'goto':
        loop = __import__('asyncio').get_running_loop()
        if cmd.get('origin'):
            ok = start_goto_tag(0)
            if not ok:
                await context.bot.send_message(chat_id, '原点 (tag 0) 未在地图')
            else:
                await context.bot.send_message(chat_id, '前往原点 ...')
                threading.Thread(target=_poll_nav,
                                 args=(loop, context, chat_id, gen),
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
                                 args=(loop, context, chat_id, gen),
                                 daemon=True).start()
            return
        if 'target_class' in cmd:
            cands = fetch_candidates(cmd['target_class'])
            if len(cands) <= 1:
                # none or exactly one on the map -> a class seek: go to the
                # map position if there is one, vision takes over on sight.
                if not start_seek(cmd['target_class']):
                    await context.bot.send_message(
                        chat_id, '导航未就绪（需 laptop_detector --slam 运行）')
                    return
                await context.bot.send_message(
                    chat_id, f"前往 {cmd['target_class']} ...")
                threading.Thread(target=_poll_nav,
                                 args=(loop, context, chat_id, gen),
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

    # `stop` must abort any laptop-side NavSession in addition to telling the
    # Pi to switch to IDLE — otherwise the next tick re-issues a nav_pulse
    # and the robot resumes navigating.
    if cmd['action'] == 'stop':
        stop_nav()

    # `去找X`: the detector owns the whole behaviour (map + vision). Only if
    # it is not running do we fall back to a plain Pi-side visual search.
    if cmd['action'] == 'find' and start_seek(cmd['target']):
        await context.bot.send_message(chat_id, f"前往寻找 {cmd['target']} ...")
        threading.Thread(target=_poll_nav, args=(
            __import__('asyncio').get_running_loop(), context, chat_id, gen),
            daemon=True).start()
        return

    # Position the event cursor BEFORE sending, so the watcher sees exactly
    # the events this command causes — not history, and none missed.
    cursor = EventCursor()
    if cmd['action'] in ('find', 'rotate_photo'):
        cursor.sync(get_status())

    if not post_command(cmd):
        await context.bot.send_message(chat_id, '小车失联')
        return

    if cmd['action'] == 'find':
        await context.bot.send_message(
            chat_id, f"前往寻找 {cmd['target']} ...（仅视觉，地图未运行）")
        loop = __import__('asyncio').get_running_loop()
        threading.Thread(target=_poll_events,
                         args=(loop, context, chat_id, cursor, gen),
                         daemon=True).start()
    elif cmd['action'] == 'rotate_photo':
        await context.bot.send_message(chat_id, '开始旋转拍照 ...')
        loop = __import__('asyncio').get_running_loop()
        threading.Thread(target=_poll_events,
                         args=(loop, context, chat_id, cursor, gen),
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
            gen = _new_watch()
            await q.edit_message_text(f'前往 landmark id {lid} ...')
            threading.Thread(target=_poll_nav, args=(
                __import__('asyncio').get_running_loop(), context,
                q.message.chat.id, gen), daemon=True).start()
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
