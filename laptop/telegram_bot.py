"""Telegram bot — phone natural-language control for the search robot.

Receives messages, keyword-parses them (nl_parser), POSTs commands to the Pi
agent_node, fetches camera snapshots, and polls agent status to relay arrival
and rotate-photo events back to the phone.

Setup:
    pip install python-telegram-bot requests
Config: set BOT_TOKEN, PI_IP, AUTHORIZED_IDS below.
Run:    python telegram_bot.py
"""
import io
import os
import threading
import time

import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

from nl_parser import parse, SUPPORTED_TARGETS_CN


def _load_config():
    """Load secrets from telegram_bot_config.txt (gitignored — never committed).

    Copy telegram_bot_config.example.txt to telegram_bot_config.txt and fill in
    BOT_TOKEN / PI_IP / AUTHORIZED_IDS. Keeping the real token out of the
    tracked source prevents it leaking on the public repo.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'telegram_bot_config.txt')
    if not os.path.exists(path):
        raise SystemExit(
            f'config not found: {path}\n'
            'Copy telegram_bot_config.example.txt to telegram_bot_config.txt '
            'and fill in BOT_TOKEN / PI_IP / AUTHORIZED_IDS.')
    cfg = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip()
    return cfg


_cfg = _load_config()
BOT_TOKEN = _cfg['BOT_TOKEN']
PI_IP = _cfg['PI_IP']
AUTHORIZED_IDS = {int(x) for x in _cfg.get('AUTHORIZED_IDS', '').split(',') if x.strip()}

CMD_URL = f'http://{PI_IP}:9091/command'
STATUS_URL = f'http://{PI_IP}:9091/status'
SNAPSHOT_URL = f'http://{PI_IP}:8080/snapshot'
# laptop_detector serves detection-boxed frames here (same laptop as the bot)
DETECTOR_URL = 'http://127.0.0.1:8090/annotated'

HELP = ('支持指令：前进/后退/左移/右移/左转/右转/停/拍照/旋转拍照/'
        '去找<目标>。目标：' + SUPPORTED_TARGETS_CN)

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


def get_status():
    try:
        return _session.get(STATUS_URL, timeout=2.0).json()
    except requests.RequestException:
        return None


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


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    print('telegram bot running — message your bot from the phone')
    app.run_polling()


if __name__ == '__main__':
    main()
