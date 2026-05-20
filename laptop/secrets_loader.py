"""Load secrets from the untracked ``telegram_bot_config.txt``.

Used by ``telegram_bot.py`` so the BOT token, authorized Telegram user IDs, and
Pi IP never live in a tracked file. See ``telegram_bot_config.example.txt``
for the format.
"""
import os

CONFIG_NAME = 'telegram_bot_config.txt'
EXAMPLE_NAME = 'telegram_bot_config.example.txt'


def _config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_NAME)


def load_config() -> dict:
    """Parse the config file into a dict. Missing file -> FileNotFoundError
    with a hint pointing at the example."""
    path = _config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'create {path} (copy from {EXAMPLE_NAME} and fill in values)')
    kv: dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            kv[k.strip()] = v.strip()
    return kv


def load_bot_secrets():
    """Return ``(bot_token, pi_ip, authorized_ids_set)`` for the Telegram bot.
    Raises ``ValueError`` on missing required values."""
    kv = load_config()
    token = kv.get('BOT_TOKEN', '')
    pi_ip = kv.get('PI_IP', '')
    ids_raw = kv.get('AUTHORIZED_IDS', '')
    if not token or not pi_ip or not ids_raw:
        raise ValueError(
            f'BOT_TOKEN, PI_IP, AUTHORIZED_IDS must all be set in {CONFIG_NAME}')
    ids = {int(x) for x in ids_raw.split(',') if x.strip()}
    return token, pi_ip, ids
