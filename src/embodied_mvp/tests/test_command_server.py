import json
import time
import urllib.request

from embodied_mvp.command_server import CommandServer


def _post(port, body):
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}/command',
        data=json.dumps(body).encode(), method='POST')
    return urllib.request.urlopen(req, timeout=2).read()


def _get_status(port):
    return json.loads(
        urllib.request.urlopen(f'http://127.0.0.1:{port}/status', timeout=2).read())


def test_command_roundtrip():
    srv = CommandServer(port=19099)
    try:
        time.sleep(0.2)
        _post(19099, {'action': 'stop'})
        assert srv.take_command() == {'action': 'stop'}
        assert srv.take_command() is None
    finally:
        srv.shutdown()


def test_status_and_event_oneshot():
    srv = CommandServer(port=19098)
    try:
        time.sleep(0.2)
        srv.set_mode('SEARCH')
        srv.post_event('arrived:bottle')
        s1 = _get_status(19098)
        assert s1['mode'] == 'SEARCH'
        assert s1['event'] == 'arrived:bottle'
        s2 = _get_status(19098)
        assert s2['event'] == ''
    finally:
        srv.shutdown()
