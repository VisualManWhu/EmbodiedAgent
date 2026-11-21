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


def test_status_event_survives_repeated_reads():
    """A GET must not consume the event — a second reader still sees it."""
    srv = CommandServer(port=19098)
    try:
        time.sleep(0.2)
        srv.set_mode('SEARCH')
        srv.post_event('arrived:bottle')
        s1 = _get_status(19098)
        assert s1['mode'] == 'SEARCH'
        assert s1['event_seq'] == 1
        assert s1['events'] == [{'seq': 1, 'event': 'arrived:bottle'}]
        s2 = _get_status(19098)
        assert s2['events'] == s1['events']
        assert s2['boot_id'] == s1['boot_id']
    finally:
        srv.shutdown()


def test_events_are_sequenced_and_none_overwritten():
    """Two events posted between polls are both delivered, in order."""
    srv = CommandServer(port=19097)
    try:
        time.sleep(0.2)
        assert _get_status(19097)['events'] == []
        srv.post_event('blocked:front')
        srv.post_event('pulse_done')
        s = _get_status(19097)
        assert s['event_seq'] == 2
        assert [e['event'] for e in s['events']] == ['blocked:front',
                                                     'pulse_done']
        assert [e['seq'] for e in s['events']] == [1, 2]
        assert s['event'] == 'pulse_done'
    finally:
        srv.shutdown()


def test_event_history_is_bounded():
    from embodied_mvp.command_server import EVENT_HISTORY
    srv = CommandServer(port=19096)
    try:
        time.sleep(0.2)
        for i in range(EVENT_HISTORY + 5):
            srv.post_event(f'e{i}')
        s = _get_status(19096)
        assert len(s['events']) == EVENT_HISTORY
        assert s['event_seq'] == EVENT_HISTORY + 5
        assert s['events'][-1] == {'seq': EVENT_HISTORY + 5,
                                   'event': f'e{EVENT_HISTORY + 4}'}
    finally:
        srv.shutdown()
