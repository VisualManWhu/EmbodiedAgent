from pi_events import EventCursor


def _status(seq, events, boot='b1'):
    return {'mode': 'IDLE', 'boot_id': boot, 'event_seq': seq,
            'events': [{'seq': s, 'event': e} for s, e in events],
            'event': events[-1][1] if events else ''}


def test_sync_skips_history_then_delivers_new_events_once():
    c = EventCursor()
    c.sync(_status(3, [(2, 'pulse_done'), (3, 'arrived:chair')]))
    assert c.take(_status(3, [(2, 'pulse_done'), (3, 'arrived:chair')])) == []
    st = _status(4, [(2, 'pulse_done'), (3, 'arrived:chair'),
                     (4, 'arrived:bicycle')])
    assert c.take(st) == ['arrived:bicycle']
    assert c.take(st) == []                     # not delivered twice


def test_two_cursors_do_not_steal_from_each_other():
    """The regression: a second reader polling /status must not hide the
    arrival from the bot."""
    bot, other = EventCursor(), EventCursor()
    base = _status(0, [])
    bot.sync(base)
    other.sync(base)
    st = _status(1, [(1, 'arrived:bicycle')])
    assert other.take(st) == ['arrived:bicycle']
    assert bot.take(st) == ['arrived:bicycle']


def test_multiple_events_between_polls_all_delivered_in_order():
    c = EventCursor()
    c.sync(_status(0, []))
    st = _status(2, [(1, 'blocked:front'), (2, 'pulse_done')])
    assert c.take(st) == ['blocked:front', 'pulse_done']


def test_unsynced_cursor_starts_from_now():
    c = EventCursor()
    c.sync(None)                                # Pi unreachable at sync time
    assert c.take(_status(5, [(5, 'arrived:old')])) == []
    assert c.take(_status(6, [(5, 'arrived:old'), (6, 'arrived:new')])) == [
        'arrived:new']


def test_pi_restart_resets_sequence():
    c = EventCursor()
    c.sync(_status(40, [(40, 'pulse_done')], boot='b1'))
    st = _status(2, [(1, 'photo_ready:1'), (2, 'photo_ready:2')], boot='b2')
    assert c.take(st) == ['photo_ready:1', 'photo_ready:2']


def test_none_status_and_legacy_oneshot_server():
    c = EventCursor()
    assert c.take(None) == []
    assert c.take({'mode': 'IDLE', 'event': ''}) == []
    assert c.take({'mode': 'IDLE', 'event': 'arrived:cup'}) == ['arrived:cup']
