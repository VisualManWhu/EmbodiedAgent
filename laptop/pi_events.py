"""Reader-side cursor over the Pi agent's sequenced /status events.

The Pi keeps the last few events, each with an increasing ``seq``; a GET does
not consume them. Every independent reader owns one EventCursor, which
remembers the highest seq it has handled, so readers never steal events from
each other.

Typical use::

    cursor = EventCursor()
    cursor.sync(get_status())        # BEFORE sending the command: ignore history
    post_command(...)
    for event in cursor.take(get_status()):
        ...
"""


class EventCursor:
    def __init__(self):
        self._boot_id = None
        self._seq = None            # None = not positioned yet

    def sync(self, status):
        """Skip everything the Pi has emitted so far. ``status`` may be None
        (Pi unreachable) — the cursor then positions itself on first take()."""
        if not status or 'event_seq' not in status:
            return
        self._boot_id = status.get('boot_id')
        self._seq = int(status['event_seq'])

    def take(self, status):
        """Return the event strings in ``status`` not yet seen, oldest first."""
        if not status:
            return []
        if 'events' not in status:
            # Legacy Pi build: single one-shot event, cleared by the GET.
            event = status.get('event') or ''
            return [event] if event else []
        boot_id = status.get('boot_id')
        if self._seq is None:
            # Never positioned (sync failed): start from "now" rather than
            # replaying stale history as if it were a response to our command.
            self.sync(status)
            return []
        if boot_id != self._boot_id:
            # Pi restarted — its counter started over; everything is new.
            self._boot_id = boot_id
            self._seq = 0
        fresh = [e for e in status['events'] if e['seq'] > self._seq]
        if fresh:
            self._seq = fresh[-1]['seq']
        return [e['event'] for e in fresh]
