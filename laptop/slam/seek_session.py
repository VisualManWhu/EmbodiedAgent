"""Find-an-object-by-class state machine: map navigation + visual search as
ONE behaviour instead of "spin for N seconds, then fall back to the map".

Division of labour
  * The map is a PRIOR: it says roughly where the object was last fused, so
    the robot has somewhere to go when the object is out of view.
  * Vision is GROUND TRUTH: whenever the object is actually in frame it wins —
    navigation is cancelled and the Pi's visual-servo approach takes over.
    Arrival is only ever declared with the object in sight.

Flow
  1. target confirmed in view            -> VISUAL (Pi `find`: approach)
  2. not in view, map has the class      -> NAV to the nearest landmark, with
                                            vision watching the whole way
  3. not in view, map has nothing        -> VISUAL (Pi `find`: spin-search)

The session is pure logic. ``tick`` / ``on_nav_done`` return a list of actions
for the caller (SlamRunner) to execute:

  ('find', cls)        POST {'action': 'find', 'target': cls} to the Pi
  ('goto', landmark)   start a NavSession toward that landmark dict
  ('nav_cancel',)      drop the active NavSession (vision is taking over)
  ('stop',)            POST {'action': 'stop'} — halt the Pi's search
"""
from dataclasses import dataclass, field
from enum import Enum


class SeekState(str, Enum):
    INIT = 'INIT'
    VISUAL = 'VISUAL'
    NAV = 'NAV'
    ARRIVED = 'ARRIVED'
    FAILED = 'FAILED'


@dataclass
class SeekConfig:
    # Vision must CONFIRM before it overrides navigation: YOLO misclassifies
    # single frames, and one stray box must not hijack a goto.
    confirm_hits: int = 3
    confirm_window_sec: float = 2.0
    # Hysteresis the other way: after vision takes over, tolerate this long
    # without a sighting (covers the Pi's own local re-acquire sweep) before
    # handing back to the map.
    visual_lost_sec: float = 12.0
    # Blind spin-search budget when the map has no landmark of the class.
    # One stop-and-scan revolution is ~45 s; allow two.
    search_sec: float = 90.0
    # Spin-search budget at a spot the map sent us to where the object was
    # not seen (moved / stale landmark), or after a failed navigation.
    recheck_sec: float = 50.0
    # A map arrival counts as "seen" if the class was in frame this recently.
    # Generous on purpose: the camera sits ~15 cm off the floor, so in the
    # last metre it only sees chair legs / a box side and YOLO goes blind.
    arrival_seen_window_sec: float = 10.0
    # Close-loss arrival. Same blindness during a visual approach: the object
    # grows until it no longer looks like itself, detection drops out, and
    # the Pi (which only arrives on a live detection) starts spinning next to
    # it. So: the last sighting was CLOSE and nothing has been seen for
    # close_lost_sec -> we are there.
    close_lost_sec: float = 3.0
    # Thresholds are bbox height / frame height. Field-calibrated on this
    # robot: a chair read 0.46 at a measured 1.3 m, i.e. roughly
    # h_frac ~= 0.28 + 0.19 / distance_m for anything tall enough to run off
    # the top of the frame (and ~0.31 / d for a 25 cm bottle), so:
    #   0.55 -> ~0.7 m     0.62 -> ~0.55 m     0.75 -> ~0.4 m
    # The Pi's own arrival needs 0.75 with a live detection — at 0.4 m YOLO
    # has usually already lost the object, which is why it never fired.
    # "Cut by the frame edge" is NOT evidence of closeness: with the camera
    # this low a chair top leaves the frame from ~2 m away.
    close_h_frac: float = 0.55          # last sighting this big, then blind
    arrive_h_frac: float = 0.62         # live sighting this big -> arrived
    arrive_fresh_sec: float = 0.6
    total_timeout_sec: float = 420.0


@dataclass
class SeekSession:
    target_class: str
    config: SeekConfig = field(default_factory=SeekConfig)
    state: SeekState = SeekState.INIT
    via: str | None = None              # 'vision' | 'map' once ARRIVED
    goal_id: int | None = None          # landmark being / last navigated to
    fail_reason: str | None = None
    tried: set = field(default_factory=set)      # landmark ids already visited
    unseen_ids: list = field(default_factory=list)   # reached, object absent
    trail: list = field(default_factory=list)    # human-readable history
    _started_t: float | None = None
    _visual_deadline: float = 0.0
    _visual_entered_t: float = 0.0

    @property
    def done(self) -> bool:
        return self.state in (SeekState.ARRIVED, SeekState.FAILED)

    # ---- helpers -------------------------------------------------------

    def _confirmed(self, now, hits) -> bool:
        recent = [t for t in hits if t >= now - self.config.confirm_window_sec]
        return len(recent) >= self.config.confirm_hits

    def _next_candidate(self, candidates):
        for c in candidates:
            if c['id'] not in self.tried:
                return c
        return None

    def _enter_visual(self, now, budget_sec, note):
        self.state = SeekState.VISUAL
        self._visual_deadline = now + budget_sec
        self._visual_entered_t = now
        self.trail.append(note)
        return [('find', self.target_class)]

    def _enter_nav(self, cand):
        self.state = SeekState.NAV
        self.goal_id = cand['id']
        self.trail.append(f"nav:{cand['id']}")
        return [('goto', cand)]

    def _fail(self, reason):
        was_nav = self.state is SeekState.NAV
        self.state = SeekState.FAILED
        self.fail_reason = reason
        self.trail.append(f'failed:{reason}')
        return ([('nav_cancel',)] if was_nav else []) + [('stop',)]

    # ---- inputs --------------------------------------------------------

    def tick(self, now, hits, candidates, sight=None):
        """``hits``: recent timestamps at which the target class was detected.
        ``candidates``: confirmed map landmarks of the class, nearest first,
        as dicts with at least ``id``.
        ``sight``: the latest sighting of the class, or None —
        ``{'t': time, 'h_frac': bbox height / frame height,
        'clipped': bbox cut by the top or bottom frame edge}``."""
        if self.done:
            return []
        if self._started_t is None:
            self._started_t = now
        if now - self._started_t > self.config.total_timeout_sec:
            return self._fail('timeout')

        if self.state is SeekState.INIT:
            if self._confirmed(now, hits):
                return self._enter_visual(now, self.config.visual_lost_sec,
                                          'vision')
            cand = self._next_candidate(candidates)
            if cand is not None:
                return self._enter_nav(cand)
            return self._enter_visual(now, self.config.search_sec, 'search')

        if self.state is SeekState.VISUAL:
            if hits:
                # every sighting extends the visual phase
                self._visual_deadline = max(
                    self._visual_deadline,
                    max(hits) + self.config.visual_lost_sec)
            if (sight is not None and sight['t'] >= self._visual_entered_t
                    - self.config.confirm_window_sec):
                age = now - sight['t']
                note = None
                if (age <= self.config.arrive_fresh_sec
                        and sight['h_frac'] >= self.config.arrive_h_frac):
                    note = 'arrived:near'       # in view and close enough
                elif (age >= self.config.close_lost_sec
                        and sight['h_frac'] >= self.config.close_h_frac):
                    note = 'arrived:close'      # went blind right next to it
                if note is not None:
                    self.state = SeekState.ARRIVED
                    self.via = 'vision'
                    self.trail.append(note)
                    return [('stop',)]
            if now <= self._visual_deadline:
                return []
            cand = self._next_candidate(candidates)
            if cand is not None:
                return self._enter_nav(cand)
            return self._fail('not_found')

        # NAV — vision watches the whole way and wins on a confirmed sighting
        if self._confirmed(now, hits):
            return [('nav_cancel',)] + self._enter_visual(
                now, self.config.visual_lost_sec, 'vision')
        return []

    def on_pi_event(self, event):
        """The Pi posts ``arrived:<class>`` when its visual approach ends with
        the object filling the frame (or the sonar tripping on it)."""
        if (self.state is SeekState.VISUAL
                and event == f'arrived:{self.target_class}'):
            self.state = SeekState.ARRIVED
            self.via = 'vision'
            self.trail.append('arrived')

    def on_nav_done(self, nav_state, now, hits):
        """The NavSession started by a ('goto', ...) action finished."""
        if self.state is not SeekState.NAV:
            return []
        self.tried.add(self.goal_id)
        seen = any(t >= now - self.config.arrival_seen_window_sec
                   for t in hits)
        if nav_state == 'ARRIVED' and seen:
            self.state = SeekState.ARRIVED
            self.via = 'map'
            self.trail.append('arrived')
            return []
        if nav_state == 'ARRIVED':
            # At the recorded position but the object is not there: the map is
            # stale. Look around here before trying anything else.
            self.unseen_ids.append(self.goal_id)
            note = f'unseen:{self.goal_id}'
        else:
            note = f'nav_failed:{self.goal_id}'
        return self._enter_visual(now, self.config.recheck_sec, note)
