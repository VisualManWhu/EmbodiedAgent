"""Did a forward pulse actually move the robot?

The only obstacle sensor is one narrow forward ultrasonic. What stops the
robot in practice is often something it cannot see — a chair leg caught on a
wheel — and then nothing reports a block: the motors are commanded, the robot
stays where it is, dead reckoning sails on without it. (In the field that
also stalled the motors hard enough to trip the battery protection.)

So motion is checked against two things that do not depend on that sensor:

  fix    a tag fix just before the pulse and one just after: the measured
         displacement, compared with what the pulse should have covered.
         Trustworthy, but only available when a tag was in view both times.
  image  the camera frame before the pulse and after it. A robot that drove
         half a metre sees a different picture; one that did not move sees
         the same one. Available for every pulse.

The image test compares band-pass filtered frames (fine texture: carpet,
edges), not raw brightness — large smooth areas look alike from anywhere, fine
texture only lines up if the camera has not moved.
"""
import math

import cv2
import numpy as np

SIG_SIZE = (320, 240)


def frame_signature(frame):
    """Small band-passed grey image used to compare two views."""
    grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, SIG_SIZE, interpolation=cv2.INTER_AREA)
    grey = grey.astype(np.float32)
    fine = cv2.GaussianBlur(grey, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(grey, (0, 0), 4.0)
    return fine - coarse


def similarity(a, b):
    """``(ncc, texture)`` — normalised correlation of two signatures (1 = the
    same view, ~0 = unrelated) and how much texture there was to compare."""
    sa, sb = float(a.std()), float(b.std())
    texture = min(sa, sb)
    if texture < 1e-6:
        return 0.0, texture
    ncc = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return ncc, texture


class StuckCheck:
    """One forward pulse under observation.

    ``verdict`` stays None until the evidence is in, then is 'moved' or
    'stuck'; ``how`` says which evidence decided ('fix' or 'image').
    """

    SAME_VIEW_NCC = 0.6         # at or above: the camera has not moved
    MIN_TEXTURE = 1.5           # below: too bland a view to judge
    MIN_MOVED_FRAC = 0.3        # of the expected distance, by tag fix
    SETTLE_SEC = 0.3            # after the pulse ends, before looking
    FIX_WAIT_SEC = 1.6          # how long to wait for a confirming fix
    GIVE_UP_SEC = 3.0

    def __init__(self, now, expected_m, pre_pose, pre_sig, pan,
                 pre_fix=None):
        self.t0 = now
        self.expected_m = expected_m
        self.pre_pose = pre_pose        # (x, y, yaw) believed at the start
        self.pre_sig = pre_sig
        self.pan = pan
        self.pre_fix = pre_fix          # (x, y, yaw) from a fresh tag fix
        self.done_t = None
        self.verdict = None
        self.how = None
        self.ncc = None
        self.moved_m = None
        self._image_says = None         # None | 'same' | 'changed' | 'unsure'

    def on_done(self, now):
        if self.done_t is None:
            self.done_t = now

    def wants_camera_still(self, now):
        """True while the 'after' picture has yet to be taken: the camera
        must not be panned away before then."""
        return (self.verdict is None and self._image_says is None
                and self.done_t is not None
                and now < self.done_t + self.GIVE_UP_SEC)

    def evaluate(self, now, sig, pan, frame_ok, fix=None):
        """Feed one frame (and a tag fix ``(x, y, yaw)`` if this frame gave
        one). Returns the verdict once reached, else None."""
        if self.verdict is not None or self.done_t is None:
            return self.verdict
        if now < self.done_t + self.SETTLE_SEC:
            return None

        if fix is not None and self.pre_fix is not None:
            x0, y0, yaw0 = self.pre_fix
            self.moved_m = ((fix[0] - x0) * math.cos(yaw0)
                            + (fix[1] - y0) * math.sin(yaw0))
            return self._decide(
                'moved' if self.moved_m >= self.MIN_MOVED_FRAC * self.expected_m
                else 'stuck', 'fix')

        if self._image_says is None:
            if (frame_ok and sig is not None and self.pre_sig is not None
                    and abs(pan - self.pan) < 2e-3):
                self.ncc, texture = similarity(self.pre_sig, sig)
                if texture < self.MIN_TEXTURE:
                    self._image_says = 'unsure'
                elif self.ncc >= self.SAME_VIEW_NCC:
                    self._image_says = 'same'
                else:
                    self._image_says = 'changed'
            elif now > self.done_t + self.GIVE_UP_SEC:
                self._image_says = 'unsure'

        if self._image_says == 'changed':
            return self._decide('moved', 'image')
        since = now - self.done_t
        if self._image_says == 'same':
            # the picture did not change; give a tag fix a moment to confirm
            # or overrule it before acting on the picture alone
            if self.pre_fix is None or since >= self.FIX_WAIT_SEC:
                return self._decide('stuck', 'image')
            return None
        if self._image_says == 'unsure' and (
                self.pre_fix is None or since >= self.GIVE_UP_SEC):
            return self._decide('moved', 'unknown')     # no evidence: assume ok
        return None

    def _decide(self, verdict, how):
        self.verdict, self.how = verdict, how
        return verdict
