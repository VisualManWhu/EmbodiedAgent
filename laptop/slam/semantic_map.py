"""Semantic object map with multi-view fusion.

Each landmark accumulates observations from many frames so the map does not
depend on single-frame YOLO accuracy:

- **class** is a confidence-weighted vote histogram — transient
  misclassifications are outvoted.
- **position** is bearing-ray triangulated once the landmark has been seen
  from >= 2 distinct viewpoints; a single-view landmark falls back to the
  ground-plane projection of its observations.
- **state** is ``TENTATIVE`` until the landmark has >= ``confirm_min_obs``
  detections from >= ``confirm_min_viewpoints`` distinct viewpoints — a false
  positive that does not reproduce from a second angle never confirms.
- **negative evidence**: a ``CONFIRMED`` landmark repeatedly expected in the
  camera FOV with no matching detection is demoted, then pruned.
"""
import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import projection, transforms

TENTATIVE = 'TENTATIVE'
CONFIRMED = 'CONFIRMED'


@dataclass
class FusionConfig:
    gate_radius_m: float = 0.5          # detection<->landmark association radius
    max_observations: int = 40          # ring-buffer cap per landmark
    confirm_min_obs: int = 4             # N: detections needed to confirm
    confirm_min_viewpoints: int = 2      # M: distinct viewpoints needed
    viewpoint_min_dist_m: float = 0.25   # poses this far apart = distinct
    viewpoint_min_angle_rad: float = 0.35
    tentative_stale_sec: float = 25.0    # prune unconfirmed landmarks idle this long
    miss_demote: int = 4                 # misses -> CONFIRMED demoted to TENTATIVE
    miss_prune: int = 8                  # misses -> landmark removed entirely


@dataclass
class Observation:
    cam_origin: np.ndarray      # camera centre in map frame (3,)
    ray_dir: np.ndarray         # unit bearing ray in map frame (3,)
    ground_xy: np.ndarray       # ground-plane projection (3,), z=0
    robot_pose: tuple           # (x, y, yaw) of the robot base
    det_conf: float
    t: float


@dataclass
class Landmark:
    id: int
    class_votes: dict = field(default_factory=dict)
    observations: list = field(default_factory=list)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    state: str = TENTATIVE
    miss_count: int = 0
    created_t: float = 0.0
    last_seen_t: float = 0.0

    @property
    def label(self) -> str:
        return max(self.class_votes, key=self.class_votes.get) if self.class_votes else '?'

    @property
    def obs_count(self) -> int:
        return len(self.observations)

    @property
    def confidence(self) -> float:
        """Vote purity of the winning class — 1.0 = every vote agrees."""
        total = sum(self.class_votes.values())
        if total <= 0:
            return 0.0
        return self.class_votes[self.label] / total


def _count_viewpoints(poses, min_dist, min_angle) -> int:
    """Greedy count of mutually distinct viewpoints among robot poses."""
    reps = []
    for p in poses:
        if all(projection.viewpoints_distinct(p, r, min_dist, min_angle)
               for r in reps):
            reps.append(p)
    return len(reps)


class SemanticMap:
    def __init__(self, config: FusionConfig | None = None):
        self.cfg = config or FusionConfig()
        self.landmarks: dict[int, Landmark] = {}
        self._next_id = 1
        self._updated_this_frame: set[int] = set()

    # ---- ingest -----------------------------------------------------------

    def add_detection(self, det: dict, T_map_cam: np.ndarray, robot_xytheta,
                      K, now: float | None = None):
        """Fuse one YOLO detection into the map.

        ``det`` — ``{'cls', 'score', 'cx', 'cy', 'w', 'h'}`` (pixels).
        ``T_map_cam`` — camera pose in the map frame.
        Returns the affected ``Landmark``, or ``None`` if the detection could
        not be ground-projected (ray points at/above the horizon).
        """
        now = time.time() if now is None else now
        u = float(det['cx'])
        v = float(det['cy']) + float(det['h']) / 2.0   # bbox bottom-centre
        origin, ray = projection.pixel_ray_in_world(T_map_cam, u, v, K)
        ground = projection.ground_plane_intersect(origin, ray)
        if ground is None:
            return None

        obs = Observation(cam_origin=origin, ray_dir=ray, ground_xy=ground,
                          robot_pose=tuple(robot_xytheta),
                          det_conf=float(det['score']), t=now)
        cls = str(det['cls'])
        lm = self._associate(ground)
        if lm is None:
            lm = Landmark(id=self._next_id, created_t=now)
            self.landmarks[lm.id] = lm
            self._next_id += 1
        self._ingest(lm, obs, cls, now)
        self._updated_this_frame.add(lm.id)
        return lm

    def _associate(self, ground_xy: np.ndarray):
        """Nearest existing landmark within the gating radius, or None.

        Association is by position, NOT class: a single-frame misclassification
        of an already-mapped object must join that object's landmark so its
        wrong class becomes an outvoted minority — not spawn a phantom. Two
        distinct same-class objects closer than the gate radius merging into
        one landmark is the documented out-of-scope case.
        """
        best, best_d = None, self.cfg.gate_radius_m
        for lm in self.landmarks.values():
            d = float(np.hypot(*(lm.position[:2] - ground_xy[:2])))
            if d <= best_d:
                best, best_d = lm, d
        return best

    def _ingest(self, lm: Landmark, obs: Observation, cls: str, now: float):
        lm.observations.append(obs)
        if len(lm.observations) > self.cfg.max_observations:
            lm.observations.pop(0)
        lm.class_votes[cls] = lm.class_votes.get(cls, 0.0) + obs.det_conf
        lm.last_seen_t = now
        lm.miss_count = max(0, lm.miss_count - 1)   # recover on a fresh hit
        self._refit(lm)

    def _refit(self, lm: Landmark):
        """Re-fuse position and re-evaluate the confirmation state."""
        origins = [o.cam_origin for o in lm.observations]
        dirs = [o.ray_dir for o in lm.observations]
        pos = projection.triangulate_rays(origins, dirs)
        if pos is None:   # single viewpoint / degenerate -> ground-plane median
            pos = np.median([o.ground_xy for o in lm.observations], axis=0)
        lm.position = np.asarray(pos, dtype=float)

        n_views = _count_viewpoints([o.robot_pose for o in lm.observations],
                                    self.cfg.viewpoint_min_dist_m,
                                    self.cfg.viewpoint_min_angle_rad)
        if (lm.state == TENTATIVE
                and lm.obs_count >= self.cfg.confirm_min_obs
                and n_views >= self.cfg.confirm_min_viewpoints):
            lm.state = CONFIRMED

    # ---- per-frame bookkeeping -------------------------------------------

    def end_frame(self, T_map_cam: np.ndarray, K, image_wh, now: float | None = None):
        """Apply negative evidence and prune. Call once after all of a frame's
        detections have been added via :meth:`add_detection`."""
        now = time.time() if now is None else now
        w, h = image_wh
        for lm in list(self.landmarks.values()):
            if lm.id in self._updated_this_frame:
                continue
            # negative evidence: expected in view this frame but not detected.
            if self._in_fov(lm, T_map_cam, K, w, h):
                lm.miss_count += 1
            if lm.miss_count >= self.cfg.miss_prune:
                del self.landmarks[lm.id]
            elif lm.state == CONFIRMED and lm.miss_count >= self.cfg.miss_demote:
                lm.state = TENTATIVE
            elif (lm.state == TENTATIVE
                  and now - lm.last_seen_t > self.cfg.tentative_stale_sec):
                del self.landmarks[lm.id]
        self._updated_this_frame.clear()

    def _in_fov(self, lm: Landmark, T_map_cam, K, w, h) -> bool:
        px = projection.world_to_pixel(T_map_cam, K, lm.position)
        if px is None:
            return False
        u, v = px
        return 0 <= u < w and 0 <= v < h

    # ---- output -----------------------------------------------------------

    def snapshot(self, confirmed_only: bool = False) -> list:
        """Lightweight list of landmarks for rendering / POSTing to the Pi."""
        out = []
        for lm in self.landmarks.values():
            if confirmed_only and lm.state != CONFIRMED:
                continue
            out.append({
                'id': lm.id,
                'label': lm.label,
                'x': float(lm.position[0]),
                'y': float(lm.position[1]),
                'z': float(lm.position[2]),
                'state': lm.state,
                'obs_count': lm.obs_count,
                'confidence': round(lm.confidence, 3),
            })
        return out

    # ---- persistence ------------------------------------------------------

    def save(self, path: str):
        data = {'next_id': self._next_id, 'landmarks': []}
        for lm in self.landmarks.values():
            data['landmarks'].append({
                'id': lm.id,
                'class_votes': lm.class_votes,
                'position': lm.position.tolist(),
                'state': lm.state,
                'miss_count': lm.miss_count,
                'created_t': lm.created_t,
                'last_seen_t': lm.last_seen_t,
                'observations': [{
                    'cam_origin': o.cam_origin.tolist(),
                    'ray_dir': o.ray_dir.tolist(),
                    'ground_xy': o.ground_xy.tolist(),
                    'robot_pose': list(o.robot_pose),
                    'det_conf': o.det_conf,
                    't': o.t,
                } for o in lm.observations],
            })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str, config: FusionConfig | None = None) -> 'SemanticMap':
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        m = cls(config)
        m._next_id = data.get('next_id', 1)
        for ld in data.get('landmarks', []):
            lm = Landmark(
                id=ld['id'],
                class_votes={k: float(v) for k, v in ld['class_votes'].items()},
                position=np.asarray(ld['position'], dtype=float),
                state=ld['state'],
                miss_count=ld.get('miss_count', 0),
                created_t=ld.get('created_t', 0.0),
                last_seen_t=ld.get('last_seen_t', 0.0),
                observations=[Observation(
                    cam_origin=np.asarray(o['cam_origin'], dtype=float),
                    ray_dir=np.asarray(o['ray_dir'], dtype=float),
                    ground_xy=np.asarray(o['ground_xy'], dtype=float),
                    robot_pose=tuple(o['robot_pose']),
                    det_conf=o['det_conf'],
                    t=o['t'],
                ) for o in ld.get('observations', [])],
            )
            m.landmarks[lm.id] = lm
        return m
