"""Unit tests for the semantic-SLAM geometry and fusion modules.

Mostly synthetic — forward-projects known geometry, feeds it through the
pipeline, and checks the recovered values. One end-to-end test renders a real
tag36h11 marker and round-trips it through the AprilTag library to verify the
detector's corner order matches ``pose_estimator``'s object points.
"""
import cv2
import numpy as np
import pytest

from slam import projection, transforms
from slam.pose_estimator import (estimate_camera_pose, tag_object_points,
                                 tag_pose_in_map)
from slam.semantic_map import CONFIRMED, TENTATIVE, FusionConfig, SemanticMap

K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0, 0.0, 1.0]])


def look_at(eye, target):
    """Camera pose (T_map_cam) at ``eye`` looking at ``target``.

    Optical frame: z forward (eye->target), x right, y down.
    """
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.column_stack([right, down, fwd])
    return transforms.make_transform(R, eye)


def project(T_map_cam, point):
    """Forward-project a world point to a pixel (no distortion)."""
    p_cam = transforms.apply_point(transforms.invert(T_map_cam), point)
    uvw = K @ p_cam
    return uvw[0] / uvw[2], uvw[1] / uvw[2]


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------

def test_invert_roundtrip():
    T = transforms.make_transform(transforms.rot_z(0.7) @ transforms.rot_x(0.2),
                                  [1.0, -2.0, 0.5])
    assert np.allclose(T @ transforms.invert(T), np.eye(4), atol=1e-9)


def test_camera_in_base_centered():
    mount = {'forward_m': 0.05, 'lateral_m': 0.0, 'height_m': 0.30}
    T = transforms.camera_pose_in_base(0.0, 0.0, mount)
    # camera centre at the mount point
    assert np.allclose(T[:3, 3], [0.05, 0.0, 0.30])
    # optical +z (forward) maps to robot +x (forward)
    assert np.allclose(transforms.apply_direction(T, [0, 0, 1]), [1, 0, 0],
                       atol=1e-9)


def test_camera_in_base_pan():
    mount = {'forward_m': 0.0, 'lateral_m': 0.0, 'height_m': 0.3}
    T = transforms.camera_pose_in_base(np.pi / 2, 0.0, mount)
    # pan +90deg: camera forward now points to robot +y (left)
    assert np.allclose(transforms.apply_direction(T, [0, 0, 1]), [0, 1, 0],
                       atol=1e-9)


def test_pose_xytheta():
    T = transforms.make_transform(transforms.rot_z(0.6), [3.0, 1.0, 0.0])
    x, y, yaw = transforms.pose_xytheta(T)
    assert (x, y) == pytest.approx((3.0, 1.0))
    assert yaw == pytest.approx(0.6)


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def test_pixel_to_ray_centre():
    ray = projection.pixel_to_ray_cam(320.0, 240.0, K)
    assert np.allclose(ray, [0, 0, 1], atol=1e-9)


def test_ground_plane_intersect():
    origin = np.array([0.0, 0.0, 2.0])
    direction = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
    pt = projection.ground_plane_intersect(origin, direction)
    assert np.allclose(pt, [2.0, 0.0, 0.0], atol=1e-9)


def test_ground_plane_intersect_upward_ray_is_none():
    assert projection.ground_plane_intersect([0, 0, 1], [0, 0, 1]) is None


def test_triangulate_elevated_point():
    """Triangulation recovers a point off the floor that ground-plane
    projection alone could not place."""
    true_pt = np.array([2.0, 1.0, 0.8])
    eyes = [(-1.0, 0.0, 0.5), (0.0, 3.0, 0.6), (3.0, -1.0, 0.4)]
    origins, dirs = [], []
    for e in eyes:
        e = np.array(e)
        origins.append(e)
        d = true_pt - e
        dirs.append(d / np.linalg.norm(d))
    est = projection.triangulate_rays(origins, dirs)
    assert np.allclose(est, true_pt, atol=1e-6)


def test_triangulate_parallel_rays_degenerate():
    origins = [[0, 0, 0], [0, 1, 0]]
    dirs = [[1, 0, 0], [1, 0, 0]]
    assert projection.triangulate_rays(origins, dirs) is None


def test_world_to_pixel_roundtrip():
    T = look_at([0, 0, 0.5], [3, 0, 0.5])
    pt = np.array([3.0, 0.4, 0.2])
    u, v = projection.world_to_pixel(T, K, pt)
    o, d = projection.pixel_ray_in_world(T, u, v, K)
    # the pixel's ray must pass through the original point
    to_pt = pt - o
    along = to_pt @ d
    perp = np.linalg.norm(to_pt - along * d)
    assert perp < 1e-6


def test_viewpoints_distinct():
    a = (0.0, 0.0, 0.0)
    assert not projection.viewpoints_distinct(a, (0.1, 0.0, 0.05), 0.25, 0.35)
    assert projection.viewpoints_distinct(a, (1.0, 0.0, 0.0), 0.25, 0.35)
    assert projection.viewpoints_distinct(a, (0.0, 0.0, 0.9), 0.25, 0.35)


# --------------------------------------------------------------------------
# pose_estimator
# --------------------------------------------------------------------------

TAG_MAP = {
    0: {'x': 2.0, 'y': 0.0, 'z': 0.5, 'yaw_deg': 180.0, 'size_m': 0.16},
    1: {'x': 1.5, 'y': 1.5, 'z': 0.5, 'yaw_deg': 270.0, 'size_m': 0.16},
}


def _project_tag_corners(T_map_cam, tag_id):
    entry = TAG_MAP[tag_id]
    T_map_tag = tag_pose_in_map(entry)
    obj = tag_object_points(entry['size_m'])
    corners = []
    for p in obj:
        world = transforms.apply_point(T_map_tag, p)
        corners.append(project(T_map_cam, world))
    return np.array(corners)


def test_pose_estimator_single_tag():
    T_true = transforms.make_transform(transforms.R_ROBOT_FROM_OPTICAL,
                                       [0.0, 0.0, 0.5])
    dets = [{'id': 0, 'corners': _project_tag_corners(T_true, 0)}]
    result = estimate_camera_pose(dets, TAG_MAP, K)
    assert result is not None
    T_est, err, n = result
    assert n == 1
    assert err < 1e-3
    assert np.allclose(T_est, T_true, atol=1e-3)


def test_pose_estimator_two_tags():
    T_true = transforms.make_transform(
        transforms.rot_z(0.3) @ transforms.R_ROBOT_FROM_OPTICAL,
        [0.4, 0.2, 0.5])
    dets = [{'id': 0, 'corners': _project_tag_corners(T_true, 0)},
            {'id': 1, 'corners': _project_tag_corners(T_true, 1)}]
    result = estimate_camera_pose(dets, TAG_MAP, K)
    assert result is not None
    T_est, err, n = result
    assert n == 2
    assert np.allclose(T_est[:3, 3], T_true[:3, 3], atol=5e-3)


def test_pose_estimator_unknown_tag_ignored():
    T_true = transforms.make_transform(transforms.R_ROBOT_FROM_OPTICAL,
                                       [0.0, 0.0, 0.5])
    dets = [{'id': 99, 'corners': _project_tag_corners(T_true, 0)}]
    assert estimate_camera_pose(dets, TAG_MAP, K) is None


def test_real_apriltag_corner_order_and_pose():
    """End-to-end: render a real tag36h11 marker into a synthetic camera view,
    detect it with the AprilTag library, and recover the known camera pose.

    This verifies the live detector's corner order matches
    ``pose_estimator.tag_object_points`` — the one assumption that synthetic
    tests cannot check.
    """
    pytest.importorskip('pupil_apriltags')
    if not hasattr(cv2, 'aruco'):
        pytest.skip('cv2.aruco unavailable')
    from slam.apriltag_detector import AprilTagDetector

    # tag 0.8 m ahead so it is large in frame (sharp pose); camera 0.5 m high
    tag_map = {0: {'x': 0.8, 'y': 0.0, 'z': 0.5, 'yaw_deg': 180.0,
                   'size_m': 0.16}}
    T_true = transforms.make_transform(transforms.R_ROBOT_FROM_OPTICAL,
                                       [0.0, 0.0, 0.5])
    obj = tag_object_points(tag_map[0]['size_m'])
    dst = np.array([project(T_true, transforms.apply_point(
        tag_pose_in_map(tag_map[0]), p)) for p in obj], np.float32)

    # marker-image pixel corners for obj order bl, br, tr, tl (image y is down)
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    S = 240
    marker = cv2.aruco.generateImageMarker(adict, 0, S)
    src = np.array([[0, S], [S, S], [S, 0], [0, 0]], np.float32)

    H, _ = cv2.findHomography(src, dst)
    canvas = np.full((480, 640), 255, np.uint8)        # white = quiet zone
    cv2.warpPerspective(marker, H, (640, 480), dst=canvas,
                        borderMode=cv2.BORDER_TRANSPARENT)

    dets = AprilTagDetector().detect(canvas)
    assert len(dets) == 1 and dets[0]['id'] == 0

    result = estimate_camera_pose(dets, tag_map, K)
    assert result is not None
    T_est, err, _ = result
    assert err < 2.0                                    # px reprojection error
    assert np.allclose(T_est[:3, 3], T_true[:3, 3], atol=0.03)
    assert np.allclose(T_est[:3, :3], T_true[:3, :3], atol=0.03)


# --------------------------------------------------------------------------
# semantic_map fusion
# --------------------------------------------------------------------------

def _det(cls, u, v, score=0.9):
    """Detection whose bbox bottom-centre is exactly pixel (u, v)."""
    return {'cls': cls, 'score': score, 'cx': u, 'cy': v, 'w': 20.0, 'h': 0.0}


def _observe(smap, obj_pt, eye, cls='chair', score=0.9, now=0.0):
    T = look_at(eye, obj_pt)
    u, v = project(T, obj_pt)
    pose = (eye[0], eye[1], np.arctan2(obj_pt[1] - eye[1], obj_pt[0] - eye[0]))
    return smap.add_detection(_det(cls, u, v, score), T, pose, K, now=now)


def test_fusion_triangulates_one_landmark():
    obj = np.array([3.0, 0.5, 0.0])
    eyes = [(0.0, 0.0, 0.5), (0.0, 1.6, 0.5), (1.2, -1.4, 0.5)]
    smap = SemanticMap()
    for e in eyes:
        _observe(smap, obj, e)
    assert len(smap.landmarks) == 1
    lm = next(iter(smap.landmarks.values()))
    assert np.allclose(lm.position[:2], obj[:2], atol=0.05)
    assert lm.label == 'chair'


def test_fusion_class_voting_outvotes_misdetections():
    obj = np.array([3.0, 0.5, 0.0])
    eyes = [(0.0, 0.0, 0.5), (0.0, 1.6, 0.5), (1.2, -1.4, 0.5),
            (2.0, 2.0, 0.5), (-1.0, 1.0, 0.5), (1.0, 3.0, 0.5)]
    classes = ['chair', 'chair', 'bench', 'chair', 'couch', 'chair']
    smap = SemanticMap()
    for e, c in zip(eyes, classes):
        _observe(smap, obj, e, cls=c)
    assert len(smap.landmarks) == 1
    lm = next(iter(smap.landmarks.values()))
    assert lm.label == 'chair'          # 4 chair votes outvote bench+couch
    assert lm.confidence == pytest.approx(4 / 6, abs=1e-6)


def test_confirmation_needs_distinct_viewpoints():
    """Many detections from ONE spot stay TENTATIVE; detections from a second
    distinct viewpoint promote to CONFIRMED."""
    obj = np.array([3.0, 0.5, 0.0])
    smap = SemanticMap(FusionConfig(confirm_min_obs=4,
                                    confirm_min_viewpoints=2))
    for _ in range(5):
        _observe(smap, obj, (0.0, 0.0, 0.5))
    lm = next(iter(smap.landmarks.values()))
    assert lm.obs_count == 5
    assert lm.state == TENTATIVE         # 5 frames, but only 1 viewpoint

    for _ in range(2):
        _observe(smap, obj, (1.5, 0.0, 0.5))
    assert lm.state == CONFIRMED          # second viewpoint -> confirmed


def test_false_positive_pruned():
    obj = np.array([3.0, 0.5, 0.0])
    smap = SemanticMap(FusionConfig(tentative_stale_sec=25.0))
    _observe(smap, obj, (0.0, 0.0, 0.5), now=0.0)
    assert len(smap.landmarks) == 1
    T = look_at((10.0, 10.0, 0.5), (11.0, 10.0, 0.5))   # looking elsewhere
    # the frame it was seen in does not penalise it
    smap.end_frame(T, K, (640, 480), now=0.5)
    assert len(smap.landmarks) == 1
    # a later frame with no detection of it -> stale -> pruned
    smap.end_frame(T, K, (640, 480), now=100.0)
    assert len(smap.landmarks) == 0


def test_negative_evidence_demotes_then_prunes():
    obj = np.array([3.0, 0.5, 0.0])
    cfg = FusionConfig(confirm_min_obs=4, confirm_min_viewpoints=2,
                       miss_demote=4, miss_prune=8)
    smap = SemanticMap(cfg)
    for e in [(0.0, 0.0, 0.5), (0.0, 0.1, 0.5),
              (1.5, 0.0, 0.5), (1.5, 0.1, 0.5)]:
        _observe(smap, obj, e)
    lm = next(iter(smap.landmarks.values()))
    assert lm.state == CONFIRMED

    # camera keeps the landmark in view but never detects it again
    T = look_at((0.0, 0.5, 0.5), obj)
    smap.end_frame(T, K, (640, 480), now=0.5)   # frame it was last seen in
    assert lm.state == CONFIRMED
    for i in range(4):                           # 4 misses -> demote
        smap.end_frame(T, K, (640, 480), now=float(i + 1))
    assert lm.state == TENTATIVE
    for i in range(4):                           # 4 more -> miss_prune=8
        smap.end_frame(T, K, (640, 480), now=float(i + 5))
    assert len(smap.landmarks) == 0


def test_map_renderer_smoke():
    from slam.map_renderer import MapRenderer
    r = MapRenderer()
    r.update_path(0.0, 0.0)
    r.update_path(1.0, 0.5)
    tags = [{'id': 0, 'x': 0.0, 'y': 0.0}, {'id': 1, 'x': 2.4, 'y': 0.0}]
    landmarks = [
        {'id': 1, 'label': 'chair', 'x': 1.5, 'y': 1.2,
         'state': 'CONFIRMED', 'confidence': 0.9},
        {'id': 2, 'label': 'bottle', 'x': 0.4, 'y': 2.0,
         'state': 'TENTATIVE', 'confidence': 0.4},
    ]
    img = r.render(tags, landmarks, robot=(1.0, 0.5, 0.3))
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0


def test_persistence_roundtrip(tmp_path):
    obj = np.array([3.0, 0.5, 0.0])
    smap = SemanticMap()
    for e in [(0.0, 0.0, 0.5), (1.5, 0.0, 0.5), (0.5, 1.5, 0.5)]:
        _observe(smap, obj, e)
    path = str(tmp_path / 'semantic_map.json')
    smap.save(path)
    loaded = SemanticMap.load(path)
    assert len(loaded.landmarks) == 1
    a = next(iter(smap.landmarks.values()))
    b = next(iter(loaded.landmarks.values()))
    assert b.label == a.label
    assert np.allclose(b.position, a.position)
    assert b.obs_count == a.obs_count


def test_detection_projected_absurdly_far_is_rejected():
    """Field bug: a bbox whose bottom edge sits near the horizon intersects
    the ground hundreds of metres away. One such landmark at (96, -278) blew
    the rendered map up to 31070 x 10662 px."""
    smap = SemanticMap()
    far = np.array([120.0, 3.0, 0.0])
    assert _observe(smap, far, (0.0, 0.0, 0.5)) is None
    assert len(smap.landmarks) == 0
    near = np.array([3.0, 0.5, 0.0])
    assert _observe(smap, near, (0.0, 0.0, 0.5)) is not None


# --------------------------------------------------------------------------
# pose_estimator: prefer obliquely viewed tags for orientation
# --------------------------------------------------------------------------

from slam.pose_estimator import PoseEstimator, tag_incidence_rad  # noqa: E402

OBLIQUE_MAP = {
    # straight ahead of the camera below, seen face-on
    10: {'x': 2.0, 'y': 0.0, 'z': 0.3, 'yaw_deg': 180.0, 'size_m': 0.11},
    # on a side wall, seen at a steep angle
    11: {'x': 2.0, 'y': 0.55, 'z': 0.3, 'yaw_deg': 270.0, 'size_m': 0.11},
}
CAM_TRUE = transforms.make_transform(transforms.R_ROBOT_FROM_OPTICAL,
                                     [0.8, 0.0, 0.3])


def _corners(T_map_cam, tag_id, tag_map):
    entry = tag_map[tag_id]
    T_map_tag = tag_pose_in_map(entry)
    return np.array([project(T_map_cam, transforms.apply_point(T_map_tag, p))
                     for p in tag_object_points(entry['size_m'])])


def _cam_yaw(T):
    return float(np.arctan2(T[1, 2], T[0, 2]))


def _swung_about_tag(T_map_cam, tag_entry, angle):
    """The camera pose a face-on tag is easily mistaken for: swung around the
    tag by ``angle``. The tag looks almost identical from there."""
    c = np.array([tag_entry['x'], tag_entry['y'], 0.0])
    to = transforms.make_transform(np.eye(3), c)
    back = transforms.make_transform(np.eye(3), -c)
    spin = transforms.make_transform(transforms.rot_z(angle), [0, 0, 0])
    return to @ spin @ back @ T_map_cam


def test_incidence_is_zero_face_on_and_large_from_the_side():
    from slam.pose_estimator import _solve_single
    for tag_id, lo, hi in ((10, 0.0, 3.0), (11, 55.0, 75.0)):
        T_cam_tag, _ = _solve_single(_corners(CAM_TRUE, tag_id, OBLIQUE_MAP),
                                     0.11, K, np.zeros(5))
        assert lo <= np.degrees(tag_incidence_rad(T_cam_tag)) <= hi


def test_orientation_comes_from_the_oblique_tag_not_the_lowest_error_one():
    """Field result: a tag seen face-on gave headings scattered over 10 deg
    while tags seen from the side were steady to 0.1 deg — yet the face-on one
    often had the lower reprojection error and was the one trusted."""
    wrong = _swung_about_tag(CAM_TRUE, OBLIQUE_MAP[10], np.radians(6.0))
    rng = np.random.default_rng(0)
    dets = [
        # face-on tag: a perfect fit (error ~0) to a pose 6 deg off
        {'id': 10, 'corners': _corners(wrong, 10, OBLIQUE_MAP)},
        # oblique tag: the true pose, with a little pixel noise
        {'id': 11, 'corners': _corners(CAM_TRUE, 11, OBLIQUE_MAP)
         + rng.normal(0.0, 0.05, (4, 2))},
    ]
    T_est, _, n = estimate_camera_pose(dets, OBLIQUE_MAP, K)
    assert n == 2
    assert abs(np.degrees(_cam_yaw(T_est) - _cam_yaw(CAM_TRUE))) < 1.5


def test_pose_reports_whether_its_heading_is_trustworthy():
    est = PoseEstimator(OBLIQUE_MAP, K, mount={
        'forward_m': 0.0, 'lateral_m': 0.0, 'height_m': CAM_TRUE[2, 3]})
    face_on = est.estimate([{'id': 10,
                             'corners': _corners(CAM_TRUE, 10, OBLIQUE_MAP)}])
    assert face_on['yaw_reliable'] is False
    assert np.degrees(face_on['incidence_rad']) < 5.0
    side = est.estimate([{'id': 11,
                          'corners': _corners(CAM_TRUE, 11, OBLIQUE_MAP)}])
    assert side['yaw_reliable'] is True
    both = est.estimate([
        {'id': 10, 'corners': _corners(CAM_TRUE, 10, OBLIQUE_MAP)},
        {'id': 11, 'corners': _corners(CAM_TRUE, 11, OBLIQUE_MAP)}])
    assert both['yaw_reliable'] is True


# --------------------------------------------------------------------------
# pose_estimator: reject physically impossible fixes
# --------------------------------------------------------------------------

def test_fix_that_puts_the_camera_at_an_impossible_height_is_rejected():
    """The camera rides ~15 cm above the floor, always. Field data: every
    wrong planar-pose solution put it at 0.30-0.86 m, every right one at
    0.10-0.21 m. During a room tour such fixes threw the pose 3 m back and
    forth, one of them outside the room."""
    good = CAM_TRUE.copy()
    good[2, 3] = 0.15
    floating = CAM_TRUE.copy()
    floating[2, 3] = 0.45
    est = PoseEstimator(OBLIQUE_MAP, K, mount={'forward_m': 0.0,
                                               'lateral_m': 0.0,
                                               'height_m': 0.15})
    assert est.estimate([{'id': 11,
                          'corners': _corners(good, 11, OBLIQUE_MAP)}])
    assert est.estimate([{'id': 11,
                          'corners': _corners(floating, 11, OBLIQUE_MAP)}]) \
        is None


def test_impossible_tag_is_dropped_but_a_sane_one_still_gives_a_fix():
    good = CAM_TRUE.copy()
    good[2, 3] = 0.15
    floating = CAM_TRUE.copy()
    floating[2, 3] = 0.45
    est = PoseEstimator(OBLIQUE_MAP, K, mount={'forward_m': 0.0,
                                               'lateral_m': 0.0,
                                               'height_m': 0.15})
    pose = est.estimate([
        {'id': 10, 'corners': _corners(floating, 10, OBLIQUE_MAP)},
        {'id': 11, 'corners': _corners(good, 11, OBLIQUE_MAP)}])
    assert pose['n_tags'] == 1 and pose['rotation_tag'] == 11
    assert pose['n_rejected'] == 1
    assert abs(pose['T_map_cam'][2, 3] - 0.15) < 0.01


def test_height_check_is_off_unless_a_camera_height_is_given():
    floating = CAM_TRUE.copy()
    floating[2, 3] = 0.45
    assert estimate_camera_pose(
        [{'id': 11, 'corners': _corners(floating, 11, OBLIQUE_MAP)}],
        OBLIQUE_MAP, K) is not None


def test_map_renderer_draws_grid_nodes_in_their_state_colours():
    from slam.map_renderer import _NODE, MapRenderer
    r = MapRenderer()
    grid = [{'x': 0.0, 'y': 0.0, 'state': 'safe'},
            {'x': 1.0, 'y': 0.0, 'state': 'unknown'},
            {'x': 2.0, 'y': 0.0, 'state': 'near'},
            {'x': 3.0, 'y': 0.0, 'state': 'danger'}]
    img = r.render([], [], grid=grid, route=[(0.0, 0.5), (3.0, 0.5)])
    present = {tuple(int(c) for c in px)
               for px in img.reshape(-1, 3)[::1]}
    for state in ('safe', 'unknown', 'near', 'danger'):
        assert _NODE[state] in present, state
    # red / yellow / blue / green, as specified (BGR)
    b, g, red = _NODE['danger']
    assert red > 200 and b < 80 and g < 80
    b, g, red = _NODE['near']
    assert red > 200 and g > 200 and b < 80
    b, g, red = _NODE['unknown']
    assert b > 200 and red < 100
    b, g, red = _NODE['safe']
    assert g > 180 and b < 100 and red < 100
