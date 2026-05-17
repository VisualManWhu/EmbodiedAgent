"""Camera intrinsics: load/save plus a one-time chessboard calibration helper.

The OV5647 intrinsics are needed by ``pose_estimator`` (solvePnP) and
``projection`` (pixel rays). Run the calibration once and commit the result to
``camera_intrinsics.yaml``.

Calibrate (laptop, PowerShell), with ~15-20 photos of a printed chessboard
held at varied angles/distances, all captured at the Pi camera's run
resolution:

    python -m slam.camera_calib calibrate --images calib_shots --cols 9 --rows 6 --square-m 0.025

Or grab the shots straight from the running Pi camera first:

    python -m slam.camera_calib capture --url http://192.168.178.37:8080/snapshot --out calib_shots
"""
import argparse
import glob
import os

import cv2
import numpy as np
import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'camera_intrinsics.yaml')


def load_intrinsics(path: str = DEFAULT_PATH):
    """Return ``(K, dist, (width, height))``. K is 3x3, dist is length-5."""
    with open(path, encoding='utf-8') as f:
        d = yaml.safe_load(f)
    K = np.array(d['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist = np.array(d['distortion'], dtype=np.float64).reshape(-1)
    return K, dist, (int(d['image_width']), int(d['image_height']))


def save_intrinsics(path: str, K, dist, image_size, rms: float | None = None):
    K = np.asarray(K, dtype=float)
    d = {
        'image_width': int(image_size[0]),
        'image_height': int(image_size[1]),
        'camera_matrix': [[float(v) for v in row] for row in K],
        'distortion': [float(v) for v in np.asarray(dist).reshape(-1)],
        'calibrated': rms is not None,
    }
    if rms is not None:
        d['rms_reproj_error_px'] = float(rms)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(d, f, sort_keys=False)


def calibrate_from_images(image_paths, cols: int, rows: int, square_m: float):
    """Chessboard calibration. ``cols``/``rows`` = count of inner corners.

    Returns ``(K, dist, rms, image_size, n_used)``.
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    obj_points, img_points = [], []
    image_size = None
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not found:
            print(f'  no chessboard: {os.path.basename(path)}')
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)
        print(f'  ok: {os.path.basename(path)}')

    if len(obj_points) < 5:
        raise RuntimeError(
            f'need >= 5 good chessboard views, got {len(obj_points)}')
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)
    return K, dist, rms, image_size, len(obj_points)


def _cmd_capture(args):
    import requests
    os.makedirs(args.out, exist_ok=True)
    print(f'snapshot from {args.url} — press SPACE to save, q to quit')
    i = 0
    while True:
        try:
            r = requests.get(args.url, timeout=3.0)
        except requests.RequestException as e:
            print(f'fetch failed: {e}')
            continue
        frame = cv2.imdecode(np.frombuffer(r.content, np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            continue
        cv2.imshow('capture (SPACE=save, q=quit)', frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            p = os.path.join(args.out, f'calib_{i:03d}.png')
            cv2.imwrite(p, frame)
            print(f'saved {p}')
            i += 1
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()


def _cmd_calibrate(args):
    paths = sorted(glob.glob(os.path.join(args.images, '*')))
    K, dist, rms, size, n = calibrate_from_images(
        paths, args.cols, args.rows, args.square_m)
    save_intrinsics(args.out, K, dist, size, rms)
    print(f'\ncalibrated from {n} views, RMS reproj error {rms:.3f} px')
    print(f'wrote {args.out}')


def main():
    ap = argparse.ArgumentParser(description='camera intrinsics calibration')
    sub = ap.add_subparsers(dest='cmd', required=True)

    cap = sub.add_parser('capture', help='grab calibration shots from the Pi')
    cap.add_argument('--url', required=True)
    cap.add_argument('--out', default='calib_shots')
    cap.set_defaults(func=_cmd_capture)

    cal = sub.add_parser('calibrate', help='compute intrinsics from shots')
    cal.add_argument('--images', default='calib_shots')
    cal.add_argument('--cols', type=int, default=9, help='inner corners per row')
    cal.add_argument('--rows', type=int, default=6, help='inner corners per col')
    cal.add_argument('--square-m', type=float, default=0.025)
    cal.add_argument('--out', default=DEFAULT_PATH)
    cal.set_defaults(func=_cmd_calibrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
