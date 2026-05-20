"""Generate printable calibration / AprilTag targets.

Writes PNGs to ``slam/print_targets/``:
  - ``chessboard_9x6.png``   — 9x6 inner-corner calibration board
  - ``tag36h11_id<N>.png``   — tag36h11 markers with a white quiet-zone border

Printing note: printers rescale, so do NOT trust a nominal size. Print each
target, then MEASURE the result with a ruler:
  - calibration: measure one chessboard square -> pass as --square-m
  - tags: measure the black tag square side -> put in tag_map.yaml `size_m`

Run:
    python -m slam.make_targets                 # tags 0-7 + chessboard
    python -m slam.make_targets --tags 0-15     # more tags
"""
import argparse
import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), 'print_targets')


def make_chessboard(cols: int = 9, rows: int = 6, square_px: int = 160,
                    quiet_px: int = 120) -> np.ndarray:
    """Chessboard with ``cols`` x ``rows`` INNER corners (=> cols+1 x rows+1
    squares), on a white margin so the outer corners are detectable."""
    nx, ny = cols + 1, rows + 1
    board = np.zeros((ny * square_px, nx * square_px), np.uint8)
    for r in range(ny):
        for c in range(nx):
            if (r + c) % 2 == 0:
                y, x = r * square_px, c * square_px
                board[y:y + square_px, x:x + square_px] = 255
    canvas = np.full((board.shape[0] + 2 * quiet_px,
                      board.shape[1] + 2 * quiet_px), 255, np.uint8)
    canvas[quiet_px:quiet_px + board.shape[0],
           quiet_px:quiet_px + board.shape[1]] = board
    return canvas


def make_tag(tag_id: int, marker_px: int = 720) -> np.ndarray:
    """tag36h11 marker padded with a white quiet zone (>= 1 tag cell)."""
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(adict, tag_id, marker_px)
    quiet = marker_px // 5
    canvas = np.full((marker_px + 2 * quiet, marker_px + 2 * quiet),
                     255, np.uint8)
    canvas[quiet:quiet + marker_px, quiet:quiet + marker_px] = marker
    label = f'tag36h11  id={tag_id}   measure the BLACK square side'
    cv2.putText(canvas, label, (quiet, quiet - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    return canvas


def _parse_ids(spec: str) -> list:
    if '-' in spec:
        lo, hi = spec.split('-')
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(',')]


def main():
    ap = argparse.ArgumentParser(description='generate printable targets')
    ap.add_argument('--tags', default='0-7', help='tag id range, e.g. 0-7')
    ap.add_argument('--cols', type=int, default=9, help='chessboard inner cols')
    ap.add_argument('--rows', type=int, default=6, help='chessboard inner rows')
    ap.add_argument('--out', default=OUT_DIR)
    args = ap.parse_args()

    if not hasattr(cv2, 'aruco'):
        raise RuntimeError('cv2.aruco unavailable — need opencv-contrib-python')
    os.makedirs(args.out, exist_ok=True)

    board = make_chessboard(args.cols, args.rows)
    bpath = os.path.join(args.out, f'chessboard_{args.cols}x{args.rows}.png')
    cv2.imwrite(bpath, board)
    print(f'wrote {bpath}  ({args.cols}x{args.rows} inner corners)')

    for tag_id in _parse_ids(args.tags):
        tag = make_tag(tag_id)
        tpath = os.path.join(args.out, f'tag36h11_id{tag_id}.png')
        cv2.imwrite(tpath, tag)
        print(f'wrote {tpath}')

    print('\nprint each file, then MEASURE the result:')
    print('  chessboard square -> --square-m for camera_calib')
    print('  tag black square  -> size_m in tag_map.yaml')


if __name__ == '__main__':
    main()
