"""Pan-tilt consistency check: can the camera be aimed with the servos without
corrupting localization?

The pose estimator turns a tag sighting into a robot BASE pose using the
commanded pan/tilt angles (``transforms.camera_pose_in_base``). That is only
right if the servos really go where they are told, in the direction the model
assumes. Before navigation starts panning the camera to look for tags, measure
it.

Method — the robot must NOT move during the run:
  1. Sweep the pan (and optionally tilt) servo through a list of angles.
  2. At each angle solve the camera pose from whatever AprilTags are in view.
     That pose comes straight from the image, independent of the servo model.
  3. The camera's measured yaw versus the commanded pan gives the servo's real
     direction and gain. The base pose derived through the model should be
     the same at every angle, since the base never moved; its spread is the
     localization error panning would introduce.

A constant pan offset (camera not exactly centred at pan=0) cannot be seen by
this test — it is indistinguishable from the robot simply facing a slightly
different way — and it does not matter here: it is already baked into every
pose taken at pan=0.

Run (Pi stack up, robot IDLE, at least one tag in view, ideally several
around the robot so wide pan angles still see one):

    python -m slam.pantilt_check --pi 192.168.178.8
    python -m slam.pantilt_check --pi 192.168.178.8 --pan-deg=-75:75:15 --tilt-deg 0,10,20

(Write --pan-deg=... with the equals sign: a value starting with "-" is
otherwise taken for an option.)
"""
import argparse
import math
import os
import time

import numpy as np

from . import transforms

YAW_TOL_RAD = math.radians(3.0)
POS_TOL_M = 0.05


# --------------------------------------------------------------------------
# analysis (pure — unit tested)
# --------------------------------------------------------------------------

def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def camera_yaw_elev(T_map_cam):
    """Heading and elevation (rad) of the camera's optical axis in the map."""
    d = np.asarray(T_map_cam)[:3, 2]
    d = d / np.linalg.norm(d)
    return float(math.atan2(d[1], d[0])), float(math.asin(d[2]))


def fit_linear(xs, ys):
    """Least-squares ``y = gain * x + offset``. Returns ``(gain, offset,
    rms_residual)``, or None when there are not two distinct x values."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2 or np.ptp(xs) < 1e-9:
        return None
    gain, offset = np.polyfit(xs, ys, 1)
    resid = ys - (gain * xs + offset)
    return float(gain), float(offset), float(np.sqrt(np.mean(resid ** 2)))


def _base_poses(samples, mount, pan_of, tilt_of):
    out = []
    for s in samples:
        T_base_cam = transforms.camera_pose_in_base(
            pan_of(s['pan_cmd']), tilt_of(s['tilt_cmd']), mount)
        T_map_base = np.asarray(s['T_map_cam']) @ transforms.invert(T_base_cam)
        out.append(transforms.pose_xytheta(T_map_base))
    return out


def _spread(poses, ref, yaw_ok=None):
    """Max position / heading deviation from ``ref``. ``yaw_ok`` masks out
    poses whose heading is not trustworthy (tag seen face-on): they still
    count for position, where a face-on tag is perfectly good."""
    pos = max(math.hypot(p[0] - ref[0], p[1] - ref[1]) for p in poses)
    if yaw_ok is None:
        yaw_ok = [True] * len(poses)
    good = [p for p, ok in zip(poses, yaw_ok) if ok]
    if not good:
        return pos, 0.0
    # heading reference: the first trustworthy pose, not necessarily ``ref``
    yaw = max(abs(_wrap(p[2] - good[0][2])) for p in good)
    return pos, yaw


def _yaw_ok(samples):
    from .pose_estimator import YAW_RELIABLE_INCIDENCE_RAD
    return [s.get('incidence_rad', math.pi / 2) >= YAW_RELIABLE_INCIDENCE_RAD
            for s in samples]


def analyze(samples, mount, yaw_tol_rad=YAW_TOL_RAD, pos_tol_m=POS_TOL_M):
    """``samples``: one dict per servo setting that produced a tag fix —
    ``{'pan_cmd', 'tilt_cmd', 'T_map_cam', 'tag_ids'}`` (angles in rad).

    Returns a report dict; ``verdict`` is one of
      'ok'            the model matches the hardware — safe to pan
      'calibrate'     wrong, but consistently so: apply the measured gain /
                      inversion (and mount offset) and it becomes usable
      'tag_map'       the servo is fine — the pose only jumps when the fix
                      switches from one tag to another, i.e. the tag map
                      disagrees with itself (hurts ALL navigation, panned
                      or not)
      'unreliable'    the servo is not repeatable enough to localize through
      'inconclusive'  not enough fixes to tell
    """
    report = {'n_samples': len(samples), 'verdict': 'inconclusive',
              'n_yaw_reliable': 0,
              'pan_gain': None, 'pan_inverted': None, 'pan_fit_rms_rad': None,
              'tilt_gain': None, 'tilt_inverted': None,
              'base_pos_spread_m': None, 'base_yaw_spread_rad': None,
              'corrected_pos_spread_m': None,
              'corrected_yaw_spread_rad': None, 'base_poses': [],
              'within_tag_pos_spread_m': None,
              'within_tag_yaw_spread_rad': None, 'per_tag': {}}
    if not samples:
        return report

    ref_i = min(range(len(samples)),
                key=lambda i: (abs(samples[i]['pan_cmd'])
                               + abs(samples[i]['tilt_cmd'])))
    ref = samples[ref_i]
    ref_yaw, _ = camera_yaw_elev(ref['T_map_cam'])

    # pan response: measured camera heading vs commanded pan, at the ref tilt
    pan_set = [s for s in samples
               if abs(s['tilt_cmd'] - ref['tilt_cmd']) < 1e-6]
    # Fit WITHIN each tag (relative to that tag's first sample) and pool:
    # a jump in heading when the fix changes tags is the tag map's doing and
    # must not be mistaken for servo gain.
    by_tag = {}
    for s in pan_set:
        by_tag.setdefault(tuple(s['tag_ids']), []).append(s)
    px, py = [], []
    for group in by_tag.values():
        if len(group) < 2:
            continue
        g0 = min(group, key=lambda s: abs(s['pan_cmd']))
        y0 = camera_yaw_elev(g0['T_map_cam'])[0]
        for s in group:
            px.append(s['pan_cmd'] - g0['pan_cmd'])
            py.append(_wrap(camera_yaw_elev(s['T_map_cam'])[0] - y0))
    pan_fit = fit_linear(px, py)
    if pan_fit is None:                 # no tag seen at two angles: global
        pan_fit = fit_linear(
            [s['pan_cmd'] - ref['pan_cmd'] for s in pan_set],
            [_wrap(camera_yaw_elev(s['T_map_cam'])[0] - ref_yaw)
             for s in pan_set])
    # tilt response: the model's positive tilt pitches the camera DOWN, so
    # true_tilt = -(elevation change) per commanded tilt
    # Pooled over every pan angle that has fixes at two or more tilts, each
    # relative to that pan's lowest-tilt sample — the tag seen straight
    # ahead usually leaves the frame as soon as the camera tilts.
    by_pan = {}
    for s in samples:
        by_pan.setdefault(round(s['pan_cmd'], 6), []).append(s)
    tx, ty = [], []
    for group in by_pan.values():
        if len(group) < 2:
            continue
        g0 = min(group, key=lambda s: abs(s['tilt_cmd']))
        e0 = camera_yaw_elev(g0['T_map_cam'])[1]
        for s in group:
            tx.append(s['tilt_cmd'] - g0['tilt_cmd'])
            ty.append(-(camera_yaw_elev(s['T_map_cam'])[1] - e0))
    tilt_fit = fit_linear(tx, ty)
    if pan_fit is None and tilt_fit is None:
        return report

    pan_gain = pan_fit[0] if pan_fit else 1.0
    tilt_gain = tilt_fit[0] if tilt_fit else 1.0
    if pan_fit:
        report.update(pan_gain=pan_gain, pan_inverted=pan_gain < 0,
                      pan_fit_rms_rad=pan_fit[2])
    if tilt_fit:
        report.update(tilt_gain=tilt_gain, tilt_inverted=tilt_gain < 0)

    # base pose through the model as it stands today
    yaw_ok = _yaw_ok(samples)
    report['n_yaw_reliable'] = sum(yaw_ok)
    poses = _base_poses(samples, mount, lambda p: p, lambda t: t)
    pos, yaw = _spread(poses, poses[ref_i], yaw_ok)
    # ...and with the measured servo response substituted
    fixed = _base_poses(
        samples, mount,
        lambda p: ref['pan_cmd'] + pan_gain * (p - ref['pan_cmd']),
        lambda t: ref['tilt_cmd'] + tilt_gain * (t - ref['tilt_cmd']))
    fpos, fyaw = _spread(fixed, fixed[ref_i], yaw_ok)
    report.update(base_pos_spread_m=pos, base_yaw_spread_rad=yaw,
                  corrected_pos_spread_m=fpos, corrected_yaw_spread_rad=fyaw,
                  base_poses=poses)

    # Same analysis restricted to angles that used the SAME tag(s): if the
    # pose is steady within each group and only jumps between groups, the
    # servo is innocent and the tag map is not.
    groups = {}
    for s, p, ok in zip(samples, fixed, yaw_ok):
        groups.setdefault(tuple(s['tag_ids']), []).append((p, ok))
    wpos = wyaw = 0.0
    for ids, members in groups.items():
        ps = [m[0] for m in members]
        oks = [m[1] for m in members]
        mean = (float(np.mean([p[0] for p in ps])),
                float(np.mean([p[1] for p in ps])),
                math.atan2(float(np.mean([math.sin(p[2]) for p in ps])),
                           float(np.mean([math.cos(p[2]) for p in ps]))))
        gp, gy = _spread(ps, ps[0], oks)
        wpos, wyaw = max(wpos, gp), max(wyaw, gy)
        report['per_tag'][ids] = {'n': len(ps), 'n_yaw_reliable': sum(oks),
                                  'mean_pose': mean,
                                  'pos_spread_m': gp, 'yaw_spread_rad': gy}
    report.update(within_tag_pos_spread_m=wpos,
                  within_tag_yaw_spread_rad=wyaw)
    multi = any(g['n'] >= 2 for g in report['per_tag'].values())

    if yaw <= yaw_tol_rad and pos <= pos_tol_m:
        report['verdict'] = 'ok'
    elif fyaw <= yaw_tol_rad and fpos <= pos_tol_m:
        report['verdict'] = 'calibrate'
    elif (len(groups) >= 2 and multi
            and wyaw <= yaw_tol_rad and wpos <= pos_tol_m):
        report['verdict'] = 'tag_map'
    elif fyaw <= yaw_tol_rad:
        report['verdict'] = 'calibrate'
    else:
        report['verdict'] = 'unreliable'
    return report


# --------------------------------------------------------------------------
# data collection (talks to the Pi)
# --------------------------------------------------------------------------

def _parse_angles(spec: str):
    """'-60:60:15' (start:stop:step, inclusive) or '0,10,20' -> degrees."""
    if ':' in spec:
        start, stop, step = (float(v) for v in spec.split(':'))
        n = int(round((stop - start) / step))
        return [start + i * step for i in range(n + 1)]
    return [float(v) for v in spec.split(',') if v.strip() != '']


class PiDidNotAim(RuntimeError):
    pass


def _collect(args):
    import cv2
    import requests
    from . import camera_calib
    from .apriltag_detector import AprilTagDetector
    from .pose_estimator import estimate_camera_pose, load_tag_map

    K, dist, _ = camera_calib.load_intrinsics(args.intrinsics)
    tag_map = load_tag_map(args.tag_map)
    detector = AprilTagDetector()
    http = requests.Session()
    cmd_url = f'http://{args.pi}:9091/command'
    status_url = f'http://{args.pi}:9091/status'
    snap_url = f'http://{args.pi}:8080/snapshot'

    mode = http.get(status_url, timeout=3.0).json().get('mode')
    if mode != 'IDLE':
        raise SystemExit(f'robot is in mode {mode}, not IDLE — send `停` '
                         f'first; the robot must stand still for this check')

    def aim(pan, tilt):
        http.post(cmd_url, json={'action': 'pantilt', 'yaw': pan,
                                 'pitch': tilt}, timeout=3.0)

    samples, rows = [], []
    pans = [math.radians(d) for d in _parse_angles(args.pan_deg)]
    tilts = [math.radians(d) for d in _parse_angles(args.tilt_deg)]
    try:
        for tilt in tilts:
            for pan in pans:
                aim(pan, tilt)
                time.sleep(args.settle)
                best, fixes, ids, tries = None, 0, set(), 0
                deadline = time.time() + args.settle + 4.0
                while tries < args.frames and time.time() < deadline:
                    r = http.get(snap_url, timeout=3.0)
                    hy = float(r.headers.get('X-Pan-Yaw', 'nan'))
                    ht = float(r.headers.get('X-Pan-Tilt', 'nan'))
                    if not (abs(hy - pan) < 2e-3 and abs(ht - tilt) < 2e-3):
                        time.sleep(0.1)     # frame predates the command
                        continue
                    tries += 1
                    frame = cv2.imdecode(
                        np.frombuffer(r.content, dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    dets = detector.detect(frame)
                    info = {}
                    solved = estimate_camera_pose(dets, tag_map, K, dist,
                                                  details=info)
                    if solved is None:
                        continue
                    fixes += 1
                    ids.update(d['id'] for d in dets if d['id'] in tag_map)
                    if best is None or solved[1] < best[1]:
                        best = solved + (info['incidence_rad'],)
                if tries == 0:
                    raise PiDidNotAim(
                        'the Pi never reported the commanded pan-tilt angle '
                        'in its /snapshot headers. Its agent_node probably '
                        'predates the `pantilt` command — copy the updated '
                        'agent_node.py to the Pi, colcon build, relaunch.')
                rows.append((pan, tilt, fixes, tries, sorted(ids), best))
                if best is not None:
                    samples.append({'pan_cmd': pan, 'tilt_cmd': tilt,
                                    'T_map_cam': best[0],
                                    'incidence_rad': best[3],
                                    'tag_ids': sorted(ids)})
                print(f'  pan {math.degrees(pan):+6.1f}  '
                      f'tilt {math.degrees(tilt):+5.1f}  '
                      f'fixes {fixes}/{tries}  tags {sorted(ids)}')
    finally:
        try:
            aim(0.0, 0.0)                   # always leave the camera centred
        except Exception:                   # noqa: BLE001
            print('warning: could not re-centre the camera')
    return samples, rows


def _print_report(samples, rows, report):
    deg = math.degrees
    print('\n pan_cmd  tilt_cmd  fix   tags        cam_yaw   base_x  base_y'
          '  base_yaw  reproj  view')
    poses = iter(report['base_poses'])
    ref_yaw = None
    for pan, tilt, fixes, tries, ids, best in rows:
        head = f' {deg(pan):+6.1f}°  {deg(tilt):+6.1f}°  {fixes}/{tries}'
        if best is None:
            print(f'{head}   (no tag in view)')
            continue
        cam_yaw, _ = camera_yaw_elev(best[0])
        ref_yaw = cam_yaw if ref_yaw is None else ref_yaw
        x, y, yaw = next(poses, (float('nan'),) * 3)
        from .pose_estimator import YAW_RELIABLE_INCIDENCE_RAD
        view = (f'{deg(best[3]):4.0f}°'
                + ('' if best[3] >= YAW_RELIABLE_INCIDENCE_RAD
                   else '  face-on: heading not counted'))
        print(f'{head}   {str(ids):10s}  {deg(cam_yaw):+7.1f}°  '
              f'{x:+6.2f}  {y:+6.2f}  {deg(yaw):+7.1f}°  {best[1]:5.2f}px'
              f'  {view}')

    def fmt(v, unit='', scale=1.0, nd=2):
        return 'n/a' if v is None else f'{v * scale:.{nd}f}{unit}'

    print('\nservo response (1.00 = moves exactly as commanded):')
    print(f'  pan  gain {fmt(report["pan_gain"])}'
          f'{"  INVERTED" if report["pan_inverted"] else ""}'
          f'   fit rms {fmt(report["pan_fit_rms_rad"], "°", 180 / math.pi)}')
    print(f'  tilt gain {fmt(report["tilt_gain"])}'
          f'{"  INVERTED" if report["tilt_inverted"] else ""}')
    print(f'base pose spread across the sweep (the robot did not move, so '
          f'ideally 0).\nHeading is judged only on the '
          f'{report["n_yaw_reliable"]}/{report["n_samples"]} fixes whose tag '
          f'was seen obliquely:')
    print(f'  as modelled today : '
          f'{fmt(report["base_pos_spread_m"], " cm", 100, 1)}, '
          f'{fmt(report["base_yaw_spread_rad"], "°", 180 / math.pi, 1)}')
    print(f'  with measured gain: '
          f'{fmt(report["corrected_pos_spread_m"], " cm", 100, 1)}, '
          f'{fmt(report["corrected_yaw_spread_rad"], "°", 180 / math.pi, 1)}')
    print(f'  within one tag    : '
          f'{fmt(report["within_tag_pos_spread_m"], " cm", 100, 1)}, '
          f'{fmt(report["within_tag_yaw_spread_rad"], "°", 180 / math.pi, 1)}')
    print(f'tolerance: {POS_TOL_M * 100:.0f} cm, {deg(YAW_TOL_RAD):.0f}°')
    print('base pose as seen through each tag:')
    for ids, g in sorted(report['per_tag'].items()):
        x, y, yaw = g['mean_pose']
        print(f'  tags {str(list(ids)):10s} n={g["n"]}  '
              f'({x:+.2f}, {y:+.2f}) yaw {deg(yaw):+6.1f}°   '
              f'spread {g["pos_spread_m"] * 100:.1f} cm, '
              f'{deg(g["yaw_spread_rad"]):.1f}°')
    print()

    verdict = report['verdict']
    if verdict == 'ok':
        print('VERDICT: OK — the pan-tilt model matches the hardware. Panning '
              'the camera to find tags will not corrupt localization.')
    elif verdict == 'calibrate':
        print('VERDICT: CALIBRATE — the servo does not move as the model '
              'assumes, but consistently. Before panning during navigation:')
        if report['pan_inverted']:
            print('  - pan turns the opposite way: set pantilt_node '
                  'pan_invert: true in params.yaml')
        elif report['pan_gain'] is not None \
                and abs(report['pan_gain'] - 1.0) > 0.05:
            print(f'  - pan gain is {report["pan_gain"]:.2f}: scale pan '
                  f'commands (or the angle fed to the pose estimator)')
        if report['tilt_inverted']:
            print('  - tilt moves the opposite way to the model: set '
                  'pantilt_node tilt_invert: true (or flip the model sign)')
        if (report['corrected_pos_spread_m'] or 0.0) > POS_TOL_M:
            print('  - position still shifts with pan: the lens sits off the '
                  'pan axis, which the model does not represent yet — it '
                  'needs a lens-offset term (the shift is roughly that offset)')
    elif verdict == 'tag_map':
        print('VERDICT: TAG MAP — the servo behaves: the base pose is steady '
              'while the same tag stays in view. It jumps only when the fix '
              'switches to a different tag, so the measured tag poses in '
              'tag_map disagree with each other (see the per-tag poses '
              'above). This corrupts navigation whenever the robot changes '
              'tags, with or without panning — fix the tag map first.')
    elif verdict == 'unreliable':
        print('VERDICT: UNRELIABLE — even with the measured gain the base '
              'pose jumps between angles. Either the servo is not repeatable '
              'or the tag map is inconsistent (check whether the jumps '
              'coincide with a change in the `tags` column). Do not localize '
              'through the pan-tilt yet.')
    else:
        print('VERDICT: INCONCLUSIVE — fewer than two angles produced a tag '
              'fix. Put the robot where tags are visible over a wider arc.')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pi', required=True, help='Raspberry Pi IP address')
    ap.add_argument('--pan-deg', default='-60:60:15',
                    help="'start:stop:step' or a comma list, degrees")
    ap.add_argument('--tilt-deg', default='0',
                    help='same format; the kit servo only tilts one way '
                         'from centre, e.g. 0,10,20')
    ap.add_argument('--frames', type=int, default=5,
                    help='frames examined per angle')
    ap.add_argument('--settle', type=float, default=1.2,
                    help='seconds to let the servo settle before reading')
    ap.add_argument('--tag-map', default=os.path.join(here, 'tag_map.yaml'))
    ap.add_argument('--intrinsics',
                    default=os.path.join(here, 'camera_intrinsics.yaml'))
    ap.add_argument('--cam-height', type=float, default=0.15)
    ap.add_argument('--save', default=None,
                    help='write the raw samples (camera poses) to this .json')
    args = ap.parse_args()

    mount = {'forward_m': 0.0, 'lateral_m': 0.0, 'height_m': args.cam_height}
    print('sweeping the pan-tilt — keep the robot still...')
    try:
        samples, rows = _collect(args)
    except PiDidNotAim as e:
        raise SystemExit(f'error: {e}')
    if args.save:
        import json
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump([{'pan_cmd': s['pan_cmd'], 'tilt_cmd': s['tilt_cmd'],
                        'tag_ids': s['tag_ids'],
                        'incidence_rad': s.get('incidence_rad'),
                        'T_map_cam': np.asarray(s['T_map_cam']).tolist()}
                       for s in samples], f)
        print(f'raw samples saved to {args.save}')
    report = analyze(samples, mount)
    _print_report(samples, rows, report)


if __name__ == '__main__':
    main()
