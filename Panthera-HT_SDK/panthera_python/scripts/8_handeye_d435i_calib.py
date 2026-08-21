#!/usr/bin/env python3
"""
Wrist-mounted D435i eye-in-hand hand-eye calibration.

Goal: board pose consistency RMS in base frame <= 5 mm.
  1) Multi-frame average + subpixel corners + IPPE
  2) Depth-based ArUco translation scale correction
  3) Stage1 small joint perturbations for rough extrinsics
     -> Stage2 densify near reachable joints + small approach steps
  4) Multi-method + high inlier-ratio RANSAC + nonlinear refinement

Usage:
  python3 8_handeye_d435i_calib.py --preview
  python3 8_handeye_d435i_calib.py
  python3 8_handeye_d435i_calib.py --reprocess handeye_output/20260818_135656
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Panthera_lib import Panthera  # noqa: E402


ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

HANDEYE_METHOD_MAP = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def T_to_Rt(T: np.ndarray):
    return T[:3, :3].copy(), T[:3, 3].copy()


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T_to_Rt(T)
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rotation_matrix_to_quaternion(R: np.ndarray) -> list:
    q = np.empty(4, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        nxt = [1, 2, 0]
        j, k = nxt[i], nxt[nxt[i]]
        s = 2.0 * np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12))
        q[i + 1] = 0.25 * s
        q[0] = (R[k, j] - R[j, k]) / s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return (q / np.linalg.norm(q)).tolist()


def clamp_joints(q, lower, upper):
    q = np.asarray(q, dtype=float).copy()
    return np.minimum(np.maximum(q, lower), upper)


def look_at_opencv(cam_pos, target, up_hint=np.array([0.0, 0.0, 1.0])):
    """OpenCV camera frame: Z forward, X right, Y down."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    n = np.linalg.norm(z)
    if n < 1e-9:
        return np.eye(3)
    z = z / n
    up = np.asarray(up_hint, float)
    if abs(np.dot(z, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def median_depth(depth_frame, u, v, win=21):
    if depth_frame is None:
        return None
    h, w = depth_frame.get_height(), depth_frame.get_width()
    u, v = int(round(u)), int(round(v))
    r = win // 2
    vals = []
    for y in range(v - r, v + r + 1):
        for x in range(u - r, u + r + 1):
            if 0 <= x < w and 0 <= y < h:
                d = depth_frame.get_distance(x, y)
                if 0.15 < d < 2.5:
                    vals.append(d)
    if len(vals) < 8:
        return None
    return float(np.median(vals))


class RealSenseAruco:
    def __init__(self, cfg: dict):
        cam = cfg["camera"]
        ar = cfg["aruco"]
        det_cfg = cfg.get("detection", {}) or {}
        self.marker_id = int(ar["marker_id"])
        self.marker_length = float(ar["marker_length_m"])
        self.avg_frames = int(det_cfg.get("avg_frames", 12))
        self.use_depth_scale = bool(det_cfg.get("use_depth_scale", True))
        self.min_side_px = float(det_cfg.get("min_side_px", 140))
        self.border_px = int(det_cfg.get("border_px", 40))
        self.max_reproj_px = float(det_cfg.get("max_reproj_px", 1.5))
        dict_name = ar["dictionary"]
        if dict_name not in ARUCO_DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[dict_name])
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.minMarkerPerimeterRate = 0.02
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

        ctx = rs.context()
        if len(list(ctx.query_devices())) == 0:
            raise RuntimeError(
                "No RealSense detected. Check USB (prefer USB3), replug, then retry."
            )
        w, h, fps = int(cam["width"]), int(cam["height"]), int(cam["fps"])
        self.pipeline, self.profile = self._start_camera(w, h, fps)
        self.align = rs.align(rs.stream.color)
        for _ in range(30):
            self.pipeline.wait_for_frames()

        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64
        )
        self.dist = np.array(intr.coeffs, dtype=np.float64)
        self.serial = self.profile.get_device().get_info(rs.camera_info.serial_number)
        self.last_reject = ""
        self.last_raw_ids = []
        s = self.marker_length / 2.0
        self.obj_pts = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64
        )

    @staticmethod
    def _start_camera(w, h, fps):
        attempts = [
            (w, h, w, h),
            (w, h, 1280, 720),
            (1280, 720, 1280, 720),
            (1280, 720, 640, 480),
        ]
        last_err = None
        for cw, ch, dw, dh in attempts:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
            try:
                profile = pipeline.start(config)
                print(f"[camera] color={cw}x{ch} depth={dw}x{dh}")
                return pipeline, profile
            except Exception as e:
                last_err = e
                try:
                    pipeline.stop()
                except Exception:
                    pass
        raise RuntimeError(f"Failed to open D435i: {last_err}")

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def grab_aligned(self):
        frames = self.pipeline.wait_for_frames(5000)
        frames = self.align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color:
            return None, None
        return np.asanyarray(color.get_data()), depth

    def _pose_from_corners(self, corners_2d, depth_frame=None):
        img_pts = np.asarray(corners_2d, dtype=np.float64).reshape(4, 1, 2)
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            self.obj_pts.reshape(-1, 1, 3),
            img_pts,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok or len(tvecs) == 0:
            ok2, rvec, tvec = cv2.solvePnP(
                self.obj_pts, img_pts.reshape(4, 2), self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok2:
                return None
            rvecs, tvecs = [rvec], [tvec]

        best = None
        for rvec, tvec in zip(rvecs, tvecs):
            proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist)
            err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).mean())
            if best is None or err < best[0]:
                best = (err, rvec, tvec)
        _, rvec, tvec = best
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

        depth_m = None
        if depth_frame is not None and self.use_depth_scale:
            cx, cy = img_pts.reshape(-1, 2).mean(axis=0)
            depth_m = median_depth(depth_frame, cx, cy, win=21)
            if depth_m is not None and abs(float(tvec[2])) > 1e-3:
                tvec = tvec * (depth_m / float(tvec[2]))

        c = img_pts.reshape(4, 2)
        side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
        return {
            "corners": np.asarray(corners_2d, dtype=np.float32).reshape(1, 4, 2),
            "ids": np.array([[self.marker_id]]),
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
            "tvec": tvec,
            "T_cam_marker": rvec_tvec_to_T(rvec, tvec),
            "distance_m": float(np.linalg.norm(tvec)),
            "reproj_err_px": float(best[0]),
            "depth_m": depth_m,
            "side_px": side,
        }

    def detect(self, image_bgr, depth_frame=None):
        self.last_reject = ""
        self.last_raw_ids = []
        if image_bgr is None:
            return None
        corners, ids, _ = self.detector.detectMarkers(image_bgr)
        if ids is None:
            self.last_reject = "no_aruco"
            return None
        ids = ids.flatten()
        self.last_raw_ids = ids.astype(int).tolist()
        matches = np.where(ids == self.marker_id)[0]
        if len(matches) == 0:
            self.last_reject = f"ids={self.last_raw_ids} (want {self.marker_id})"
            return None
        c = corners[int(matches[0])].reshape(4, 2).astype(np.float32)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        c = cv2.cornerSubPix(gray, c.reshape(-1, 1, 2), (5, 5), (-1, -1), term).reshape(4, 2)
        det = self._pose_from_corners(c, depth_frame)
        if det is None:
            self.last_reject = "pnp_fail"
            return None
        ok, reason = self._quality_ok(det, image_bgr.shape)
        if not ok:
            self.last_reject = reason
            det["reject_reason"] = reason
            return None
        det["reject_reason"] = ""
        return det

    def _quality_ok(self, det, shape):
        h, w = shape[:2]
        c = det["corners"].reshape(4, 2)
        if np.any(c[:, 0] < self.border_px) or np.any(c[:, 0] > w - 1 - self.border_px):
            return False, "near_border"
        if np.any(c[:, 1] < self.border_px) or np.any(c[:, 1] > h - 1 - self.border_px):
            return False, "near_border"
        if det.get("side_px", 0) < self.min_side_px:
            return False, f"small {det.get('side_px', 0):.0f}px"
        if det.get("reproj_err_px", 99) > self.max_reproj_px:
            return False, f"reproj {det.get('reproj_err_px'):.2f}px"
        return True, ""

    def detect_averaged(self):
        corners_list = []
        depth_vals = []
        last_img = None
        need = max(4, self.avg_frames // 3)
        for _ in range(self.avg_frames):
            img, depth = self.grab_aligned()
            if img is None:
                continue
            last_img = img
            det = self.detect(img, depth)
            if det is None:
                continue
            corners_list.append(det["corners"].reshape(4, 2))
            if det.get("depth_m"):
                depth_vals.append(det["depth_m"])
            time.sleep(0.015)
        if len(corners_list) < need:
            return None, last_img
        c_mean = np.mean(np.stack(corners_list, axis=0), axis=0)
        _, depth = self.grab_aligned()
        det = self._pose_from_corners(c_mean, depth)
        if det is None:
            return None, last_img
        if depth_vals:
            det["depth_m"] = float(np.median(depth_vals))
        ok, reason = self._quality_ok(det, last_img.shape) if last_img is not None else (True, "")
        if not ok:
            return None, last_img
        return det, last_img

    def annotate(self, image_bgr, det):
        vis = image_bgr.copy() if image_bgr is not None else np.zeros((480, 640, 3), np.uint8)
        if det is None:
            cv2.putText(
                vis, f"NO id={self.marker_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
            return vis
        cv2.aruco.drawDetectedMarkers(vis, [det["corners"]], det["ids"])
        cv2.drawFrameAxes(vis, self.K, self.dist, det["rvec"], det["tvec"], self.marker_length * 0.5)
        msg = f"id={self.marker_id} dist={det['distance_m']:.3f}m err={det.get('reproj_err_px', 0):.2f}px"
        if det.get("depth_m"):
            msg += f" depth={det['depth_m']:.3f}m"
        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        return vis


def use_link6_frame(robot: Panthera, link_name: str = "link6"):
    if robot.model is None:
        raise RuntimeError("URDF not loaded; cannot switch end-effector frame")
    if not robot.model.existFrame(link_name):
        raise RuntimeError(f"Frame not found in URDF: {link_name}")
    robot.end_effector_frame_id = robot.model.getFrameId(link_name)
    print(f"[FK] End-effector frame switched to {link_name} (ID={robot.end_effector_frame_id})")


def refresh_state(robot: Panthera):
    robot.send_get_motor_state_cmd()
    robot.motor_send_cmd()
    time.sleep(0.05)


def wait_settle(robot: Panthera, settle_s: float, vel_eps: float = 0.08):
    time.sleep(settle_s)
    t0 = time.time()
    while time.time() - t0 < 1.2:
        refresh_state(robot)
        vel = np.asarray(robot.get_current_vel(), dtype=float)
        if float(np.max(np.abs(vel))) < vel_eps:
            return
        time.sleep(0.05)


def get_T_base_gripper(robot: Panthera, n: int = 5) -> np.ndarray:
    Ts = []
    for _ in range(n):
        refresh_state(robot)
        fk = robot.forward_kinematics()
        if not fk:
            raise RuntimeError("Forward kinematics failed")
        Ts.append(np.asarray(fk["transform"], dtype=np.float64))
        time.sleep(0.02)
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    R = np.mean([T[:3, :3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def check_serial_free():
    ports = sorted(Path("/dev").glob("ttyACM*"))
    if not ports:
        print("[warn] No /dev/ttyACM* found")
        return False
    occupied = []
    for p in ports:
        try:
            r = subprocess.run(["fuser", str(p)], capture_output=True, text=True)
            pids = (r.stdout or "") + " " + (r.stderr or "")
            nums = [x for x in pids.replace(":", " ").split() if x.isdigit()]
            if nums:
                occupied.append((str(p), nums))
        except FileNotFoundError:
            break
    if not occupied:
        return True
    print("\n" + "=" * 60)
    print("ERROR: Serial port is busy (often Host app.py)")
    for port, pids in occupied:
        print(f"  {port} <- PID {', '.join(pids)}")
    print("Stop app.py first, then rerun")
    print("=" * 60 + "\n")
    return False


def solve_handeye_raw(Tg, Tc, method):
    Rg = [A[:3, :3] for A in Tg]
    tg = [A[:3, 3].reshape(3, 1) for A in Tg]
    Rt = [B[:3, :3] for B in Tc]
    tt = [B[:3, 3].reshape(3, 1) for B in Tc]
    R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=method)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def sample_errors(Tg, Tc, T_gc):
    P = np.vstack([(A @ T_gc @ B)[:3, 3] for A, B in zip(Tg, Tc)])
    mean = P.mean(axis=0)
    err = np.linalg.norm(P - mean, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    return err, rms, float(err.max()), mean


def refine_handeye(T0, Tg, Tc, iters: int = 40):
    rvec, _ = cv2.Rodrigues(T0[:3, :3])
    p = np.hstack([rvec.ravel(), T0[:3, 3]])

    def pack(vec):
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(vec[:3])[0]
        T[:3, 3] = vec[3:]
        return T

    def residual(vec):
        T = pack(vec)
        P = np.vstack([(A @ T @ B)[:3, 3] for A, B in zip(Tg, Tc)])
        return (P - P.mean(axis=0)).ravel()

    lam = 1e-3
    for _ in range(iters):
        r = residual(p)
        J = np.zeros((len(r), 6))
        eps = 1e-6
        for k in range(6):
            dp = p.copy()
            dp[k] += eps
            J[:, k] = (residual(dp) - r) / eps
        H = J.T @ J + lam * np.eye(6)
        try:
            delta = np.linalg.solve(H, -(J.T @ r))
        except np.linalg.LinAlgError:
            break
        p2 = p + delta
        if np.sum(residual(p2) ** 2) < np.sum(r ** 2):
            p = p2
            lam = max(lam * 0.4, 1e-8)
        else:
            lam *= 5.0
        if np.linalg.norm(delta) < 1e-10:
            break
    return pack(p)


def solve_handeye_robust(samples, cfg_cal):
    """Multi-method solve. Does not mutate input samples. RANSAC requires a high inlier ratio to avoid gaming low RMS."""
    Tg = np.stack([s["T_base_gripper"] for s in samples])
    Tc0 = np.stack([s["T_cam_marker"] for s in samples])
    methods = list(HANDEYE_METHOD_MAP.values())
    if cfg_cal.get("method") in HANDEYE_METHOD_MAP:
        methods = [HANDEYE_METHOD_MAP[cfg_cal["method"]]] + [
            m for m in methods if m != HANDEYE_METHOD_MAP[cfg_cal["method"]]
        ]

    scale_min = float(cfg_cal.get("scale_search_min", 1.0))
    scale_max = float(cfg_cal.get("scale_search_max", 1.0))
    scale_steps = int(cfg_cal.get("scale_search_steps", 1))
    inlier_mm = float(cfg_cal.get("ransac_inlier_mm", 10.0))
    target_mm = float(cfg_cal.get("target_rms_mm", 5.0))
    min_ratio = float(cfg_cal.get("min_inlier_ratio", 0.7))
    rng = np.random.default_rng(42)
    N = len(samples)
    min_keep = max(8, int(np.ceil(min_ratio * N)))
    best = None  # (n_inliers, -rms, scale, method, T, inliers, mode)

    for scale in np.linspace(scale_min, scale_max, max(scale_steps, 1)):
        Tc = Tc0.copy()
        Tc[:, :3, 3] *= scale
        for method in methods:
            try:
                T_all = refine_handeye(solve_handeye_raw(Tg, Tc, method), Tg, Tc)
            except Exception:
                continue
            _, rms_all, _, _ = sample_errors(Tg, Tc, T_all)
            cand = (N, -rms_all, scale, method, T_all, np.arange(N), "full")
            if best is None or cand[:2] > best[:2]:
                best = cand

            subset = min(8, N)
            for _ in range(int(cfg_cal.get("ransac_iters", 200))):
                idx = rng.choice(N, size=subset, replace=False)
                try:
                    T = solve_handeye_raw(Tg[idx], Tc[idx], method)
                except Exception:
                    continue
                e, _, _, _ = sample_errors(Tg, Tc, T)
                inliers = np.where(e * 1000.0 <= inlier_mm)[0]
                if len(inliers) < min_keep:
                    continue
                try:
                    T2 = refine_handeye(
                        solve_handeye_raw(Tg[inliers], Tc[inliers], method),
                        Tg[inliers],
                        Tc[inliers],
                    )
                except Exception:
                    continue
                _, rms2, _, _ = sample_errors(Tg[inliers], Tc[inliers], T2)
                cand = (len(inliers), -rms2, scale, method, T2, inliers, "ransac")
                if best is None or cand[:2] > best[:2]:
                    best = cand

    if best is None:
        raise RuntimeError("Hand-eye solve failed")

    n_in, neg_rms, scale, method, T, inliers, mode = best
    method_name = [k for k, v in HANDEYE_METHOD_MAP.items() if v == method][0]
    Tc = Tc0.copy()
    Tc[:, :3, 3] *= scale
    kept = []
    for i in inliers:
        s = dict(samples[i])
        s["T_cam_marker"] = Tc[i].copy()
        s["marker_scale_applied"] = float(scale)
        kept.append(s)
    err, rms_k, max_k, mean_p = sample_errors(
        np.stack([s["T_base_gripper"] for s in kept]),
        np.stack([s["T_cam_marker"] for s in kept]),
        T,
    )
    print(
        f"[solve] mode={mode} method={method_name} scale={scale:.4f} "
        f"inliers={len(inliers)}/{N}  RMS={rms_k*1000:.2f}mm MAX={max_k*1000:.2f}mm "
        f"(target≤{target_mm}mm)"
    )
    return T, kept, {
        "board_position_mean_m": mean_p.tolist(),
        "board_position_rms_m": rms_k,
        "board_position_max_m": max_k,
        "num_samples": len(kept),
        "num_raw_samples": N,
        "marker_scale": float(scale),
        "method": method_name,
        "solve_mode": mode,
        "per_sample_err_mm": (err * 1000).tolist(),
        "target_rms_mm": target_mm,
        "passed": bool(rms_k * 1000.0 <= target_mm),
    }


def estimate_board_in_base(samples, T_gripper_cam):
    Ps, Rs = [], []
    for s in samples:
        T_bm = s["T_base_gripper"] @ T_gripper_cam @ s["T_cam_marker"]
        Ps.append(T_bm[:3, 3])
        Rs.append(T_bm[:3, :3])
    p = np.mean(Ps, axis=0)
    R = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def camera_nudge_to_gripper(T_bg, T_gc, dx, dy, dz, rx, ry, rz):
    """Apply a small translation/rotation in the current camera frame, then convert back to flange pose."""
    T_bc = T_bg @ T_gc
    dT = np.eye(4)
    dT[:3, :3] = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    dT[:3, 3] = np.array([dx, dy, dz], dtype=float)
    return (T_bc @ dT) @ inv_T(T_gc)


def collect_at_q(robot, cam, q_tgt, duration, max_tqu, settle, out_dir, tag, samples):
    ok = robot.moveJ(q_tgt.tolist(), duration=duration, max_tqu=max_tqu, iswait=True, tolerance=0.03)
    if not ok:
        print(f"  [{tag}] moveJ failed")
        return False
    wait_settle(robot, settle)
    det, img = cam.detect_averaged()
    if img is not None:
        cv2.imwrite(str(out_dir / f"{tag}_raw.png"), img)
        cv2.imwrite(str(out_dir / f"{tag}.png"), cam.annotate(img, det))
    if det is None:
        print(f"  [{tag}] No valid detection for id={cam.marker_id}")
        return False
    T_bg = get_T_base_gripper(robot)
    samples.append(
        {
            "index": len(samples),
            "tag": tag,
            "q": robot.get_current_pos(),
            "T_base_gripper": T_bg,
            "T_cam_marker": det["T_cam_marker"],
            "marker_distance_m": det["distance_m"],
            "reproj_err_px": det.get("reproj_err_px"),
            "depth_m": det.get("depth_m"),
            "side_px": det.get("side_px"),
        }
    )
    span = ""
    if len(samples) >= 2:
        P = np.stack([s["T_base_gripper"][:3, 3] for s in samples])
        span = f"  span_mm={np.round((P.max(0) - P.min(0)) * 1000, 1)}"
    print(
        f"  [{tag}] OK n={len(samples)} dist={det['distance_m']:.3f}m "
        f"side={det.get('side_px', 0):.0f}px reproj={det.get('reproj_err_px', 0):.2f}px{span}"
    )
    return True


def ik_to_pose(robot, T_bg, q_seed, max_jump):
    q = robot.inverse_kinematics(
        target_position=T_bg[:3, 3].tolist(),
        target_rotation=T_bg[:3, :3],
        init_q=q_seed,
        multi_init=False,
        max_iter=400,
        eps=2e-3,
    )
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    if np.max(np.abs(q - q_seed)) > max_jump:
        return None
    return q


def run_preview(cfg: dict, save_dir: Path):
    cam = RealSenseAruco(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Preview: q=quit, s=save. Place the board centered at about 40-50 cm.")
    try:
        while True:
            img, depth = cam.grab_aligned()
            if img is None:
                continue
            det = cam.detect(img, depth)
            vis = cam.annotate(img, det)
            cv2.imshow("D435i handeye preview", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = save_dir / f"preview_{datetime.now().strftime('%H%M%S')}.png"
                cv2.imwrite(str(path), vis)
                print("Saved", path)
    finally:
        cam.stop()
        cv2.destroyAllWindows()


def save_result(out_dir, cfg, config_path, cam, robot_cfg, T_gc, metrics, samples, stamp):
    result = {
        "type": "eye_in_hand",
        "frame": {
            "robot_base": "base_link",
            "gripper": robot_cfg.get("end_effector_link", "link6"),
            "camera": "d435i_color",
            "description": "T_gripper_cam: X_gripper = T * X_cam",
        },
        "aruco": {
            **cfg["aruco"],
            "marker_length_effective_m": float(
                cfg["aruco"]["marker_length_m"] * metrics.get("marker_scale", 1.0)
            ),
        },
        "camera_serial": cam.serial if cam else None,
        "camera_intrinsics": {"K": cam.K.tolist(), "dist": cam.dist.tolist()} if cam else None,
        "method": metrics.get("method"),
        "T_gripper_cam": T_gc.tolist(),
        "quaternion_wxyz": rotation_matrix_to_quaternion(T_gc[:3, :3]),
        "translation_m": T_gc[:3, 3].tolist(),
        "validation": metrics,
        "num_samples": len(samples),
        "config_file": str(config_path),
        "timestamp": stamp,
    }
    yaml_path = out_dir / "handeye_result.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    with open(out_dir / "handeye_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.savez(
        out_dir / "samples.npz",
        T_base_gripper=np.stack([s["T_base_gripper"] for s in samples]),
        T_cam_marker=np.stack([s["T_cam_marker"] for s in samples]),
        q=np.stack([s["q"] for s in samples]),
    )
    return yaml_path


def run_reprocess(cfg, config_path, sample_dir: Path):
    data = np.load(sample_dir / "samples.npz")
    samples = []
    for i in range(len(data["q"])):
        samples.append(
            {
                "index": i,
                "q": data["q"][i],
                "T_base_gripper": data["T_base_gripper"][i],
                "T_cam_marker": data["T_cam_marker"][i],
            }
        )
    _, _, metrics = solve_handeye_robust(samples, cfg["calibration"])
    print(json.dumps({k: metrics[k] for k in metrics if k != "per_sample_err_mm"}, indent=2, ensure_ascii=False))
    print("per_sample_err_mm", np.round(metrics["per_sample_err_mm"], 2))


def run_calibration(cfg: dict, config_path: Path):
    out_root = SCRIPT_DIR / cfg["calibration"]["output_dir"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    robot_cfg = cfg["robot"]
    max_tqu = robot_cfg["max_torque"]
    duration = float(robot_cfg["move_duration_s"])
    settle = float(robot_cfg["settle_s"])
    min_samples = int(cfg["calibration"]["min_samples"])
    target_mm = float(cfg["calibration"].get("target_rms_mm", 5.0))
    orbit_cfg = cfg.get("orbit", {}) or {}
    max_jump = float(orbit_cfg.get("max_joint_jump_rad", 0.9))

    print("=" * 60)
    print("D435i wrist hand-eye calibration (target RMS <= {:.1f} mm)".format(target_mm))
    print("=" * 60)
    print("1) Stop Host app.py")
    print("2) Mount board flat, avoid glare; camera 40-50 cm away, full board centered")
    print("3) Stage1 small perturbations -> Stage2 densify near reachable joints (no absolute orbit IK)")
    if not check_serial_free():
        raise RuntimeError("Serial port is busy")
    input("Press Enter when ready ... ")
    if not check_serial_free():
        raise RuntimeError("Serial port is busy")

    cam = RealSenseAruco(cfg)
    robot = None
    samples = []
    try:
        robot = Panthera()
        if getattr(robot, "motor_count", 0) < 6:
            raise RuntimeError(f"Motor init failed motor_count={robot.motor_count}")
        use_link6_frame(robot, robot_cfg.get("end_effector_link", "link6"))
        refresh_state(robot)
        if robot_cfg.get("keep_gripper_closed", True):
            try:
                robot.gripper_close()
            except Exception as e:
                print(f"[gripper] {e}")

        det0, img0 = cam.detect_averaged()
        if det0 is None:
            if img0 is not None:
                cv2.imwrite(str(out_dir / "fail_start_no_marker.png"), cam.annotate(img0, None))
            raise RuntimeError("Start pose: no valid ArUco id=7; move to 40-50 cm with full board in view")
        q0 = np.array(robot.get_current_pos(), dtype=float)
        last_q = q0.copy()
        print(
            f"[start] dist={det0['distance_m']:.3f}m depth={det0.get('depth_m')} "
            f"side={det0.get('side_px', 0):.0f}px"
        )
        if det0["distance_m"] > 0.58:
            print("Warning: distance > 58 cm, single-marker noise is high. Stage2 will try to approach; still prefer 40-50 cm at start.")
        cv2.imwrite(str(out_dir / "start_view.png"), cam.annotate(img0, det0))

        lower = np.array(robot.joint_limits["lower"], dtype=float)
        upper = np.array(robot.joint_limits["upper"], dtype=float)

        print("\n===== Stage1 small-perturbation coarse capture =====")
        for i, dq in enumerate(cfg.get("joint_deltas") or [[0] * 6]):
            q_tgt = clamp_joints(q0 + np.asarray(dq, float), lower, upper)
            if collect_at_q(robot, cam, q_tgt, duration, max_tqu, settle, out_dir, f"s1_{i:02d}", samples):
                last_q = np.array(robot.get_current_pos(), dtype=float)
            else:
                robot.moveJ(last_q.tolist(), duration=duration, max_tqu=max_tqu, iswait=True)

        if len(samples) < 6:
            raise RuntimeError(f"Stage1 too few valid samples: {len(samples)}; make board larger/centered and retry")

        T_rough, kept1, metrics1 = solve_handeye_robust(samples, cfg["calibration"])
        print(f"[Stage1] rough RMS={metrics1['board_position_rms_m']*1000:.1f} mm")

        print("\n===== Stage2 densify reachable poses (joint deltas, no absolute orbit IK) =====")
        print("Return to start, then run larger joint deltas than Stage1 (hand-reachable amplitudes)")
        robot.moveJ(q0.tolist(), duration=duration, max_tqu=max_tqu, iswait=True)
        last_q = q0.copy()
        s2_deltas = cfg.get("stage2_joint_deltas") or []
        ok_s2 = 0
        max_s2 = int(orbit_cfg.get("max_success", 18))
        for i, dq in enumerate(s2_deltas):
            if ok_s2 >= max_s2:
                break
            q_tgt = clamp_joints(q0 + np.asarray(dq, float), lower, upper)
            if np.linalg.norm(q_tgt - last_q) < 0.03:
                continue
            if collect_at_q(robot, cam, q_tgt, duration, max_tqu, settle, out_dir, f"s2j_{i:02d}", samples):
                ok_s2 += 1
                last_q = np.array(robot.get_current_pos(), dtype=float)
            else:
                robot.moveJ(last_q.tolist(), duration=max(2.5, duration * 0.7), max_tqu=max_tqu, iswait=True)

        nudges = cfg.get("camera_nudges") or []
        if nudges and ok_s2 < max_s2:
            print(f"\nTry {len(nudges)} camera-frame micro steps (skip on failure, stop after 4 consecutive)")
            T_now = get_T_base_gripper(robot)
            q_seed = np.array(robot.get_current_pos(), dtype=float)
            ik_fail = 0
            for i, n in enumerate(nudges):
                if ok_s2 >= max_s2 or ik_fail >= 4:
                    if ik_fail >= 4:
                        print("Consecutive IK failures; stopping camera-frame micro steps")
                    break
                T_tgt = camera_nudge_to_gripper(
                    T_now, T_rough, n[0], n[1], n[2], n[3], n[4], n[5]
                )
                q_ik = ik_to_pose(robot, T_tgt, q_seed, max_jump)
                if q_ik is None:
                    ik_fail += 1
                    print(f"  [s2c_{i:02d}] skip (IK unreachable)")
                    continue
                q_ik = clamp_joints(q_ik, lower, upper)
                ik_fail = 0
                if collect_at_q(robot, cam, q_ik, duration, max_tqu, settle, out_dir, f"s2c_{i:02d}", samples):
                    ok_s2 += 1
                    last_q = q_ik
                    q_seed = q_ik
                    T_now = get_T_base_gripper(robot)
                else:
                    robot.moveJ(last_q.tolist(), duration=max(2.5, duration * 0.7), max_tqu=max_tqu, iswait=True)
                    q_seed = last_q
        print(f"[Stage2] newly succeeded {ok_s2} sets")

        print("\nReturning to start...")
        robot.moveJ(q0.tolist(), duration=duration, max_tqu=max_tqu, iswait=True)

        if len(samples) < min_samples:
            raise RuntimeError(f"Total valid samples {len(samples)} < {min_samples}")

        print(f"\n===== Final solve N={len(samples)} =====")
        T_gc, kept, metrics = solve_handeye_robust(samples, cfg["calibration"])
        P = np.stack([s["T_base_gripper"][:3, 3] for s in kept])
        metrics["gripper_span_m"] = (P.max(0) - P.min(0)).tolist()
        yaml_path = save_result(out_dir, cfg, config_path, cam, robot_cfg, T_gc, metrics, kept, stamp)

        print("\n" + "=" * 60)
        print("Calibration done")
        print(np.array2string(T_gc, precision=6, suppress_small=True))
        print(
            f"Consistency RMS={metrics['board_position_rms_m']*1000:.2f} mm  "
            f"MAX={metrics['board_position_max_m']*1000:.2f} mm  "
            f"span_xyz_mm={np.round(np.array(metrics['gripper_span_m'])*1000, 1)}"
        )
        print(f"Result: {yaml_path}")
        if metrics["passed"]:
            print(f"Met target <= {target_mm} mm")
        else:
            print(f"Did not meet <= {target_mm} mm. Place camera at 40-50 cm, board flat/no glare, recapture.")
            print("Measure the black outer square side with a ruler and set marker_length_m.")
        print("=" * 60)

    finally:
        cam.stop()
        if robot is not None:
            try:
                robot.set_stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Panthera-HT D435i high-precision hand-eye calibration")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "handeye_d435i_config.yaml"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--reprocess", type=str, default="", help="Re-solve from existing samples.npz only")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    if args.reprocess:
        run_reprocess(cfg, config_path, Path(args.reprocess))
    elif args.preview:
        run_preview(cfg, SCRIPT_DIR / cfg["calibration"]["output_dir"] / "preview")
    else:
        run_calibration(cfg, config_path)


if __name__ == "__main__":
    main()
