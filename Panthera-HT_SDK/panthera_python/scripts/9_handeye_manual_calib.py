#!/usr/bin/env python3
"""
Manual hand-eye calibration + gravity-comp teaching (free-drag).

After power-on motors are position-locked and cannot be dragged by hand.
This program continuously sends:
  kp=0 (no position lock) + gravity/friction compensation + small kd (damping)
so the arm can be moved by hand.

Keys:
  SPACE      Capture 1 sample after settling
  BACKSPACE  Undo last sample
  ENTER      Solve early if >= 8 samples (auto-solve at 15 by default)
  Q / ESC    Quit (motors brake; hold the arm first)

Usage (stop Host app.py first):
  source ~/venvs/panthera/bin/activate
  cd ~/Desktop/Panthera-HT/Panthera-HT_SDK/panthera_python/scripts
  python3 9_handeye_manual_calib.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Panthera_lib import Panthera  # noqa: E402

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "handeye_core", SCRIPT_DIR / "8_handeye_d435i_calib.py"
)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)

load_config = _core.load_config
RealSenseAruco = _core.RealSenseAruco
check_serial_free = _core.check_serial_free
use_link6_frame = _core.use_link6_frame
get_T_base_gripper = _core.get_T_base_gripper
solve_handeye_robust = _core.solve_handeye_robust
save_result = _core.save_result

# Same as 2_gravity_friction_compensation_control.py
FC = np.array([0.20, 0.15, 0.15, 0.15, 0.04, 0.04])
FV = np.array([0.06, 0.06, 0.06, 0.03, 0.02, 0.02])
VEL_EPS = 0.02
TAU_LIMIT = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
# Small damping for a sticky feel while dragging; prevents free-fall
KD_DRAG = np.array([0.6, 0.8, 0.8, 0.4, 0.25, 0.25])


class DragTeach:
    """Gravity-comp teaching: kp=0, free-drag by hand."""

    def __init__(self, robot: Panthera):
        n = robot.motor_count
        self.robot = robot
        self.zero_pos = [0.0] * n
        self.zero_vel = [0.0] * n
        self.zero_kp = [0.0] * n
        self.kd = KD_DRAG[:n].tolist()

    def tick(self):
        vel = np.asarray(self.robot.get_current_vel(), dtype=float)
        tau_g = np.asarray(self.robot.get_Gravity(), dtype=float)
        tau_f = np.asarray(
            self.robot.get_friction_compensation(vel, FC, FV, VEL_EPS), dtype=float
        )
        tau = np.clip(tau_g + tau_f, -TAU_LIMIT, TAU_LIMIT)
        self.robot.pos_vel_tqe_kp_kd(
            self.zero_pos, self.zero_vel, tau.tolist(), self.zero_kp, self.kd
        )
        try:
            self.robot.gripper_control_MIT(0, 0, 0, 0, 0)
        except Exception:
            pass


def poll_frame(cam: RealSenseAruco):
    frames = cam.pipeline.poll_for_frames()
    if not frames:
        return None, None
    frames = cam.align.process(frames)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color:
        return None, None
    return np.asanyarray(color.get_data()), depth


def soft_detect(cam: RealSenseAruco, image_bgr, depth_frame=None):
    if image_bgr is None:
        return None
    corners, ids, _ = cam.detector.detectMarkers(image_bgr)
    if ids is None:
        return None
    ids = ids.flatten()
    matches = np.where(ids == cam.marker_id)[0]
    if len(matches) == 0:
        return None
    c = corners[int(matches[0])].reshape(4, 2).astype(np.float32)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    c = cv2.cornerSubPix(gray, c.reshape(-1, 1, 2), (5, 5), (-1, -1), term).reshape(4, 2)
    det = cam._pose_from_corners(c, depth_frame)
    if det is None:
        return None
    ok, reason = cam._quality_ok(det, image_bgr.shape)
    det["quality_ok"] = ok
    det["reject_reason"] = "" if ok else reason
    return det


def draw_hud(vis, n, need, det, status: str):
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

    color = (0, 255, 0) if det is not None and det.get("quality_ok", False) else (0, 165, 255)
    if det is None:
        color = (0, 0, 255)
        tip = "Board not detected"
    elif not det.get("quality_ok", False):
        tip = f"Detected but rejected: {det.get('reject_reason', '')}"
    else:
        tip = "OK; press SPACE to capture"

    lines = [
        f"Teach-drag  samples {n}/{need}  |  SPACE=capture  BACKSPACE=undo  ENTER=solve  Q=quit",
        tip,
        status,
    ]
    y = 28
    for i, t in enumerate(lines):
        cv2.putText(
            vis, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.62 if i else 0.68, color if i else (255, 255, 255), 2,
        )
        y += 28
    return vis


def capture_one(cam, robot, drag: DragTeach, avg_frames: int = 10):
    """Keep sending gravity compensation during capture to prevent arm droop."""
    t_end = time.time() + 0.2
    while time.time() < t_end:
        drag.tick()
        time.sleep(0.002)

    corners_list = []
    depth_vals = []
    last_img = None
    tries = 0
    need = max(4, avg_frames // 3)
    while len(corners_list) < avg_frames and tries < avg_frames * 4:
        drag.tick()
        img, depth = poll_frame(cam)
        tries += 1
        if img is None:
            time.sleep(0.002)
            continue
        last_img = img
        det = cam.detect(img, depth)
        if det is None:
            continue
        corners_list.append(det["corners"].reshape(4, 2))
        if det.get("depth_m"):
            depth_vals.append(det["depth_m"])

    if len(corners_list) < need:
        return None, last_img, "Detection rejected (hold still, center board); not captured"

    c_mean = np.mean(np.stack(corners_list, axis=0), axis=0)
    drag.tick()
    _, depth = poll_frame(cam)
    det = cam._pose_from_corners(c_mean, depth)
    if det is None:
        return None, last_img, "Pose estimate failed; not captured"
    if depth_vals:
        det["depth_m"] = float(np.median(depth_vals))
    if last_img is not None:
        ok, reason = cam._quality_ok(det, last_img.shape)
        if not ok:
            return None, last_img, f"Rejected: {reason}"

    drag.tick()
    T_bg = get_T_base_gripper(robot, n=3)
    drag.tick()
    sample = {
        "index": 0,
        "tag": "manual",
        "q": robot.get_current_pos(),
        "T_base_gripper": T_bg,
        "T_cam_marker": det["T_cam_marker"],
        "marker_distance_m": det["distance_m"],
        "reproj_err_px": det.get("reproj_err_px"),
        "depth_m": det.get("depth_m"),
        "side_px": det.get("side_px"),
    }
    msg = (
        f"OK dist={det['distance_m']:.3f}m side={det.get('side_px', 0):.0f}px "
        f"reproj={det.get('reproj_err_px', 0):.2f}px"
    )
    return sample, last_img, msg


def run_manual(cfg: dict, config_path: Path, num_samples: int):
    robot_cfg = cfg["robot"]
    target_mm = float(cfg["calibration"].get("target_rms_mm", 5.0))
    out_root = SCRIPT_DIR / cfg["calibration"]["output_dir"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"manual_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Manual hand-eye calibration (gravity-comp teach, free-drag)")
    print("=" * 60)
    print("1) Must stop Host app.py")
    print("2) After start the arm feels light; drag slowly by hand; keep board fixed"
    print("3) Use diverse poses; press SPACE after settling")
    print("4) Hold the arm before quit (motors will brake)")
    print(f"Output: {out_dir}")
    if not check_serial_free():
        raise RuntimeError("Serial port busy; stop app.py first")

    cam = RealSenseAruco(cfg)
    robot = None
    samples = []
    status = "Teach enabled: drag the arm slowly"
    try:
        robot = Panthera()
        if getattr(robot, "motor_count", 0) < 6:
            raise RuntimeError(f"Motor init failed motor_count={robot.motor_count}")
        use_link6_frame(robot, robot_cfg.get("end_effector_link", "link6"))
        drag = DragTeach(robot)
        # Send a few compensation ticks so position lock releases
        for _ in range(30):
            drag.tick()
            time.sleep(0.002)

        win = "handeye_manual_drag"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        solving = False
        last_vis = None
        last_det = None
        while True:
            drag.tick()
            img, depth = poll_frame(cam)
            if img is not None:
                last_det = soft_detect(cam, img, depth)
                vis = cam.annotate(img, last_det)
                last_vis = draw_hud(vis, len(samples), num_samples, last_det, status)
                cv2.imshow(win, last_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), ord("Q"), 27):
                print("User quit; not solving. Hold the arm.")
                break

            if key in (8, 127):
                if samples:
                    samples.pop()
                    status = f"Undone; now {len(samples)}/{num_samples}"
                    print(status)
                continue

            if key in (ord("\r"), 13):
                if len(samples) >= 8:
                    solving = True
                    break
                status = f"Need at least 8 samples to solve; now {len(samples)}"
                continue

            if key == ord(" "):
                if last_det is None or not last_det.get("quality_ok", False):
                    status = "Current view rejected; cannot capture"
                    print(status)
                    continue
                print("Capturing; keep still...")
                status = "Capturing..."
                if last_vis is not None:
                    cv2.imshow(win, draw_hud(last_vis, len(samples), num_samples, last_det, status))
                sample, shot, msg = capture_one(cam, robot, drag)
                if sample is None:
                    status = msg
                    print(msg)
                    continue
                sample["index"] = len(samples)
                sample["tag"] = f"m_{len(samples):02d}"
                samples.append(sample)
                if shot is not None:
                    cv2.imwrite(str(out_dir / f"{sample['tag']}.png"), cam.annotate(shot, last_det))
                P = np.stack([s["T_base_gripper"][:3, 3] for s in samples])
                span = (P.max(0) - P.min(0)) * 1000
                status = f"{msg} | span_mm={np.round(span, 1)}"
                print(f"[{len(samples)}/{num_samples}] {status}")
                if len(samples) >= num_samples:
                    solving = True
                    break

        if not solving or len(samples) < 8:
            print(f"Not enough samples ({len(samples)}); not solving")
            return

        print(f"\nSolving N={len(samples)} ...")
        T_gc, kept, metrics = solve_handeye_robust(samples, cfg["calibration"])
        P = np.stack([s["T_base_gripper"][:3, 3] for s in kept])
        metrics["gripper_span_m"] = (P.max(0) - P.min(0)).tolist()
        yaml_path = save_result(
            out_dir, cfg, config_path, cam, robot_cfg, T_gc, metrics, kept, stamp
        )
        print("\n" + "=" * 60)
        print("Calibration done")
        print(np.array2string(T_gc, precision=6, suppress_small=True))
        print(
            f"Consistency RMS={metrics['board_position_rms_m']*1000:.2f} mm  "
            f"MAX={metrics['board_position_max_m']*1000:.2f} mm  "
            f"span_xyz_mm={np.round(np.array(metrics['gripper_span_m'])*1000, 1)}"
        )
        print(f"Result: {yaml_path}")
        if metrics.get("passed"):
            print(f"Met target <= {target_mm} mm")
        else:
            print(f"Did not meet <= {target_mm} mm: try more large-angle poses")
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
    parser = argparse.ArgumentParser(description="Free-drag hand-eye calibration")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "handeye_d435i_config.yaml"))
    parser.add_argument("--num", type=int, default=15)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    det = cfg.setdefault("detection", {})
    det.setdefault("min_side_px", 120)
    det.setdefault("border_px", 25)
    det.setdefault("max_reproj_px", 2.0)
    det.setdefault("avg_frames", 10)
    run_manual(cfg, config_path, max(8, int(args.num)))


if __name__ == "__main__":
    main()
