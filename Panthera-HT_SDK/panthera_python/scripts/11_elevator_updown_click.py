#!/usr/bin/env python3
"""
电梯厅外 UP/DOWN 按键：LocateAnything 单帧识别 → 深度+手眼 → 平进点击。

依赖（已链到仓库）:
  third_party/LocationAnything/LocateAnything-3B
  third_party/LocationAnything/Eagle
  conda 环境 locateanything（4bit 推理子进程）

用法:
  # 先停 app.py
  source ~/venvs/panthera/bin/activate
  cd ~/桌面/Panthera-HT/Panthera-HT_SDK/panthera_python/scripts
  python3 11_elevator_updown_click.py

预览键:
  SPACE = LocateAnything 检测 up/down
  u / d = 点击上/下键
  ESC/q = 退出回起始
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # Panthera-HT
LA_ROOT = REPO_ROOT / "third_party" / "LocationAnything"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(LA_ROOT))

from Panthera_lib import Panthera  # noqa: E402
from elevator_detect import (  # noqa: E402
    box_center_px,
    detect_hall_up_down,
)
from locateanything_ipc import LocateAnythingClient  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "handeye_core", SCRIPT_DIR / "8_handeye_d435i_calib.py"
)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)

_spec10 = importlib.util.spec_from_file_location(
    "touch10", SCRIPT_DIR / "10_handeye_touch_test.py"
)
_t10 = importlib.util.module_from_spec(_spec10)
_spec10.loader.exec_module(_t10)

load_config = _core.load_config
RealSenseAruco = _core.RealSenseAruco
check_serial_free = _core.check_serial_free
use_link6_frame = _core.use_link6_frame
get_T_base_gripper = _core.get_T_base_gripper
median_depth = _core.median_depth

backproject_uvz = _t10.backproject_uvz
board_up_axis = _t10.board_up_axis
flat_approach_rotation = _t10.flat_approach_rotation
get_tcp_fk = _t10.get_tcp_fk
solve_ik_flat = _t10.solve_ik_flat
stream_move_mit = _t10.stream_move_mit
hold_joints_mit = _t10.hold_joints_mit
# 位置1：与标定板点击同一观察位（验证手眼/姿态杠杆）
FIXED_Q_START = np.array([-0.001, 0.0, 0.355, -0.199, 0.037, 0.0], dtype=float)
# 补偿 MIT 下垂 + 触点偏低（日志常见到位 Z 低 5–9mm）
DEFAULT_UP_M = 0.012
DEFAULT_OFF_Y_M = 0.0
# n 朝外；负 standoff = 往板内过冲按实
DEFAULT_STANDOFF_M = -0.004
DEFAULT_HOLD_S = 0.8
MAX_TQU = _t10.MAX_TQU
HOLD_KP = _t10.HOLD_KP
HOLD_KD = _t10.HOLD_KD

DEFAULT_CALIB = (
    SCRIPT_DIR / "handeye_output" / "manual_20260819_150758" / "handeye_result.yaml"
)
WIN = "Elevator UP/DOWN | SPACE=detect  u/d=click  ESC=quit"


def pixel_to_base(cam, depth_frame, u, v, T_base_cam, win=21):
    z = median_depth(depth_frame, u, v, win=win)
    if z is None:
        return None, None
    p_cam = backproject_uvz(u, v, z, cam.K)
    p_base = (T_base_cam @ np.array([*p_cam, 1.0]))[:3]
    return p_base, float(z)


def estimate_panel_normal(cam, depth_frame, u, v, T_base_cam, half=40):
    """用按键附近三点深度估板法向（朝向相机一侧）。"""
    pts = []
    for du, dv in ((0, 0), (half, 0), (0, half), (-half, 0), (0, -half)):
        p, _ = pixel_to_base(cam, depth_frame, u + du, v + dv, T_base_cam, win=11)
        if p is not None:
            pts.append(p)
    if len(pts) < 3:
        return None
    P = np.asarray(pts, float)
    c = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - c)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    # 法向朝向相机：相机在 T_base_cam 原点附近
    cam_o = T_base_cam[:3, 3]
    if np.dot(cam_o - c, n) < 0:
        n = -n
    return n


def draw_dets(img, dets, selected=None):
    out = img.copy()
    h, w = out.shape[:2]
    colors = {"up": (0, 220, 0), "down": (0, 140, 255)}
    for name, box in dets.items():
        x1, y1, x2, y2 = [int(round(v / 1000.0 * s)) for v, s in zip(box, (w, h, w, h))]
        color = colors.get(name, (255, 255, 0))
        thick = 4 if name == selected else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        tag = name.upper() + (" *" if name == selected else "")
        cv2.putText(
            out, tag, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
        cx, cy = box_center_px(box, w, h)
        cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 22, 2)
    return out


def click_button(
    robot,
    cam,
    T_gc,
    box,
    *,
    up_m,
    standoff,
    duration,
    hold_s,
    off_y=0.0,
    freeze=None,
):
    """freeze: SPACE 时冻结的 {img, depth, T_bg}，与检测框同一帧。"""
    if freeze is not None:
        img = freeze["img"]
        depth = freeze["depth"]
        T_bg = freeze["T_bg"]
    else:
        img, depth = cam.grab_aligned()
        if img is None or depth is None:
            print("取流失败")
            return False
        T_bg = get_T_base_gripper(robot, n=3)
    h, w = img.shape[:2]
    u, v = box_center_px(box, w, h)
    T_bc = T_bg @ T_gc
    p_btn, z = pixel_to_base(cam, depth, u, v, T_bc)
    if p_btn is None:
        print(f"按键中心 ({u},{v}) 深度无效")
        return False
    n = estimate_panel_normal(cam, depth, u, v, T_bc)
    if n is None:
        # 默认墙面朝 -X
        n = np.array([-1.0, 0.0, 0.0])
    p_tcp, _ = get_tcp_fk(robot)
    if np.dot(p_tcp - p_btn, n) < 0:
        n = -n
    up = board_up_axis(n)
    R_flat = flat_approach_rotation(n)
    # 板上横向：与 up、n 正交（基座 Y 近似，侧偏微调用）
    lateral = np.cross(up, n)
    ln = float(np.linalg.norm(lateral))
    if ln < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0])
    else:
        lateral = lateral / ln
    p_aim = (
        p_btn
        + up * float(up_m)
        + lateral * float(off_y)
    )
    target = p_aim + n * float(standoff)
    d_aim = (p_aim - p_btn) * 1000.0
    print(
        f"  像素=({u},{v}) depthZ={z:.3f}m  "
        f"up_m={up_m*1000:.0f}mm off_y={off_y*1000:.0f}mm standoff={standoff*1000:.1f}mm"
        + ("  [freeze]" if freeze is not None else "")
    )
    print(
        f"  板心={np.round(p_btn, 4).tolist()}  "
        f"瞄准相对板心={np.round(d_aim, 1).tolist()} mm  "
        f"目标={np.round(target, 4).tolist()}"
    )
    solved = solve_ik_flat(robot, target, R_flat)
    if solved is None:
        print("  IK 失败")
        return False
    q_ik, p_pred, _, pos_err, ang_err = solved
    print(f"  IK 残差 {pos_err*1000:.1f}mm / {ang_err:.1f}°")
    stream_move_mit(robot, q_ik, duration)
    p_now, _ = get_tcp_fk(robot)
    err = (p_now - target) * 1000.0
    print(
        f"  到位 TCP={np.round(p_now, 4).tolist()}  "
        f"到位-目标={np.round(err, 1).tolist()} mm  |Δ|={np.linalg.norm(err):.1f}mm"
    )
    hold_joints_mit(robot, q_ik, seconds=hold_s)
    return True


def _freeze_depth(depth_frame):
    """把 rs2 depth_frame 拷成可复用对象（接口同 get_distance/get_*）。"""
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    raw = np.asanyarray(depth_frame.get_data())
    try:
        scale = float(depth_frame.get_units())
    except Exception:
        scale = 0.001
    z_m = raw.astype(np.float32) * scale

    class _FrozenDepth:
        def get_height(self):
            return h

        def get_width(self):
            return w

        def get_distance(self, x, y):
            if x < 0 or y < 0 or x >= w or y >= h:
                return 0.0
            return float(z_m[y, x])

    return _FrozenDepth()


def _freeze_frame(cam, robot, img, depth):
    return {
        "img": np.asarray(img).copy(),
        "depth": _freeze_depth(depth),
        "T_bg": get_T_base_gripper(robot, n=3),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", default=str(DEFAULT_CALIB))
    parser.add_argument("--config", default=str(SCRIPT_DIR / "handeye_d435i_config.yaml"))
    parser.add_argument("--button", choices=["up", "down", ""], default="", help="启动后自动点该键（仍先 SPACE 检测）")
    parser.add_argument(
        "--up_m",
        type=float,
        default=DEFAULT_UP_M,
        help="沿板面上抬补偿(m)，默认 0.012（补 MIT 下垂）",
    )
    parser.add_argument(
        "--off_y",
        type=float,
        default=DEFAULT_OFF_Y_M,
        help="沿板面横向偏置(m)，面板侧偏时可微调，默认 0",
    )
    parser.add_argument(
        "--standoff",
        type=float,
        default=DEFAULT_STANDOFF_M,
        help="相对板面沿外法向偏移(m)；负值=往里压，默认 -0.004",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--hold_s",
        type=float,
        default=DEFAULT_HOLD_S,
        help="点到位后停留秒数，默认 0.8",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (LA_ROOT / "LocateAnything-3B").exists():
        raise RuntimeError(f"缺少模型: {LA_ROOT / 'LocateAnything-3B'}")
    if not args.dry_run and not check_serial_free():
        raise RuntimeError("串口被占用，请先停 app.py")

    with open(args.calib, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)
    T_gc = np.array(calib["T_gripper_cam"], dtype=float)
    cfg = load_config(Path(args.config))

    print("加载 LocateAnything IPC（conda locateanything，约需几十秒）...")
    la = LocateAnythingClient()
    print("LocateAnything ready，warmup...")
    la.warmup()
    print("warmup done")

    cam = RealSenseAruco(cfg)
    robot = None
    q_home = FIXED_Q_START.copy()
    dets: dict[str, list[int]] = {}
    freeze = None
    try:
        robot = Panthera()
        use_link6_frame(robot, "link6")
        try:
            robot.send_get_motor_state_cmd()
            robot.motor_send_cmd()
            time.sleep(0.1)
        except Exception:
            pass

        print(f"先到固定起始 {q_home.tolist()}")
        if not args.dry_run:
            stream_move_mit(robot, q_home, 4.0)
            hold_joints_mit(robot, q_home, seconds=0.2)

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 960, 540)
        print("预览：SPACE 检测 | u/d 点击 | ESC 退出")
        print(f"瞄准 up_m={args.up_m*1000:.0f}mm  off_y={args.off_y*1000:.0f}mm  standoff={args.standoff*1000:.1f}mm")

        while True:
            # 预览期间 MIT 保持起始位
            tqe = np.asarray(robot.get_Gravity(q_home.tolist()), float)
            tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
            robot.pos_vel_tqe_kp_kd(
                q_home.tolist(),
                [0.0] * robot.motor_count,
                tqe.tolist(),
                HOLD_KP,
                HOLD_KD,
            )

            img, depth = cam.grab_aligned()
            if img is None:
                if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
                    break
                continue
            vis = draw_dets(img, dets)
            hud = f"dets={list(dets.keys()) or '-'}  SPACE=detect  u/d=click"
            cv2.putText(vis, hud, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key == 32:  # SPACE
                if depth is None:
                    print("无深度，跳过检测")
                    continue
                print("LocateAnything 检测 up/down ...")
                t0 = time.time()
                dets = detect_hall_up_down(la, img, center_crop=True, upscale=1.5, retry=False)
                freeze = _freeze_frame(cam, robot, img, depth)
                print(f"结果 {dets}  耗时 {time.time()-t0:.2f}s  (已冻结该帧 RGBD)")
                if args.button and args.button in dets and not args.dry_run:
                    print(f"自动点击 --button {args.button}")
                    ok = click_button(
                        robot,
                        cam,
                        T_gc,
                        dets[args.button],
                        up_m=args.up_m,
                        standoff=args.standoff,
                        duration=args.duration,
                        hold_s=args.hold_s,
                        off_y=args.off_y,
                        freeze=freeze,
                    )
                    print("回起始..." if ok else "点击失败，回起始")
                    stream_move_mit(robot, q_home, 4.0)
            if key in (ord("u"), ord("U")):
                if "up" not in dets or freeze is None:
                    print("还没有 UP 检测结果，先按 SPACE")
                    continue
                if args.dry_run:
                    print("dry-run，不点")
                    continue
                print("点击 UP ...")
                click_button(
                    robot,
                    cam,
                    T_gc,
                    dets["up"],
                    up_m=args.up_m,
                    standoff=args.standoff,
                    duration=args.duration,
                    hold_s=args.hold_s,
                    off_y=args.off_y,
                    freeze=freeze,
                )
                stream_move_mit(robot, q_home, 4.0)
            if key in (ord("d"), ord("D")):
                if "down" not in dets or freeze is None:
                    print("还没有 DOWN 检测结果，先按 SPACE")
                    continue
                if args.dry_run:
                    print("dry-run，不点")
                    continue
                print("点击 DOWN ...")
                click_button(
                    robot,
                    cam,
                    T_gc,
                    dets["down"],
                    up_m=args.up_m,
                    standoff=args.standoff,
                    duration=args.duration,
                    hold_s=args.hold_s,
                    off_y=args.off_y,
                    freeze=freeze,
                )
                stream_move_mit(robot, q_home, 4.0)

        print("回起始并保持")
        if robot is not None and not args.dry_run:
            stream_move_mit(robot, q_home, 4.0)
            hold_joints_mit(robot, q_home)
    except KeyboardInterrupt:
        print("\n中断")
        if robot is not None:
            try:
                stream_move_mit(robot, q_home, 4.0)
                hold_joints_mit(robot, q_home)
            except KeyboardInterrupt:
                pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        cam.stop()
        try:
            la.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
