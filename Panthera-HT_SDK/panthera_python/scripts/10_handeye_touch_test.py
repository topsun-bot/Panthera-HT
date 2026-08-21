#!/usr/bin/env python3
"""
手眼验证：先到固定起始关节角 → 深度检板 → 平进点击 → 回起始。

固定起始（rad）:
  [-0.001, 0.0, 0.355, -0.199, 0.037, 0.0]

  python3 10_handeye_touch_test.py
  python3 10_handeye_touch_test.py --standoff 0.005
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Panthera_lib import Panthera  # noqa: E402

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
median_depth = _core.median_depth

DEFAULT_CALIB = (
    SCRIPT_DIR / "handeye_output" / "manual_20260819_150758" / "handeye_result.yaml"
)
MAX_TQU = [21.0, 36.0, 36.0, 21.0, 10.0, 10.0]
IK_POS_TOL_M = 0.008
# MIT 刚度：肩肘腕加大，减小平伸时高度塌陷
HOLD_KP = [45.0, 80.0, 90.0, 40.0, 30.0, 22.0]
HOLD_KD = [4.0, 6.0, 7.0, 3.5, 2.5, 1.8]
# 桌面臂+墙板手眼竖直偏差：上次「低4cm」含约1.4cm塌腕，净偏差约2.5cm
DEFAULT_UP_M = 0.025
DEFAULT_STANDOFF_M = 0.005
Z_CORRECT_TOL_M = 0.008
# 每次启动先到此关节角，再检测/点击，结束后也回到此位
FIXED_Q_START = np.array([-0.001, 0.0, 0.355, -0.199, 0.037, 0.0], dtype=float)


def backproject_uvz(u, v, z, K):
    """像素 + 深度(Z) → 相机坐标系 3D 点。"""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = float(z)
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)


def depth_board_in_base(cam, depth_frame, det, T_base_cam):
    """
    用 RGB 角点中心像素 + RealSense 深度反投影得到板心（基座系）。
    法向优先用四角深度点拟合；失败则退回 ArUco PnP 姿态。
    """
    corners = np.asarray(det["corners"], float).reshape(4, 2)
    u_c, v_c = corners.mean(axis=0)
    z_c = median_depth(depth_frame, u_c, v_c, win=25)
    if z_c is None:
        return None

    p_cam = backproject_uvz(u_c, v_c, z_c, cam.K)
    p_base = (T_base_cam @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]

    pts_cam = []
    for u, v in corners:
        z = median_depth(depth_frame, u, v, win=11)
        if z is None:
            continue
        pts_cam.append(backproject_uvz(u, v, z, cam.K))
    n_base = None
    if len(pts_cam) >= 3:
        P = np.asarray(pts_cam, float)
        P0 = P.mean(axis=0)
        _, _, vh = np.linalg.svd(P - P0)
        n_cam = vh[-1]
        n_cam = n_cam / (np.linalg.norm(n_cam) + 1e-12)
        # 法向朝向相机（相机看向 +Z，板在前方，法向应有 -Z 分量朝向相机）
        if n_cam[2] > 0:
            n_cam = -n_cam
        R = T_base_cam[:3, :3]
        n_base = R @ n_cam
        n_base = n_base / (np.linalg.norm(n_base) + 1e-12)

    return {
        "uv": (float(u_c), float(v_c)),
        "depth_m": float(z_c),
        "p_cam": p_cam,
        "p_base": p_base,
        "n_base": n_base,
        "corner_depth_count": len(pts_cam),
    }


def board_normal_out(T_bm, p_tcp):
    n = np.asarray(T_bm[:3, 2], float)
    n = n / (np.linalg.norm(n) + 1e-12)
    if np.dot(np.asarray(p_tcp, float) - T_bm[:3, 3], n) < 0:
        n = -n
    return n


def board_up_axis(n):
    world_up = np.array([0.0, 0.0, 1.0])
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    up = world_up - n * float(np.dot(world_up, n))
    if np.linalg.norm(up) < 1e-3:
        up = np.array([0.0, 1.0, 0.0])
        up = up - n * float(np.dot(up, n))
    return up / (np.linalg.norm(up) + 1e-12)


def flat_approach_rotation(n_out):
    """
    夹爪平进墙面：
      X = 指向板内（-n_out）
      Z = 板上向上
      Y = Z × X
    """
    n = np.asarray(n_out, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    x = -n
    z = board_up_axis(n)
    y = np.cross(z, x)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
        y = y - x * float(np.dot(y, x))
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.column_stack([x, y, z])


def preview_until_confirm(cam, T_gc, robot, q_hold):
    """
    打开实时相机窗口。SPACE/回车：用当前帧去点击；ESC/q：退出。
    预览期间持续 MIT 保持，避免臂发软。
    近距离时放宽边框/重投影阈值（否则板很大时会被 near_border 误杀）。
    """
    # 触碰预览/近距离：角点常靠近画面边缘
    cam.border_px = 5
    cam.max_reproj_px = 5.0
    cam.min_side_px = 60

    win = "D435i touch preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    print("已打开相机画面。对准标定板后按 SPACE/回车 开始点击；ESC 退出。")
    last_ok = None
    vel = [0.0] * robot.motor_count
    q_hold = np.asarray(q_hold, float).tolist()
    try:
        while True:
            tqe = np.asarray(robot.get_Gravity(q_hold), float)
            tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
            robot.pos_vel_tqe_kp_kd(q_hold, vel, tqe.tolist(), HOLD_KP, HOLD_KD)

            img, depth = cam.grab_aligned()
            if img is None:
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    return None
                continue

            det = cam.detect(img, depth)
            # 即使 quality 失败，也画出原始角点，方便看「看没看到」
            corners, ids, _ = cam.detector.detectMarkers(img)
            vis = img.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            if det is not None:
                vis = cam.annotate(img, det)

            status = "NO marker"
            color = (0, 0, 255)
            pack = None
            reject = getattr(cam, "last_reject", "")
            raw_ids = getattr(cam, "last_raw_ids", [])

            if det is not None:
                T_bg = get_T_base_gripper(robot, n=1)
                T_bc = T_bg @ T_gc
                T_bm = T_bc @ det["T_cam_marker"]
                depth_info = depth_board_in_base(cam, depth, det, T_bc)
                if depth_info is not None:
                    u, v = depth_info["uv"]
                    cv2.drawMarker(
                        vis,
                        (int(round(u)), int(round(v))),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        24,
                        2,
                    )
                    status = (
                        f"OK id=7  depthZ={depth_info['depth_m']:.3f}m  "
                        f"side={det.get('side_px', 0):.0f}px  SPACE=click"
                    )
                    color = (0, 180, 0)
                    pack = (det, img, T_bm, T_bc, depth_info)
                    last_ok = pack
                else:
                    status = "id=7 OK, depth invalid"
                    color = (0, 165, 255)
            elif raw_ids:
                status = f"raw ids={raw_ids} reject={reject}"
                color = (0, 165, 255)
            else:
                status = f"NO marker  reject={reject or 'no_aruco'}"

            cv2.putText(
                vis, status, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )
            cv2.putText(
                vis,
                "SPACE/Enter: go click   ESC: quit",
                (16, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("预览取消")
                return None
            if key in (32, 13):
                if pack is not None:
                    print("确认，开始点击")
                    return pack
                if last_ok is not None:
                    print("当前帧未检出，使用上一帧有效检测")
                    return last_ok
                print(f"还不能点：{status}")
    finally:
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass
        cv2.destroyAllWindows()


def get_tcp_fk(robot, q=None):
    if q is None:
        try:
            robot.send_get_motor_state_cmd()
            robot.motor_send_cmd()
            time.sleep(0.05)
        except Exception:
            pass
        use_link6_frame(robot, "tool_link")
        fk = robot.forward_kinematics()
        use_link6_frame(robot, "link6")
        return np.array(fk["position"], float), np.array(fk["rotation"], float)

    use_link6_frame(robot, "tool_link")
    q = np.asarray(q, float)
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    oMf = robot.data.oMf[robot.end_effector_frame_id]
    use_link6_frame(robot, "link6")
    return np.array(oMf.translation, float), np.array(oMf.rotation, float)


def rot_err_deg(R_a, R_b):
    R = np.asarray(R_a, float).T @ np.asarray(R_b, float)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def solve_ik_flat(robot, target_xyz, R_flat, q_seed=None, quiet=False):
    """单初始/少种子 IK。不用 multi_init，避免刷屏「DLS逆解未收敛」。"""
    use_link6_frame(robot, "tool_link")
    q_now = np.asarray(robot.get_current_pos(), float)
    seeds = []
    if q_seed is not None:
        seeds.append(np.asarray(q_seed, float))
    seeds.append(q_now)
    for d4 in (0.5, -0.5):
        q = q_now.copy()
        q[3] = float(np.clip(q[3] + d4, -1.55, 1.55))
        seeds.append(q)

    best = None
    for init_q in seeds:
        q_ik = robot.inverse_kinematics(
            target_position=np.asarray(target_xyz, float).tolist(),
            target_rotation=np.asarray(R_flat, float),
            init_q=init_q.tolist(),
            multi_init=False,
            max_iter=500,
            eps=2e-3,
        )
        if q_ik is None:
            continue
        q_ik = np.asarray(q_ik, float)
        p_pred, R_pred = get_tcp_fk(robot, q_ik)
        pos_err = float(np.linalg.norm(p_pred - np.asarray(target_xyz, float)))
        ang_err = rot_err_deg(R_pred, R_flat)
        score = pos_err + 0.002 * ang_err
        if best is None or score < best[0]:
            best = (score, q_ik, p_pred, R_pred, pos_err, ang_err)
        # 已经够好就停，别继续试种子刷屏
        if pos_err < 0.005 and ang_err < 2.0:
            break

    use_link6_frame(robot, "link6")
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5]


def stream_move_mit(robot, q_goal, duration, rate_hz=100):
    """
    连续发 MIT+重力，避免 moveJ「只发一次再空等」导致腕部失力下塌。
    """
    q_goal = np.asarray(q_goal, float)
    q0 = np.asarray(robot.get_current_pos(), float)
    steps = max(int(float(duration) * rate_hz), 1)
    dt = float(duration) / steps
    vel0 = [0.0] * robot.motor_count
    t0 = time.perf_counter()
    for k in range(steps + 1):
        s = min(1.0, k / steps)
        # 平滑 3 次多项式：s^2 (3-2s)
        a = s * s * (3.0 - 2.0 * s)
        q = q0 + a * (q_goal - q0)
        if k < steps:
            # 近似速度
            a_next = ((k + 1) / steps) ** 2 * (3.0 - 2.0 * ((k + 1) / steps))
            q_next = q0 + a_next * (q_goal - q0)
            vel = ((q_next - q) / dt).tolist()
        else:
            vel = vel0
        tqe = np.asarray(robot.get_Gravity(q.tolist()), float)
        tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
        robot.pos_vel_tqe_kp_kd(q.tolist(), vel, tqe.tolist(), HOLD_KP, HOLD_KD)
        target = t0 + (k + 1) * dt
        while time.perf_counter() < target:
            time.sleep(0.0002)
    # 到位后立刻用零速再钉住，不留控制空窗
    tqe = np.asarray(robot.get_Gravity(q_goal.tolist()), float)
    tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
    robot.pos_vel_tqe_kp_kd(q_goal.tolist(), vel0, tqe.tolist(), HOLD_KP, HOLD_KD)


def hold_joints_mit(robot, q, seconds=None):
    """连续 MIT+重力保持，防止『像松了一样』往下掉。"""
    q = np.asarray(q, float).tolist()
    vel = [0.0] * robot.motor_count
    if seconds is None:
        print("MIT 保持中（有重力补偿）。Ctrl+C 结束。")
        try:
            while True:
                tqe = np.asarray(robot.get_Gravity(q), float)
                tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
                robot.pos_vel_tqe_kp_kd(q, vel, tqe.tolist(), HOLD_KP, HOLD_KD)
                time.sleep(0.008)
        except KeyboardInterrupt:
            print("\n停止保持")
        return
    t_end = time.time() + float(seconds)
    while time.time() < t_end:
        tqe = np.asarray(robot.get_Gravity(q), float)
        tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
        robot.pos_vel_tqe_kp_kd(q, vel, tqe.tolist(), HOLD_KP, HOLD_KD)
        time.sleep(0.008)


def hold_joints(robot, q, seconds=None):
    hold_joints_mit(robot, q, seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", default=str(DEFAULT_CALIB))
    parser.add_argument("--config", default=str(SCRIPT_DIR / "handeye_d435i_config.yaml"))
    parser.add_argument("--standoff", type=float, default=DEFAULT_STANDOFF_M)
    parser.add_argument(
        "--up_m",
        type=float,
        default=DEFAULT_UP_M,
        help=f"沿板面上抬补偿(m)，默认 {DEFAULT_UP_M}（约 2.5cm，勿用 4cm）",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--hold_s", type=float, default=0.5, help="点到位后停留秒数再回起始，默认 0.5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    with open(args.calib, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)
    T_gc = np.array(calib["T_gripper_cam"], dtype=float)

    if not args.dry_run and not check_serial_free():
        raise RuntimeError("串口被占用，请先停 app.py")

    print("外参 T_link6_cam t =", np.round(T_gc[:3, 3], 4).tolist())
    print(
        f"流程：到固定起始 → 深度板心 + 上抬 {args.up_m*1000:.0f} mm "
        f"→ 平进点击 → 回起始"
    )
    print(f"固定起始关节 FIXED_Q_START={FIXED_Q_START.tolist()}")

    cam = RealSenseAruco(cfg)
    robot = None
    q_home = FIXED_Q_START.copy()
    try:
        robot = Panthera()
        if getattr(robot, "motor_count", 0) < 6:
            raise RuntimeError(f"电机数异常 {robot.motor_count}")
        use_link6_frame(robot, "link6")
        try:
            robot.send_get_motor_state_cmd()
            robot.motor_send_cmd()
            time.sleep(0.1)
        except Exception:
            pass

        q_now = np.asarray(robot.get_current_pos(), float)
        print(f"当前关节 {np.round(q_now, 3).tolist()}")
        if not args.dry_run:
            print("先移动到固定起始姿态...")
            stream_move_mit(robot, q_home, 4.0)
            hold_joints_mit(robot, q_home, seconds=0.3)
            q_now = np.asarray(robot.get_current_pos(), float)
            print(f"已到起始 {np.round(q_now, 3).tolist()}")
        else:
            print("dry-run：跳过移动到起始")

        print("打开相机预览（请确认能看到标定板）...")
        preview = preview_until_confirm(cam, T_gc, robot, q_home)
        if preview is None:
            print("未确认检测，回起始并退出")
            if not args.dry_run:
                stream_move_mit(robot, q_home, 3.0)
                hold_joints_mit(robot, q_home)
            return
        det, img, T_bm, T_bc, depth_info = preview
        # 再采一次 FK，提高基座位姿稳定性
        T_bg = get_T_base_gripper(robot, n=3)
        T_bc = T_bg @ T_gc
        T_bm = T_bc @ det["T_cam_marker"]
        # depth_info 的 p_base 需按新 T_bc 重算一次更稳
        img2, depth2 = cam.grab_aligned()
        if img2 is not None and depth2 is not None:
            det2 = cam.detect(img2, depth2)
            if det2 is not None:
                di = depth_board_in_base(cam, depth2, det2, T_bc)
                if di is not None:
                    det, img, depth_info = det2, img2, di
                    T_bm = T_bc @ det["T_cam_marker"]

        if depth_info is None:
            raise RuntimeError("板心深度无效，无法用深度反投影（请检查 D435i 深度）")

        tvec = np.asarray(det["tvec"], float)
        p_pnp = np.asarray(T_bm[:3, 3], float)
        p_depth = np.asarray(depth_info["p_base"], float)
        delta = p_depth - p_pnp

        print("--- 深度用法（本版点击用深度反投影，不用 PnP 中心）---")
        print(
            f"  板心像素 uv=({depth_info['uv'][0]:.1f}, {depth_info['uv'][1]:.1f})  "
            f"depth_Z={depth_info['depth_m']:.4f} m"
        )
        print(f"  相机系深度点 {np.round(depth_info['p_cam'], 4).tolist()}")
        print(f"  PnP中心(参考) 基座 {np.round(p_pnp, 4).tolist()}")
        print(f"  深度反投影中心 基座 {np.round(p_depth, 4).tolist()}")
        print(
            f"  二者差(深度-PnP) {np.round(delta * 1000, 1).tolist()} mm  "
            f"|Δ|={np.linalg.norm(delta)*1000:.1f} mm"
        )
        print(f"  PnP tvec={np.round(tvec, 4).tolist()}  (仅对照)")

        p_tcp, R_now = get_tcp_fk(robot)
        # 法向：优先四角深度平面；否则 PnP
        if depth_info.get("n_base") is not None:
            n = np.asarray(depth_info["n_base"], float)
            n = n / (np.linalg.norm(n) + 1e-12)
            if np.dot(p_tcp - p_depth, n) < 0:
                n = -n
            print(f"  法向来源=四角深度平面  n={np.round(n, 3).tolist()}")
        else:
            n = board_normal_out(T_bm, p_tcp)
            print(f"  法向来源=ArUco PnP  n={np.round(n, 3).tolist()}")

        up = board_up_axis(n)
        R_flat = flat_approach_rotation(n)
        # ★ 点击瞄准：深度反投影中心 + 手眼竖直偏差补偿（默认 +4cm）
        p_center_raw = p_depth
        p_center = p_center_raw + up * float(args.up_m)
        p_aim = p_center
        target_xyz = p_aim + n * float(args.standoff)

        tip_dir_now = R_now[:, 0]
        tip_dir_flat = R_flat[:, 0]
        print(f"深度板心(原始) {np.round(p_center_raw, 4).tolist()}")
        print(
            f"补偿后瞄准     {np.round(p_center, 4).tolist()}  "
            f"(沿板面 +{args.up_m*1000:.0f} mm，修正手眼偏低)"
        )
        print(f"目标 TCP       {np.round(target_xyz, 4).tolist()}")
        print(f"当前 tip(+X)   {np.round(tip_dir_now, 3).tolist()}")
        print(f"平进 tip(+X)   {np.round(tip_dir_flat, 3).tolist()}")

        out_png = SCRIPT_DIR / "handeye_output" / "touch_detect.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        vis = cam.annotate(img, det)
        u, v = int(round(depth_info["uv"][0])), int(round(depth_info["uv"][1]))
        cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(
            vis,
            f"depthZ={depth_info['depth_m']:.3f}m",
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_png), vis)
        print(f"检测图(红十字=深度点击中心): {out_png}")

        if args.dry_run:
            print("dry-run，只算 IK")
            solved = solve_ik_flat(robot, target_xyz, R_flat)
            if solved is None:
                print("IK 失败")
            else:
                q_ik, p_pred, _, pos_err, ang_err = solved
                dq = q_ik - q_home
                print(f"IK q={np.round(q_ik, 3).tolist()}")
                print(f"Δq(度)={np.round(np.degrees(dq), 1).tolist()}")
                print(f"joint4 Δ={np.degrees(dq[3]):.1f}°  残差 {pos_err*1000:.1f}mm / {ang_err:.1f}°")
            return

        print("\nIK 平进姿态（强制姿态，joint4 应明显变化）...")
        solved = solve_ik_flat(robot, target_xyz, R_flat)
        if solved is None:
            print("IK 失败，回初始")
            stream_move_mit(robot, q_home, 4.0)
            hold_joints_mit(robot, q_home)
            return

        q_ik, p_pred, R_pred, pos_err, ang_err = solved
        dq = q_ik - q_home
        print(f"  预测 TCP={np.round(p_pred, 4).tolist()}")
        print(f"  位置残差={pos_err * 1000:.1f} mm  姿态残差={ang_err:.1f}°")
        print(f"  目标关节={np.round(q_ik, 3).tolist()}")
        print(f"  Δq(度)={np.round(np.degrees(dq), 1).tolist()}")
        print(f"  joint4: {q_home[3]:.3f} → {q_ik[3]:.3f} rad  (Δ{np.degrees(dq[3]):.1f}°)")
        if abs(dq[3]) < 0.05:
            print("  警告：joint4 几乎不动，姿态可能仍不对")
        if pos_err > IK_POS_TOL_M:
            print(f"  警告：位置残差仍 > {IK_POS_TOL_M * 1000:.0f} mm")

        print(f"\n连续 MIT 跟踪 {args.duration:.0f}s ...")
        stream_move_mit(robot, q_ik, args.duration)

        # 高度闭环：仅偏差较大时补一次（避免刷屏）
        q_now = np.asarray(robot.get_current_pos(), float)
        p_now, R_now2 = get_tcp_fk(robot)
        dz = float(p_now[2] - target_xyz[2])
        d = p_now - p_aim
        along = float(np.dot(d, n))
        planar = float(np.linalg.norm(d - along * n))
        print(
            f"  到位: TCP={np.round(p_now, 4).tolist()}  "
            f"平面 {planar*1000:.1f}mm  法向 {along*1000:.1f}mm  ΔZ={dz*1000:.1f}mm"
        )
        if abs(dz) > 0.015 or planar > 0.020:
            target_xyz = p_now + (p_aim + n * float(args.standoff) - p_now)
            print(f"  补一次高度修正 → {np.round(target_xyz, 4).tolist()}")
            solved2 = solve_ik_flat(robot, target_xyz, R_flat, q_seed=q_now)
            if solved2 is not None:
                q_ik, p_pred, _, pos_err, ang_err = solved2
                print(f"  修正残差 {pos_err*1000:.1f}mm / {ang_err:.1f}°")
                stream_move_mit(robot, q_ik, 2.0)
            else:
                print("  修正 IK 失败，保持当前到位")

        q_now = np.asarray(robot.get_current_pos(), float)
        dq_err = q_now - q_ik
        p_now, R_now2 = get_tcp_fk(robot)
        d = p_now - p_aim
        along = float(np.dot(d, n))
        planar = float(np.linalg.norm(d - along * n))
        dz = float(p_now[2] - p_aim[2])
        print("\n" + "=" * 40)
        print(f"最终 TCP {np.round(p_now, 4).tolist()}")
        print(
            f"相对补偿瞄准：平面 {planar * 1000:.1f} mm  法向 {along * 1000:.1f} mm  "
            f"ΔZ={dz * 1000:.1f} mm"
        )
        print(f"关节误差(度)={np.round(np.degrees(dq_err), 2).tolist()}")
        print(f"到位 tip(+X) {np.round(R_now2[:, 0], 3).tolist()}")
        print("=" * 40)

        print(f"停留 {args.hold_s:.1f}s（MIT+重力）...")
        hold_joints_mit(robot, q_ik, seconds=args.hold_s)

        print("回固定起始姿态...")
        stream_move_mit(robot, q_home, 4.0)
        print("已回起始。Ctrl+C 结束。")
        hold_joints_mit(robot, q_home)
    except KeyboardInterrupt:
        print("\n中断，回固定起始")
        if robot is not None:
            try:
                stream_move_mit(robot, FIXED_Q_START, 4.0)
                hold_joints_mit(robot, FIXED_Q_START)
            except KeyboardInterrupt:
                hold_joints_mit(robot, np.asarray(robot.get_current_pos(), float))
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
