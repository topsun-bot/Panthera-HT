#!/usr/bin/env python3
# ============================================================================
# 高擎 Panthera 适配层：对标瑞尔曼 RealmanClient 的抓取所需接口
# - 手眼附着 link6；抓取 TCP 用 tool_link（+X 进给）
# - 运动：MIT+重力连续跟踪（与 10_handeye_touch_test 一致）
# ============================================================================

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as SciR
from scipy.spatial.transform import Slerp

_PKG = Path(__file__).resolve().parent
_SCRIPTS = _PKG.parent
if str(_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS))

from Panthera_lib import Panthera  # noqa: E402

_DEFAULT_CFG = _PKG / 'config' / 'hardware_panthera.yaml'
MAX_TQU = [21.0, 36.0, 36.0, 21.0, 10.0, 10.0]
HOLD_KP = [45.0, 80.0, 90.0, 40.0, 30.0, 22.0]
HOLD_KD = [4.0, 6.0, 7.0, 3.5, 2.5, 1.8]


def load_hardware_config(path: Optional[str] = None) -> dict:
  p = Path(path) if path else _DEFAULT_CFG
  with open(p, encoding='utf-8') as f:
    return yaml.safe_load(f)


def _load_use_link6_frame():
  spec = importlib.util.spec_from_file_location(
      'handeye_core', _SCRIPTS / '8_handeye_d435i_calib.py')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod.use_link6_frame, mod.check_serial_free, mod.get_T_base_gripper


class PantheraClient:
  """抓取用薄封装；.robot 指向内部 Panthera 实例（供 coord_utils 签名兼容）。"""

  def __init__(self, config_path: Optional[str] = None):
    self.config_path = str(config_path or _DEFAULT_CFG)
    self.cfg = load_hardware_config(self.config_path)
    self._robot: Optional[Panthera] = None
    self._use_link6 = None
    self._check_serial = None
    self._get_T_base_gripper = None
    self.tcp_frame = str(
        self.cfg.get('arm', {}).get('fp_grasp', {}).get('tcp_frame', 'tool_link'))

  @property
  def robot(self):
    if self._robot is None:
      raise RuntimeError('先调用 connect()')
    return self._robot

  @classmethod
  def from_config(cls, path: Optional[str] = None) -> 'PantheraClient':
    return cls(path)

  def connect(self, *, require_serial_free: bool = True):
    use_link6, check_serial, get_T = _load_use_link6_frame()
    self._use_link6 = use_link6
    self._check_serial = check_serial
    self._get_T_base_gripper = get_T
    if require_serial_free and not check_serial():
      raise RuntimeError('串口被占用，请先停 Host backend/app.py')
    self._robot = Panthera()
    if getattr(self._robot, 'motor_count', 0) < 6:
      raise RuntimeError(f'电机数异常: {getattr(self._robot, "motor_count", 0)}')
    # 手眼用 link6；运动规划位姿按 tcp_frame
    use_link6(self._robot, 'link6')
    try:
      self._robot.send_get_motor_state_cmd()
      self._robot.motor_send_cmd()
      time.sleep(0.05)
    except Exception:
      pass
    return self

  def disconnect(self) -> None:
    self._robot = None

  @property
  def connected(self) -> bool:
    return self._robot is not None

  def _set_fk_frame(self, name: str) -> None:
    self._use_link6(self._robot, name)

  def T_base_link6(self) -> np.ndarray:
    """当前法兰 link6 相对基座 4×4（手眼换算用）。"""
    return np.asarray(self._get_T_base_gripper(self._robot, n=1), dtype=float)

  def T_base_tcp(self) -> np.ndarray:
    """当前抓取 TCP（默认 tool_link）相对基座 4×4。"""
    self._set_fk_frame(self.tcp_frame)
    fk = self._robot.forward_kinematics()
    self._set_fk_frame('link6')
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(fk['rotation'], float)
    T[:3, 3] = np.asarray(fk['position'], float)
    return T

  def current_pose6(self) -> tuple[int, list[float]]:
    """读当前 TCP pose6（与规划坐标系一致）。"""
    try:
      T = self.T_base_tcp()
      return 0, matrix_to_pose6(T)
    except Exception:
      return -1, []

  def matrix_to_pose6(self, T) -> list[float]:
    return matrix_to_pose6(T)

  def pose6_to_matrix(self, pose6) -> np.ndarray:
    return pose6_to_matrix(pose6)

  def ensure_init_pose6(self) -> list[float]:
    """保证 cfg 里有 init_pose6：缺则按 init_joints 做 tool_link FK。"""
    arm = self.cfg.setdefault('arm', {})
    pose = arm.get('init_pose6') or []
    if pose and len(pose) >= 6:
      return [float(x) for x in pose[:6]]
    q = np.asarray(arm['init_joints_rad'], float)
    self._set_fk_frame(self.tcp_frame)
    # 用目标关节算 FK（不依赖当前姿态）
    import pinocchio as pin
    qq = np.zeros(self._robot.model.nq)
    for i, joint_name in enumerate(self._robot.joint_names):
      if i < len(q):
        jid = self._robot.model.getJointId(joint_name)
        idx = self._robot.model.joints[jid].idx_q
        qq[idx] = q[i]
    pin.forwardKinematics(self._robot.model, self._robot.data, qq)
    pin.updateFramePlacements(self._robot.model, self._robot.data)
    oMf = self._robot.data.oMf[self._robot.end_effector_frame_id]
    self._set_fk_frame('link6')
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.array(oMf.rotation, float)
    T[:3, 3] = np.array(oMf.translation, float)
    pose6 = matrix_to_pose6(T)
    arm['init_pose6'] = pose6
    print(f'  [auto] init_pose6 from joints FK: '
          f'xyz(mm)={[round(x*1000,1) for x in pose6[:3]]}')
    return pose6

  def movej(self, joints_rad: Sequence[float], duration: float = 4.0,
            block: bool = True) -> int:
    """关节空间 MIT 平滑移动。joints 单位：弧度。"""
    q = np.asarray(joints_rad, float)
    ok = stream_move_mit(self._robot, q, duration)
    if ok:
      # 到位后短时硬保持，避免立刻断流发软
      hold_joints_mit(self._robot, q, seconds=0.35)
    return 0 if ok else -1

  def movel(self, pose6: Sequence[float], duration: float = 3.0,
            block: bool = True, *, require_orient: bool = True) -> int:
    """笛卡尔：照抄瑞尔曼 movel —— 目标是完整 pose6（位置+姿态）。

    require_orient=True：对目标 xyz+R 求 IK（绕进给轴滚转搜索）；远点不可解时
    用 Slerp 路点逐步逼近（模拟控制器笛卡尔插值），禁止只在原地拧腕。
    require_orient=False：保当前姿态只追位置（途中微调用）。
    """
    T = pose6_to_matrix(pose6)
    xyz = T[:3, 3]
    R = T[:3, :3]
    ax = str(
        self.cfg.get('arm', {}).get('fp_grasp', {}).get('approach_axis', 'x')
    ).strip().lower()
    if require_orient:
      q = solve_ik_tcp(
          self._robot, xyz, R, tcp_frame=self.tcp_frame, approach_axis=ax,
          require_orient=True, max_ang_err_deg=22.0)
      if q is None:
        ok = _stream_pose_waypoints_slerp(
            self._robot, xyz, R, duration,
            tcp_frame=self.tcp_frame, approach_axis=ax)
        return 0 if ok else -2
      ok = stream_move_mit(self._robot, q, duration)
      if ok:
        hold_joints_mit(self._robot, q, seconds=0.15)
      return 0 if ok else -1
    q = solve_ik_tcp_translate(
        self._robot, xyz, tcp_frame=self.tcp_frame)
    if q is None:
      return -2
    ok = stream_move_mit(self._robot, q, duration)
    if ok:
      hold_joints_mit(self._robot, q, seconds=0.15)
    return 0 if ok else -1

  def gripper_open(self) -> None:
    g = self.cfg.get('gripper', {})
    self._robot.gripper_open(
        pos=float(g.get('open_pos', 1.6)),
        vel=float(g.get('open_vel', 0.5)),
        max_tqu=float(g.get('max_tqu', 0.5)))

  def gripper_close(self) -> None:
    g = self.cfg.get('gripper', {})
    self._robot.gripper_close(
        pos=float(g.get('close_pos', 0.0)),
        vel=float(g.get('close_vel', 0.5)),
        max_tqu=float(g.get('max_tqu', 0.5)))

  def slow_stop(self) -> int:
    """保持当前关节（无轨迹队列可清）。"""
    try:
      q = np.asarray(self._robot.get_current_pos(), float)
      hold_joints_mit(self._robot, q, seconds=0.2)
    except Exception:
      pass
    return 0

  def clear_current_trajectory(self) -> int:
    return 0


def pose6_to_matrix(pose6) -> np.ndarray:
  p = np.asarray(pose6, dtype=float).reshape(6)
  T = np.eye(4, dtype=float)
  T[:3, 3] = p[:3]
  T[:3, :3] = SciR.from_euler('xyz', p[3:6]).as_matrix()
  return T


def matrix_to_pose6(T) -> list[float]:
  T = np.asarray(T, dtype=float).reshape(4, 4)
  eul = SciR.from_matrix(T[:3, :3]).as_euler('xyz')
  return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
          float(eul[0]), float(eul[1]), float(eul[2])]


def stream_move_mit(robot: Panthera, q_goal, duration: float, rate_hz: float = 100.0) -> bool:
  q_goal = np.asarray(q_goal, float)
  q0 = np.asarray(robot.get_current_pos(), float)
  if len(q_goal) != robot.motor_count:
    print(f'目标关节长度 {len(q_goal)} != {robot.motor_count}')
    return False
  steps = max(int(float(duration) * rate_hz), 1)
  dt = float(duration) / steps
  vel0 = [0.0] * robot.motor_count
  t0 = time.perf_counter()
  for k in range(steps + 1):
    s = min(1.0, k / steps)
    a = s * s * (3.0 - 2.0 * s)
    q = q0 + a * (q_goal - q0)
    if k < steps:
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
  tqe = np.asarray(robot.get_Gravity(q_goal.tolist()), float)
  tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
  robot.pos_vel_tqe_kp_kd(q_goal.tolist(), vel0, tqe.tolist(), HOLD_KP, HOLD_KD)
  return True


def hold_joints_mit(robot: Panthera, q, seconds: float = 0.2) -> None:
  q = np.asarray(q, float).tolist()
  vel = [0.0] * robot.motor_count
  t_end = time.time() + float(seconds)
  while time.time() < t_end:
    tqe = np.asarray(robot.get_Gravity(q), float)
    tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
    robot.pos_vel_tqe_kp_kd(q, vel, tqe.tolist(), HOLD_KP, HOLD_KD)
    time.sleep(0.008)


def hold_init_forever(config_path: Optional[str] = None,
                      joints_rad: Optional[Sequence[float]] = None) -> None:
  """持续 MIT 锁位置1。断流就会发软，YOLO/FP 期间必须由本循环占串口。"""
  client = PantheraClient.from_config(config_path)
  client.connect(require_serial_free=True)
  q = list(joints_rad) if joints_rad is not None else list(
      client.cfg['arm']['init_joints_rad'])
  if len(q) != client.robot.motor_count:
    raise RuntimeError(f'hold 关节数 {len(q)} != {client.robot.motor_count}')
  print(f'[hold] 锁定位置1 joints={[round(float(x), 3) for x in q]}  Ctrl+C 结束')
  vel = [0.0] * client.robot.motor_count
  try:
    while True:
      tqe = np.asarray(client.robot.get_Gravity(q), float)
      tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
      client.robot.pos_vel_tqe_kp_kd(q, vel, tqe.tolist(), HOLD_KP, HOLD_KD)
      time.sleep(0.008)
  except KeyboardInterrupt:
    print('[hold] 结束')
  finally:
    client.disconnect()


def solve_ik_tcp_translate(
    robot: Panthera,
    target_xyz,
    *,
    tcp_frame: str = 'tool_link',
) -> Optional[np.ndarray]:
  """平移到目标点，优先保持当前腕姿（起始已转朝下后，带着朝下腕去正上方）。

  keepR 失败时：多种子只追位置，再把腕关节锁回当前（朝下腕），
  TCP 可微漂，由上层「上方纠偏」拉回 XY。
  """
  use_link6, _, _ = _load_use_link6_frame()
  use_link6(robot, tcp_frame)
  q_now = np.asarray(robot.get_current_pos(), float)
  xyz = np.asarray(target_xyz, float)
  import pinocchio as pin

  def _fk(q_ik):
    qq = np.zeros(robot.model.nq)
    for i, joint_name in enumerate(robot.joint_names):
      if i < len(q_ik):
        jid = robot.model.getJointId(joint_name)
        qq[robot.model.joints[jid].idx_q] = q_ik[i]
    pin.forwardKinematics(robot.model, robot.data, qq)
    pin.updateFramePlacements(robot.model, robot.data)
    oMf = robot.data.oMf[robot.end_effector_frame_id]
    return (np.array(oMf.translation, float), np.array(oMf.rotation, float))

  p_now, R_now = _fk(q_now)
  init_q = np.array([1.59, 1.15, 0.48, 0.0, -0.05, 0.04], float)
  wrist = q_now[3:6].copy()

  def _try(xyz_t, R_t, seed, attempts=16, max_pos_err=0.035):
    q_ik = robot.inverse_kinematics(
        target_position=np.asarray(xyz_t, float).tolist(),
        target_rotation=(None if R_t is None else np.asarray(R_t, float)),
        init_q=np.asarray(seed, float).tolist(),
        multi_init=True,
        num_attempts=int(attempts),
        max_iter=800,
        eps=5e-3,
    )
    if q_ik is None:
      return None
    q_ik = np.asarray(q_ik, float)
    p, _R = _fk(q_ik)
    if float(np.linalg.norm(p - np.asarray(xyz_t, float))) > max_pos_err:
      return None
    return q_ik

  # 1) 保当前腕姿（起始转朝下后的 R）
  q = _try(xyz, R_now, q_now, attempts=16)
  if q is not None:
    use_link6(robot, 'link6')
    print(f'  IK ok [translate-keepR] xyz(mm)={np.round(xyz*1000,1).tolist()}')
    return q

  seeds = [q_now, init_q,
           np.array([1.2, 0.9, 0.8, *wrist], float),
           np.array([1.8, 1.3, 0.3, *wrist], float),
           np.array([0.8, 1.0, 1.0, *wrist], float),
           np.array([1.4, 1.0, 0.6, 0.0, -0.8, 0.0], float)]

  for seed in seeds:
    q = _try(xyz, R_now, seed, attempts=10)
    if q is not None:
      use_link6(robot, 'link6')
      print(f'  IK ok [translate-keepR-seed] xyz(mm)={np.round(xyz*1000,1).tolist()}')
      return q

  # 2) 追位置 + 锁回朝下腕关节（保证看起来是竖直伸过去，不是观察姿斜伸）
  for seed in seeds:
    q_pos = _try(xyz, None, seed, attempts=12, max_pos_err=0.05)
    if q_pos is None:
      continue
    q_mix = q_pos.copy()
    q_mix[3:6] = wrist
    p_m, _R_m = _fk(q_mix)
    drift = float(np.linalg.norm(p_m - xyz)) * 1000.0
    use_link6(robot, 'link6')
    print(f'  IK ok [translate-pos+lockWrist] xyz(mm)={np.round(xyz*1000,1).tolist()} '
          f'混锁漂移={drift:.0f}mm（随后纠偏）')
    return q_mix

  # 3) 中点再终点
  mid = 0.5 * (p_now + xyz)
  q_cur = q_now.copy()
  for xyz_t in (mid, xyz):
    q_step = None
    for seed in (q_cur, init_q):
      q_step = _try(xyz_t, None, seed, attempts=12, max_pos_err=0.05)
      if q_step is not None:
        break
    if q_step is None:
      use_link6(robot, 'link6')
      print(f'  IK 过渡平移失败: xyz={np.round(xyz, 4).tolist()}')
      return None
    q_mix = q_step.copy()
    q_mix[3:6] = wrist
    q_cur = q_mix
  use_link6(robot, 'link6')
  print(f'  IK ok [translate-via-mid+lockWrist] xyz(mm)={np.round(xyz*1000,1).tolist()}')
  return q_cur


def solve_ik_orient_best_effort(
    robot: Panthera,
    target_xyz,
    R_target,
    *,
    tcp_frame: str = 'tool_link',
    approach_axis: str = 'x',
    max_tip_err_deg: float = 45.0,
) -> Optional[np.ndarray]:
  """在正上方转腕：腕关节网格优先；XY 漂移交给后续「上方纠偏」。

  远点满 6D 常超限。只要 tip 有改善就执行，避免因漂移门槛过严整次失败。
  """
  use_link6, _, _ = _load_use_link6_frame()
  use_link6(robot, tcp_frame)
  q_now = np.asarray(robot.get_current_pos(), float)
  xyz = np.asarray(target_xyz, float)
  R_tgt = np.asarray(R_target, float).reshape(3, 3)
  approach = R_tgt[:, 0].copy()  # 高擎 +X 进给
  approach = approach / max(float(np.linalg.norm(approach)), 1e-9)

  import pinocchio as pin

  def _fk(q_ik):
    qq = np.zeros(robot.model.nq)
    for i, joint_name in enumerate(robot.joint_names):
      if i < len(q_ik):
        jid = robot.model.getJointId(joint_name)
        qq[robot.model.joints[jid].idx_q] = q_ik[i]
    pin.forwardKinematics(robot.model, robot.data, qq)
    pin.updateFramePlacements(robot.model, robot.data)
    oMf = robot.data.oMf[robot.end_effector_frame_id]
    return (np.array(oMf.translation, float), np.array(oMf.rotation, float))

  def _ang(Ra, Rb):
    Re = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    c = float(np.clip((np.trace(Re) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

  def _tip_align_deg(R):
    x = np.asarray(R, float)[:, 0]
    x = x / max(float(np.linalg.norm(x)), 1e-9)
    c = float(np.clip(np.dot(x, approach), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

  def _rot_x(R0, deg):
    th = np.radians(float(deg))
    c, s = np.cos(th), np.sin(th)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    return R0 @ Rx

  p_now, R_now = _fk(q_now)
  ang0 = _ang(R_now, R_tgt)
  tip0 = _tip_align_deg(R_now)
  print(f'  转腕前: 姿态差={ang0:.1f}deg  tip(+X)对齐进给差={tip0:.1f}deg')

  # 1) 近处满姿态（远点跳过）
  reach_xy = float(np.linalg.norm(xyz[:2]))
  if reach_xy < 0.42 and tip0 > 5.0:
    for roll in (0, 90, -90, 180):
      R_try = _rot_x(R_tgt, roll)
      q_ik = robot.inverse_kinematics(
          target_position=xyz.tolist(),
          target_rotation=R_try,
          init_q=q_now.tolist(),
          multi_init=True,
          num_attempts=6,
          max_iter=300,
          eps=4e-3,
      )
      if q_ik is None:
        continue
      q_ik = np.asarray(q_ik, float)
      p, R = _fk(q_ik)
      if float(np.linalg.norm(p - xyz)) > 0.025:
        continue
      tip = _tip_align_deg(R)
      if tip <= max_tip_err_deg:
        use_link6(robot, 'link6')
        print(f'  IK ok [site-orient roll={roll}°] '
              f'姿态差={_ang(R, R_tgt):.1f}deg tip对齐={tip:.1f}deg')
        return q_ik

  # 2) 腕网格：允许一定 TCP 漂（后面有「上方纠偏」拉回 XY）
  best = None
  j4s = np.linspace(-1.55, 1.55, 15)
  j5s = np.linspace(-1.65, 1.65, 15)
  j6s = np.linspace(-2.45, 2.45, 13)
  for j4 in j4s:
    for j5 in j5s:
      for j6 in j6s:
        q = q_now.copy()
        q[3], q[4], q[5] = float(j4), float(j5), float(j6)
        p, R = _fk(q)
        drift = float(np.linalg.norm(p - p_now))
        if drift > 0.12:
          continue
        tip = _tip_align_deg(R)
        # tip 优先；轻微惩罚漂移（纠偏会修 XY）
        score = tip + 25.0 * drift + 0.03 * _ang(R, R_tgt)
        if best is None or score < best[0]:
          best = (score, q, tip, _ang(R, R_tgt), p, drift)

  use_link6(robot, 'link6')
  if best is None:
    print('  转腕失败: 腕网格无可行解')
    return None

  tip_best = float(best[2])
  improved = float(tip0 - tip_best)
  drift_mm = float(best[5]) * 1000.0
  ok_abs = tip_best <= max_tip_err_deg
  ok_rel = improved >= 8.0
  ok_any = improved >= 5.0
  if ok_abs or ok_rel or ok_any:
    if ok_abs:
      tag = '达标'
    elif ok_rel:
      tag = f'尽力(改善{improved:.1f}°)'
    else:
      tag = f'微改善{improved:.1f}°'
    print(f'  IK ok [wrist-grid/{tag}] tip对齐 {tip0:.1f}→{tip_best:.1f}deg '
          f'姿态差 {ang0:.1f}→{best[3]:.1f}deg TCP漂移={drift_mm:.0f}mm '
          f'(随后纠偏回正上方)')
    return best[1]

  print(f'  转腕失败: tip {tip0:.1f}→{tip_best:.1f}deg 无改善')
  return None



def _stream_pose_waypoints_slerp(
    robot: Panthera,
    target_xyz,
    R_target,
    duration: float,
    *,
    tcp_frame: str = 'tool_link',
    approach_axis: str = 'x',
    n_wp: int = 3,
) -> bool:
  """远点满 6D 一步不可解：前 n-1 点 Slerp 逼近姿态，末点 keepR 钉死上方 XYZ。

  高擎在 y≈0.5m 满姿态常超限；末点若仍强求满姿态会停在物体上方却不抓。
  末点改为「到位 + 保持已逼近的腕姿」，保证继续进给关爪。
  """
  use_link6, _, _ = _load_use_link6_frame()
  use_link6(robot, tcp_frame)
  q_now = np.asarray(robot.get_current_pos(), float)
  xyz = np.asarray(target_xyz, float)
  R_tgt = np.asarray(R_target, float).reshape(3, 3)
  import pinocchio as pin

  def _fk(q_ik):
    qq = np.zeros(robot.model.nq)
    for i, joint_name in enumerate(robot.joint_names):
      if i < len(q_ik):
        jid = robot.model.getJointId(joint_name)
        qq[robot.model.joints[jid].idx_q] = q_ik[i]
    pin.forwardKinematics(robot.model, robot.data, qq)
    pin.updateFramePlacements(robot.model, robot.data)
    oMf = robot.data.oMf[robot.end_effector_frame_id]
    return (np.array(oMf.translation, float), np.array(oMf.rotation, float))

  p0, R0 = _fk(q_now)
  use_link6(robot, 'link6')
  print(f'  IK 一步不可达，Slerp 逼近×{n_wp - 1} + 末点 keepR 钉上方 → '
        f'{np.round(xyz*1000,1).tolist()}')
  dt = float(duration) / float(max(n_wp, 1))
  q_seed = q_now.copy()
  slerp = Slerp([0.0, 1.0], SciR.from_matrix(np.stack([R0, R_tgt], axis=0)))

  # 前 n_wp-1：姿态逐步跟上，位置走到中途
  for k in range(1, n_wp):
    a = float(k) / float(n_wp)  # 不到 1.0
    xyz_i = (1.0 - a) * p0 + a * xyz
    R_i = slerp([a]).as_matrix()[0]
    q = solve_ik_tcp(
        robot, xyz_i, R_i, tcp_frame=tcp_frame, approach_axis=approach_axis,
        require_orient=True, max_ang_err_deg=32.0, init_q_override=q_seed)
    if q is None:
      q = solve_ik_tcp(
          robot, xyz_i, R_i, tcp_frame=tcp_frame, approach_axis=approach_axis,
          require_orient=False, max_ang_err_deg=50.0, init_q_override=q_seed)
    if q is None:
      # 中途失败：至少平移过去，腕姿用当前
      q = solve_ik_tcp_translate(robot, xyz_i, tcp_frame=tcp_frame)
    if q is None:
      print(f'  路点 {k}/{n_wp} IK 失败 xyz(mm)={np.round(xyz_i*1000,1).tolist()}')
      return False
    ok = stream_move_mit(robot, q, max(0.7, dt))
    if not ok:
      return False
    hold_joints_mit(robot, q, seconds=0.08)
    q_seed = np.asarray(q, float)
    print(f'  路点 {k}/{n_wp} 到位')

  # 末点：钉死目标 XYZ，保持当前（已逼近的）腕姿 —— 满姿态不可解也不中断
  q_final = solve_ik_tcp_translate(robot, xyz, tcp_frame=tcp_frame)
  if q_final is None:
    print('  末点 keepR 也失败，放弃上方+转腕')
    return False
  ok = stream_move_mit(robot, q_final, max(0.8, dt))
  if not ok:
    return False
  hold_joints_mit(robot, q_final, seconds=0.1)
  use_link6(robot, tcp_frame)
  p_f, R_f = _fk(q_final)
  use_link6(robot, 'link6')
  xy_err = float(np.linalg.norm(p_f[:2] - xyz[:2])) * 1000.0
  tip = R_f[:, 0]
  tip = tip / max(float(np.linalg.norm(tip)), 1e-9)
  app = R_tgt[:, 0]
  app = app / max(float(np.linalg.norm(app)), 1e-9)
  tip_deg = float(np.degrees(np.arccos(
      float(np.clip(np.dot(tip, app), -1.0, 1.0)))))
  print(f'  上方到位(末点 keepR): XY偏={xy_err:.0f}mm tip对齐进给={tip_deg:.1f}deg '
        f'→ 继续进给关爪')
  return xy_err < 60.0  # 60mm 内认为到上方，允许略斜继续抓


def solve_ik_tcp(
    robot: Panthera,
    target_xyz,
    R_target,
    *,
    tcp_frame: str = 'tool_link',
    approach_axis: str = 'x',
    require_orient: bool = True,
    max_ang_err_deg: float = 18.0,
    init_q_override: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
  """对目标 xyz+R 求 IK（瑞尔曼 movel 语义：去那个位姿，不是原地拧腕）。

  绕进给轴离散滚转；多种子；禁止 keep_R_now 冒充到位。
  """
  use_link6, _, _ = _load_use_link6_frame()
  use_link6(robot, tcp_frame)
  q_now = (np.asarray(init_q_override, float)
           if init_q_override is not None
           else np.asarray(robot.get_current_pos(), float))
  xyz = np.asarray(target_xyz, float)
  R_tgt = np.asarray(R_target, float).reshape(3, 3)
  ax = str(approach_axis).strip().lower()
  init_q = np.array([1.59, 1.15, 0.48, 0.0, -0.05, 0.04], float)

  def _fk_err(q_ik, R_ref):
    import pinocchio as pin
    qq = np.zeros(robot.model.nq)
    for i, joint_name in enumerate(robot.joint_names):
      if i < len(q_ik):
        jid = robot.model.getJointId(joint_name)
        qq[robot.model.joints[jid].idx_q] = q_ik[i]
    pin.forwardKinematics(robot.model, robot.data, qq)
    pin.updateFramePlacements(robot.model, robot.data)
    oMf = robot.data.oMf[robot.end_effector_frame_id]
    pos_err = float(np.linalg.norm(np.array(oMf.translation) - xyz))
    R_pred = np.array(oMf.rotation, float)
    R_err = R_pred.T @ R_ref
    c = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    ang_err = float(np.degrees(np.arccos(c)))
    return pos_err, ang_err

  def _rot_about_approach(R0, deg):
    th = np.radians(float(deg))
    c, s = np.cos(th), np.sin(th)
    if ax == 'x':
      Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
      return R0 @ Rx
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    return R0 @ Rz

  seeds = [
      q_now, init_q,
      np.array([1.2, 0.9, 0.8, 0.0, -0.8, 0.0], float),
      np.array([1.8, 1.3, 0.3, 0.5, -1.0, 0.0], float),
      np.array([0.9, 1.1, 0.9, -0.3, -1.0, 0.4], float),
  ]
  roll_degs = [0, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 135, -135, 150, -150, 180]
  best = None
  max_pos = 0.020 if require_orient else 0.035
  for seed in seeds:
    for roll in roll_degs:
      R_try = _rot_about_approach(R_tgt, roll)
      q_ik = robot.inverse_kinematics(
          target_position=xyz.tolist(),
          target_rotation=R_try,
          init_q=np.asarray(seed, float).tolist(),
          multi_init=True,
          num_attempts=12,
          max_iter=700,
          eps=4e-3,
      )
      if q_ik is None:
        continue
      q_ik = np.asarray(q_ik, float)
      pos_err, ang_err = _fk_err(q_ik, R_try)
      if pos_err > max_pos:
        continue
      if require_orient and ang_err > max_ang_err_deg:
        continue
      score = pos_err + 0.001 * ang_err + 0.0001 * abs(roll)
      if best is None or score < best[0]:
        best = (score, q_ik, pos_err, ang_err, roll)
      if pos_err < 0.010 and ang_err < 10.0:
        break
    if best is not None and best[2] < 0.012 and best[3] < 12.0:
      break

  use_link6(robot, 'link6')
  if best is None:
    print(f'  IK 失败(目标位姿): xyz={np.round(xyz, 4).tolist()} '
          f'approach_axis={ax}')
    return None
  print(f'  IK ok [上方+转腕 roll={best[4]:.0f}°] '
        f'pos_err={best[2]*1000:.1f}mm ang_err={best[3]:.1f}deg')
  return best[1]


if __name__ == '__main__':
  import argparse
  ap = argparse.ArgumentParser(description='PantheraClient / 位置1 保位')
  ap.add_argument('--hold-init', action='store_true',
                  help='持续 MIT 锁定 hardware yaml 中的 init_joints_rad')
  ap.add_argument('--config', default=str(_DEFAULT_CFG))
  ap.add_argument('--joints', type=float, nargs=6, default=None,
                  help='可选：覆盖 init 关节 (rad)')
  ns = ap.parse_args()
  if ns.hold_init:
    hold_init_forever(ns.config, ns.joints)
  else:
    ap.error('请指定 --hold-init')
