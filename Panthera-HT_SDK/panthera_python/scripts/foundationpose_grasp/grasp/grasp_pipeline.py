#!/usr/bin/env python3
# ============================================================================
# Panthera-HT：FoundationPose 顶面抓取流水线（抓取+抬起）
# 规则来自瑞尔曼侧 hands/grasp（top_face）；驱动改为高擎 PantheraClient
# ============================================================================

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_GRASP_DIR = Path(__file__).resolve().parent
_PKG = _GRASP_DIR.parent  # foundationpose_grasp/
_SCRIPTS = _PKG.parent    # panthera_python/scripts/
if str(_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS))
if str(_PKG) not in sys.path:
  sys.path.insert(0, str(_PKG))

from grasp.coord_utils import (  # noqa: E402
    build_top_face_grasp_staged_pose6,
    build_init_orient_grasp_pose6,
    load_fp_grasp_cfg,
    load_hand_eye_mode,
    HAND_EYE_EYE_IN_HAND,
    ob_in_cam_to_arm_xyz,
    pose6_to_matrix,
    resolve_object_half_extents_m,
    resolve_T_base_cam,
    validate_fp_grasp_plan,
    load_gripper_tcp_to_pad_m,
)
from panthera_client import (  # noqa: E402
    PantheraClient,
    load_hardware_config,
)


def _movel(client: PantheraClient, name: str, pose6: list[float], duration: float,
           *, require_orient: bool = True) -> None:
  xyz = [round(pose6[i] * 1000, 1) for i in range(3)]
  rpy = [round(pose6[i], 3) for i in range(3, 6)]
  tag = '需转腕' if require_orient else '过渡平移'
  print(f'  movel {name} [{tag}]: xyz(mm)={xyz} rpy={rpy}  t={duration:.1f}s')
  ret = client.movel(pose6, duration=duration, block=True,
                     require_orient=require_orient)
  if ret != 0:
    raise RuntimeError(f'movel {name} failed ret={ret} target xyz(mm)={xyz}')


def _movej(client: PantheraClient, name: str, joints: list[float], duration: float) -> None:
  print(f'  movej {name}: {[round(j, 3) for j in joints]}  t={duration:.1f}s')
  ret = client.movej(joints, duration=duration, block=True)
  if ret != 0:
    raise RuntimeError(f'movej {name} 失败 ret={ret}')


def listen_ob_in_cam(port: int, wait_s: float, class_name: str) -> np.ndarray:
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.bind(('0.0.0.0', port))
  sock.settimeout(0.5)
  mats = []
  t0 = time.time()
  while time.time() - t0 < wait_s:
    try:
      data, _ = sock.recvfrom(65535)
    except socket.timeout:
      continue
    msg = json.loads(data.decode('utf-8').strip())
    if class_name and msg.get('class_name') and msg.get('class_name') != class_name:
      continue
    if 'ob_in_cam' not in msg:
      continue
    mats.append(np.asarray(msg['ob_in_cam'], dtype=float).reshape(4, 4))
  sock.close()
  if not mats:
    raise RuntimeError(f'UDP :{port} 没收到位姿')
  return np.median(np.asarray(mats), axis=0)


def _resolve_T_base_cam_panthera(
    client: PantheraClient,
    config_path: str,
    *,
    lock_T_base_link6: Optional[np.ndarray] = None,
) -> np.ndarray:
  """
  眼在手上：必须用 link6 @ T_link6_cam。
  禁止用 tool_link pose6 去乘手眼（会偏 165mm）。
  """
  if load_hand_eye_mode(config_path) != HAND_EYE_EYE_IN_HAND:
    return resolve_T_base_cam(config_path)
  if lock_T_base_link6 is not None:
    T_be = np.asarray(lock_T_base_link6, dtype=float).reshape(4, 4)
  else:
    T_be = client.T_base_link6()
  return resolve_T_base_cam(config_path, T_base_ee=T_be)


def _plan_poses(
    ob_in_cam: np.ndarray,
    cfg: dict,
    client: PantheraClient,
    config_path: str,
    *,
    lock_T_base_link6: Optional[np.ndarray] = None,
    class_name: str = '',
    meta_half_extents: Optional[list[float]] = None,
    objects_yaml: Optional[str] = None,
    quiet: bool = False,
    T_base_obj: Optional[np.ndarray] = None,
) -> tuple[list[float], list[float], list[float], list[float], np.ndarray, dict]:
  g = load_fp_grasp_cfg(cfg)
  half_extents = resolve_object_half_extents_m(
      cfg, g,
      meta_half=meta_half_extents,
      class_name=class_name or None,
      objects_yaml=objects_yaml)
  if not quiet:
    print(f'  绿框半轴(m): {half_extents}')

  if client.connected:
    client.ensure_init_pose6()
    # 同步可能写回的 init_pose6
    cfg['arm']['init_pose6'] = client.cfg['arm']['init_pose6']
  init6 = list(cfg['arm']['init_pose6'])
  if not init6 or len(init6) < 6:
    raise RuntimeError('缺少 init_pose6（连臂自动算，或在 yaml 填写）')
  R_init = pose6_to_matrix(None, init6)[:3, :3]
  mode = g['orientation_mode']

  if T_base_obj is not None:
    T_obj_fixed = np.asarray(T_base_obj, dtype=float).reshape(4, 4)
    T_bc = np.eye(4, dtype=float)
  else:
    T_obj_fixed = None
    T_bc = _resolve_T_base_cam_panthera(
        client, config_path, lock_T_base_link6=lock_T_base_link6)

  if mode != 'fp':
    if T_obj_fixed is not None:
      obj_xyz = T_obj_fixed[:3, 3].copy()
    else:
      obj_xyz = ob_in_cam_to_arm_xyz(ob_in_cam, T_bc)
    hover, pre, grasp, lift = build_init_orient_grasp_pose6(
        obj_xyz, init6, g['approach_m'], g['grasp_above_m'])
    return hover, pre, grasp, lift, obj_xyz, {}

  pad_m = load_gripper_tcp_to_pad_m(cfg)
  T_hover, T_pre, T_grasp, T_lift, T_obj, face_info = build_top_face_grasp_staged_pose6(
      ob_in_cam if T_obj_fixed is None else np.eye(4),
      T_bc,
      half_extents,
      g['approach_m'], g['lift_m'], init6,
      g['retreat_axis'], g['lift_axis'],
      grasp_face_inset_m=g['grasp_face_inset_m'],
      grasp_tcp_above_top_m=g['grasp_tcp_above_top_m'],
      gripper_tcp_to_pad_m=pad_m,
      grasp_vertical_extra_m=g.get('grasp_vertical_extra_m', 0.0),
      table_z_m=g['table_z_m'],
      grasp_table_clearance_m=g['grasp_table_clearance_m'],
      robot=None, R_init=R_init, cfg=cfg,
      T_base_obj=T_obj_fixed)

  if not quiet:
    print(f'  抓取面: {face_info.get("face")} (top) up_dot={face_info.get("up_dot", 0):.2f}')
    print(f'  关爪高度: {face_info.get("grasp_z_formula", "?")}')
    print(f'  approach_axis={face_info.get("approach_axis", g.get("approach_axis"))}  '
          f'tcp_frame={g.get("tcp_frame")}')
    if face_info.get('surface_z_mm') is not None:
      print(f'  抓取面参考 z(mm): {face_info["surface_z_mm"]}  '
            f'关爪 TCP z(mm): {face_info.get("grasp_close_z_mm", "?")}')

  hover6 = client.matrix_to_pose6(T_hover)
  pre6 = client.matrix_to_pose6(T_pre)
  grasp6 = client.matrix_to_pose6(T_grasp)
  lift6 = client.matrix_to_pose6(T_lift)
  try:
    validate_fp_grasp_plan(
        hover6, pre6, grasp6, lift6, init6, T_obj[:3, 3],
        R_init, T_grasp[:3, :3], cfg=cfg, T_base_obj=T_obj,
        half_extents=half_extents)
  except Exception as e:
    if not quiet:
      print(f'  【警告】validate: {e}')
  if not quiet:
    print(f'  抓取点(mm): {np.round(T_grasp[:3, 3]*1000, 1)}')
    approach_mm = float(np.linalg.norm(
        np.asarray(pre6[:3], float) - np.asarray(grasp6[:3], float))) * 1000.0
    print(f'  流程: 一步移到上方并转腕 → 沿进给再进 {approach_mm:.0f}mm → 关爪 → 抬起'
          f' → 回位置1 → 开爪（照抄瑞尔曼冻结支线）')
  return hover6, pre6, grasp6, lift6, T_obj[:3, 3], face_info


def run_grasp_cycle(
    ob_in_cam: np.ndarray,
    *,
    config: str,
    dry_run: bool = False,
    lock_T_base_link6: Optional[np.ndarray] = None,
    class_name: str = '',
    meta_half_extents: Optional[list[float]] = None,
    objects_yaml: Optional[str] = None,
    client: Optional[PantheraClient] = None,
    release_on_exit: bool = True,
    skip_home_on_start: bool = False,
) -> None:
  """
  client: 若传入则复用已连接臂（守护进程），默认不在退出时 disconnect。
  skip_home_on_start: 已在位置1保位时跳过开头回零，避免多余运动。
  """
  cfg = load_hardware_config(config) if client is None else client.cfg
  arm = cfg['arm']
  init_j = arm.get('init_joints_rad')
  if not init_j:
    raise RuntimeError('未配置 init_joints_rad')

  dur_home = float(arm.get('move_duration_s', 4.0))
  dur_pre = float(arm.get('approach_duration_s', 3.0))
  dur_grasp = float(arm.get('descend_duration_s', 2.5))
  dur_lift = float(arm.get('lift_duration_s', 2.5))

  owns_client = client is None
  if owns_client:
    client = PantheraClient.from_config(config)
  # 仅当本函数自己 connect 时才允许 disconnect（避免守护进程 motor brake）
  do_release = bool(release_on_exit and owns_client)

  try:
    if not dry_run:
      if owns_client:
        client.connect(require_serial_free=True)
      client.ensure_init_pose6()
      if not skip_home_on_start:
        print('\n1/6 回起始关节')
        _movej(client, 'init', init_j, dur_home)
        time.sleep(0.2)
        print('2/6 夹爪张开')
        client.gripper_open()
        time.sleep(0.8)
      else:
        print('\n1-2/6 已在位置1保位，跳过回零；确认夹爪张开')
        client.gripper_open()
        time.sleep(0.3)
      if lock_T_base_link6 is None:
        lock_T_base_link6 = client.T_base_link6()
        print('  使用当前 link6 作为手眼锁定帧')
    else:
      try:
        if owns_client:
          client.connect(require_serial_free=True)
        client.ensure_init_pose6()
        cfg = client.cfg
        if lock_T_base_link6 is None:
          lock_T_base_link6 = client.T_base_link6()
      except Exception as e:
        print(f'  [dry-run] 未连臂（{e}），用单位 T_base_link6 仅测规则')
        lock_T_base_link6 = np.eye(4)
        if not cfg['arm'].get('init_pose6'):
          cfg['arm']['init_pose6'] = [0.3, 0.0, 0.35, 3.14, 0.0, 0.0]
        client.cfg = cfg

    print('=== 模式: 冻结帧（照抄瑞尔曼：空格那一帧跑死）===')
    _hover6, pre6, grasp6, lift6, obj_xyz, face_info = _plan_poses(
        ob_in_cam, cfg, client, config,
        lock_T_base_link6=lock_T_base_link6,
        class_name=class_name, meta_half_extents=meta_half_extents,
        objects_yaml=objects_yaml)
    print(f'物体中心(m): {np.round(obj_xyz, 4)}')
    print(f'规划(mm) pre={[round(pre6[i]*1000,1) for i in range(3)]} '
          f'grasp={[round(grasp6[i]*1000,1) for i in range(3)]}')

    if dry_run:
      print('\n[preview] grasp waypoints')
      for tag, p in [('上方+转腕', pre6), ('grasp', grasp6), ('lift', lift6)]:
        xyz = [round(p[i] * 1000, 1) for i in range(3)]
        rpy = [round(p[i], 3) for i in range(3, 6)]
        print(f'  {tag}: xyz(mm)={xyz} rpy={rpy}')
      return

    # —— 瑞尔曼冻结支线：上方+转腕 → 进给关爪 → 抬起 ——
    # 高擎远点满姿态常不可解：上方用 Slerp+末点 keepR；进给/抬起保当前腕只改位置
    print('3/6 移至抓取面上方并转腕（同瑞尔曼；远点用路点逼近）')
    _movel(client, '上方+转腕', pre6, dur_pre, require_orient=True)
    # 钉死规划上方 XY（末点可能略偏）
    ret_c, cur6 = client.current_pose6()
    if ret_c == 0 and cur6:
      pre_pin = list(pre6[:3]) + list(cur6[3:6])
      xy_err = float(np.linalg.norm(
          np.asarray(cur6[:2], float) - np.asarray(pre6[:2], float))) * 1000.0
      if xy_err > 15.0:
        print(f'  钉 XY：当前偏规划上方 {xy_err:.0f}mm')
        _movel(client, '钉上方XY', pre_pin, max(1.2, dur_pre * 0.4),
               require_orient=False)
    approach_mm = float(np.linalg.norm(
        np.asarray(pre6[:3], float) - np.asarray(grasp6[:3], float))) * 1000.0
    print(f'4/6 沿进给方向进关爪（{approach_mm:.0f}mm，保上方腕姿）')
    ret_c, cur6 = client.current_pose6()
    if ret_c != 0 or not cur6:
      raise RuntimeError('上方后读 TCP 失败')
    grasp_exec = list(grasp6[:3]) + list(cur6[3:6])
    lift_exec = list(lift6[:3]) + list(cur6[3:6])
    _movel(client, '进给关爪', grasp_exec, dur_grasp, require_orient=False)

    ret, cur6 = client.current_pose6()
    if ret == 0 and cur6:
      err_xyz = (np.asarray(cur6[:3], float) - np.asarray(grasp6[:3], float)) * 1000.0
      print(f'  到位实测 TCP(mm)={np.round(np.asarray(cur6[:3])*1000,1).tolist()} '
            f'规划={np.round(np.asarray(grasp6[:3])*1000,1).tolist()} '
            f'Δxyz(mm)={np.round(err_xyz,1).tolist()}')

    print('5/8 夹爪闭合')
    client.gripper_close()
    time.sleep(1.0)
    print('6/8 抬起')
    _movel(client, '抬起', lift_exec, dur_lift, require_orient=False)
    print('7/8 回位置1（初始关节）')
    _movej(client, 'init', init_j, dur_home)
    print('8/8 张开夹爪')
    client.gripper_open()
    time.sleep(0.5)
    print('=== 抓取+抬起+回位置1+开爪 完成 ===')
  except Exception:
    # 失败也尽量回到位置1并继续保位（由守护进程接着 hold），不销毁 client
    if not dry_run and client is not None and client.connected:
      try:
        print('【抓取失败】尝试回位置1…')
        _movej(client, 'init-recover', init_j, dur_home)
      except Exception as e2:
        print(f'【抓取失败】回位置1 也失败: {e2}')
    raise
  finally:
    if do_release and client is not None:
      client.disconnect()


def main():
  parser = argparse.ArgumentParser(description='Panthera FoundationPose 顶面抓取')
  parser.add_argument(
      '--config',
      default=str(_PKG / 'config' / 'hardware_panthera.yaml'))
  parser.add_argument('--pose-file', help='JSON 含 ob_in_cam / lock_T_base_link6')
  parser.add_argument('--port', type=int, default=8899)
  parser.add_argument('--class-name', default='')
  parser.add_argument('--listen-sec', type=float, default=2.0)
  parser.add_argument('--execute', action='store_true')
  parser.add_argument('--dry-run', action='store_true',
                      help='只规划不运动（默认若不加 --execute 也是预览）')
  args = parser.parse_args()

  lock_T = None
  class_name = args.class_name
  meta_half = None
  objects_yaml = None

  if args.pose_file:
    with open(args.pose_file, encoding='utf-8') as f:
      meta = json.load(f)
    ob_in_cam = np.asarray(meta['ob_in_cam'], dtype=float).reshape(4, 4)
    class_name = str(meta.get('class_name', '') or class_name)
    if meta.get('object_half_extents_m'):
      meta_half = [float(x) for x in meta['object_half_extents_m'][:3]]
    if meta.get('objects_yaml'):
      objects_yaml = str(meta['objects_yaml'])
    raw = meta.get('lock_T_base_link6') or meta.get('T_base_link6')
    if raw is not None:
      lock_T = np.asarray(raw, dtype=float).reshape(4, 4)
    print(f'locked frame={meta.get("frame_id", "?")} class={class_name}')
  else:
    ob_in_cam = listen_ob_in_cam(args.port, args.listen_sec, class_name)

  dry = (not args.execute) or args.dry_run
  if dry:
    run_grasp_cycle(
        ob_in_cam, config=args.config, dry_run=True,
        lock_T_base_link6=lock_T, class_name=class_name,
        meta_half_extents=meta_half, objects_yaml=objects_yaml)
    print('\n[preview] 加 --execute 才会真机运动（请先停 app.py）')
    return

  print('\n*** 3 秒后开始执行 — 请准备急停 ***')
  time.sleep(3.0)
  run_grasp_cycle(
      ob_in_cam, config=args.config, dry_run=False,
      lock_T_base_link6=lock_T, class_name=class_name,
      meta_half_extents=meta_half, objects_yaml=objects_yaml)


if __name__ == '__main__':
  main()
