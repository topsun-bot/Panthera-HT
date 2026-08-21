# ============================================================================
# 手眼坐标：FP 完整 6D 位姿 → 机械臂抓取路点
#
# 眼在手外 eye_to_hand：camera_to_arm_base = T_base_cam，p_base = T_base_cam @ p_cam
# 眼在手上 eye_in_hand：camera_to_end_effector = T_ee_cam，需当前末端位姿
#                       T_base_obj = T_base_ee @ T_ee_cam @ T_cam_obj
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]  # foundationpose_grasp/
_DEFAULT_CFG = _ROOT / 'config' / 'hardware_panthera.yaml'

HAND_EYE_EYE_IN_HAND = 'eye_in_hand'
HAND_EYE_EYE_TO_HAND = 'eye_to_hand'


def _load_camera_cfg(cfg_path: Optional[str] = None) -> dict:
  p = Path(cfg_path) if cfg_path else _DEFAULT_CFG
  with open(p, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
  return cfg.get('camera', {}) or {}


def load_hand_eye_mode(cfg_path: Optional[str] = None) -> str:
  """返回 eye_in_hand（默认）或 eye_to_hand。"""
  mode = _load_camera_cfg(cfg_path).get('hand_eye_mode', HAND_EYE_EYE_IN_HAND)
  mode = str(mode).strip()
  if mode not in (HAND_EYE_EYE_IN_HAND, HAND_EYE_EYE_TO_HAND):
    raise ValueError(f'未知 hand_eye_mode: {mode!r}，应为 eye_in_hand / eye_to_hand')
  return mode


def load_T_base_cam(cfg_path: Optional[str] = None) -> np.ndarray:
  """眼在手外：读固定 T_base_cam。"""
  T = _load_camera_cfg(cfg_path).get('camera_to_arm_base')
  if T is None:
    raise ValueError('hardware.yaml 里还没有 camera_to_arm_base，先完成眼在手外标定')
  return np.asarray(T, dtype=float).reshape(4, 4)


def load_T_ee_cam(cfg_path: Optional[str] = None) -> np.ndarray:
  """眼在手上：读相机相对末端 T_ee_cam。"""
  T = _load_camera_cfg(cfg_path).get('camera_to_end_effector')
  if T is None:
    raise ValueError('hardware.yaml 里还没有 camera_to_end_effector，先完成眼在手上标定')
  return np.asarray(T, dtype=float).reshape(4, 4)


def resolve_T_base_cam(
    cfg_path: Optional[str] = None,
    *,
    robot=None,
    pose6: Optional[Union[list, np.ndarray]] = None,
    T_base_ee: Optional[np.ndarray] = None,
) -> np.ndarray:
  """
  统一得到当前帧用的 T_base_cam（把相机系点变到基座系）。
  眼在手外直接返回标定矩阵；眼在手上 = T_base_ee @ T_ee_cam。
  """
  mode = load_hand_eye_mode(cfg_path)
  if mode == HAND_EYE_EYE_TO_HAND:
    return load_T_base_cam(cfg_path)
  T_ec = load_T_ee_cam(cfg_path)
  if T_base_ee is not None:
    T_be = np.asarray(T_base_ee, dtype=float).reshape(4, 4)
  elif pose6 is not None and robot is not None:
    T_be = pose6_to_matrix(robot, pose6)
  else:
    raise ValueError('眼在手上模式需要当前末端位姿（robot+pose6 或 T_base_ee）')
  return T_be @ T_ec


def ob_in_cam_to_arm_T(
    ob_in_cam,
    T_base_cam: Optional[np.ndarray] = None,
    *,
    cfg_path: Optional[str] = None,
    robot=None,
    pose6: Optional[Union[list, np.ndarray]] = None,
    T_base_ee: Optional[np.ndarray] = None,
) -> np.ndarray:
  """物体 4×4（相机系）→ 物体 4×4（基座系）。"""
  T = np.asarray(ob_in_cam, dtype=float).reshape(4, 4)
  if T_base_cam is None:
    T_base_cam = resolve_T_base_cam(
        cfg_path, robot=robot, pose6=pose6, T_base_ee=T_base_ee)
  return T_base_cam @ T


def ob_in_cam_to_arm_xyz(
    ob_in_cam,
    T_base_cam: Optional[np.ndarray] = None,
    *,
    cfg_path: Optional[str] = None,
    robot=None,
    pose6: Optional[Union[list, np.ndarray]] = None,
    T_base_ee: Optional[np.ndarray] = None,
) -> np.ndarray:
  return ob_in_cam_to_arm_T(
      ob_in_cam, T_base_cam,
      cfg_path=cfg_path, robot=robot, pose6=pose6, T_base_ee=T_base_ee)[:3, 3].copy()


def pose6_to_matrix(robot, pose6) -> np.ndarray:
  """
  pose6 [x,y,z,rx,ry,rz] → 4×4 T_base_ee。
  欧拉角约定：固定轴 xyz（弧度）。robot 参数保留以兼容原调用签名，可为 None。
  """
  from scipy.spatial.transform import Rotation as SciR
  p = np.asarray(pose6, dtype=float).reshape(6)
  T = np.eye(4, dtype=float)
  T[:3, 3] = p[:3]
  T[:3, :3] = SciR.from_euler('xyz', p[3:6]).as_matrix()
  return T


def matrix_to_pose6(robot, T: np.ndarray) -> list[float]:
  """4×4 T_base_ee → pose6 [x,y,z,rx,ry,rz]（xyz 欧拉，弧度）。"""
  from scipy.spatial.transform import Rotation as SciR
  T = np.asarray(T, dtype=float).reshape(4, 4)
  eul = SciR.from_matrix(T[:3, :3]).as_euler('xyz')
  return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
          float(eul[0]), float(eul[1]), float(eul[2])]


def load_fp_grasp_cfg(cfg: dict) -> dict:
  """读 fp_grasp 公共参数（approach 等，不含示教档案）。"""
  g = cfg.get('arm', {}).get('fp_grasp', {})
  half = g.get('object_half_extents_m', [0.0335, 0.065, 0.0265])
  return {
      'approach_m': float(g.get('approach_m', 0.12)),
      'grasp_above_m': float(g.get('grasp_above_m', 0.08)),
      'grasp_z_offset_m': float(g.get('grasp_z_offset_m', 0.0)),
      'lift_m': float(g.get('lift_m', 0.06)),
      'retreat_axis': g.get('retreat_axis', 'pos_z'),
      'lift_axis': g.get('lift_axis', 'pos_z'),
      'orientation_mode': g.get('orientation_mode', 'fp'),
      'grasp_orientation': g.get('grasp_orientation', 'top_face'),
      'object_half_extents_m': [float(x) for x in half],
      'grasp_face_inset_m': float(g.get('grasp_face_inset_m', 0.008)),
      'symmetry_flip_min_raw_deg': float(g.get('symmetry_flip_min_raw_deg', 40.0)),
      'face_pick': g.get('face_pick', 'auto'),
      'flange_face_gap_m': float(g.get('flange_face_gap_m', 0.06)),
      'face_pick_min_cam_dot': float(g.get('face_pick_min_cam_dot', 0.45)),
      'face_pick_depth_margin_m': float(g.get('face_pick_depth_margin_m', 0.02)),
      'face_pick_large_face_ratio': float(g.get('face_pick_large_face_ratio', 0.85)),
      'lying_top_min_up_dot': float(g.get('lying_top_min_up_dot', 0.55)),
      'table_z_m': float(g.get('table_z_m', 0.0)),
      'grasp_table_clearance_m': float(g.get('grasp_table_clearance_m', 0.015)),
      'grasp_tcp_above_top_m': float(g.get('grasp_tcp_above_top_m', 0.01)),
      'grasp_vertical_extra_m': float(g.get('grasp_vertical_extra_m', 0.0)),
      'flange_above_top_m': g.get('flange_above_top_m'),
      # 实时跟踪；false 走空格冻结旧支线
      'live_track': bool(g.get('live_track', False)),
      'live_track_port': int(g.get('live_track_port', 8899)),
      'live_track_hz': float(g.get('live_track_hz', 20.0)),
      'live_track_arrive_m': float(g.get('live_track_arrive_m', 0.025)),
      'live_track_lost_frames': int(g.get('live_track_lost_frames', 12)),
      'live_track_min_delta_m': float(g.get('live_track_min_delta_m', 0.003)),
      'live_track_min_delta_deg': float(g.get('live_track_min_delta_deg', 2.0)),
      # 跟丢后是否继续用最后一次基座系目标收尾（true）还是慢停放弃本段（false）
      'live_track_finish_on_lost': bool(g.get('live_track_finish_on_lost', True)),
      # 高擎 tool_link 进给轴为 +X；瑞尔曼法兰为 +Z。默认 z 保持原规则，panthera 配 x
      'approach_axis': str(g.get('approach_axis', 'z')).strip().lower(),
      'tcp_frame': str(g.get('tcp_frame', 'link6')).strip(),
  }


def half_extents_from_mesh_bbox(bbox: np.ndarray) -> list[float]:
  """FP mesh bbox(2×3) → 物体系半轴 [hx,hy,hz]（与 center_pose 同系）。"""
  b = np.asarray(bbox, dtype=float).reshape(2, 3)
  return [float(x) for x in ((b[1] - b[0]) * 0.5)]


def half_extents_from_mesh_file(mesh_path: str, mesh_scale: float = 1.0) -> list[float]:
  """从 mesh 文件算与 FP 一致的物体系半轴。"""
  import trimesh
  mesh = trimesh.load(str(mesh_path))
  if mesh_scale != 1.0:
    mesh = mesh.copy()
    mesh.apply_scale(float(mesh_scale))
  _, extents = trimesh.bounds.oriented_bounds(mesh)
  return [float(x * 0.5) for x in extents]


def half_extents_for_class(
    class_name: str,
    objects_yaml: str | Path,
    mesh_scale: float = 1.0,
) -> list[float]:
  """objects.yaml 里按 YOLO 类名查 mesh 半轴。"""
  with open(objects_yaml, encoding='utf-8') as f:
    raw = yaml.safe_load(f) or {}
  mesh_path = None
  for info in raw.values():
    if isinstance(info, dict) and info.get('yolo_class') == class_name:
      mesh_path = info.get('mesh')
      break
  if not mesh_path:
    raise KeyError(f'objects.yaml 无类 {class_name!r}')
  p = Path(mesh_path)
  if not p.is_absolute():
    p = _ROOT / p
  return half_extents_from_mesh_file(p, mesh_scale=mesh_scale)


def resolve_object_half_extents_m(
    cfg: dict,
    g: dict,
    *,
    meta_half: Optional[Sequence[float]] = None,
    class_name: Optional[str] = None,
    objects_yaml: Optional[str] = None,
    mesh_scale: float = 1.0,
) -> list[float]:
  """
  抓取方盒半轴优先级：冻结帧 meta > mesh(类名) > hardware.yaml。
  yaml 里旧茶叶罐数值轴序可能和 mesh 不一致，优先用 mesh。
  """
  if meta_half is not None and len(meta_half) >= 3:
    return [float(x) for x in meta_half[:3]]
  if class_name and objects_yaml:
    try:
      return half_extents_for_class(class_name, objects_yaml, mesh_scale=mesh_scale)
    except Exception:
      pass
  return [float(x) for x in g['object_half_extents_m']]


def load_primary_T_obj_to_ee(cfg: dict) -> np.ndarray:
  """读顶层 T_obj_to_ee（示教一次，躺/竖靠 FP 对称规范化，不切换档案）。"""
  g = cfg.get('arm', {}).get('fp_grasp', {})
  m = g.get('T_obj_to_ee')
  if m is None:
    prof = g.get('teach_profiles', {}).get('lying', {})
    m = prof.get('T_obj_to_ee')
  if m is None:
    raise RuntimeError(
        '缺少 T_obj_to_ee：python3 hands/grasp/teach_grasp_frame.py --pose lying')
  return np.asarray(m, dtype=float).reshape(4, 4)


def load_primary_symmetry_hint(cfg: dict) -> Optional[np.ndarray]:
  """对称规范化参考朝向（顶层 teach_R_base_obj）。"""
  g = cfg.get('arm', {}).get('fp_grasp', {})
  m = g.get('teach_R_base_obj')
  if m is None:
    return None
  return np.asarray(m, dtype=float).reshape(3, 3)


def load_primary_teach_ee_xyz(cfg: dict) -> Optional[np.ndarray]:
  g = cfg.get('arm', {}).get('fp_grasp', {})
  ee = g.get('teach_ee_pose6')
  if ee is None:
    return None
  return np.asarray(ee[:3], dtype=float).reshape(3)


_TEACH_PROFILE_LABELS = {
    'lying': '躺放',
    'standing': '竖放',
}


def _profile_fields_from_dict(d: dict) -> dict:
  """从 yaml 节点抽出一份示教档案字段。"""
  T = np.eye(4, dtype=float)
  m = d.get('T_obj_to_ee')
  if m is not None:
    T = np.asarray(m, dtype=float).reshape(4, 4)
  out = {'T_obj_to_ee': T}
  if d.get('teach_R_base_obj') is not None:
    out['teach_R_base_obj'] = np.asarray(d['teach_R_base_obj'], dtype=float).reshape(3, 3)
  if d.get('teach_ee_pose6') is not None:
    ee = [float(x) for x in d['teach_ee_pose6']]
    out['teach_ee_pose6'] = ee
    out['teach_ee_xyz'] = np.asarray(ee[:3], dtype=float)
  if d.get('teach_obj_arm_xyz') is not None:
    out['teach_obj_arm_xyz'] = np.asarray(d['teach_obj_arm_xyz'], dtype=float).reshape(3)
  return out


def list_teach_profiles(cfg: dict) -> dict[str, dict]:
  """
  返回 {档案名: 示教字段}。
  支持 teach_profiles.lying / standing；旧版顶层 T_obj_to_ee 当作 default。
  """
  g = cfg.get('arm', {}).get('fp_grasp', {})
  raw = g.get('teach_profiles')
  if isinstance(raw, dict) and raw:
    out = {}
    for name, body in raw.items():
      if not isinstance(body, dict) or body.get('T_obj_to_ee') is None:
        continue
      out[str(name)] = _profile_fields_from_dict(body)
    if out:
      return out
  if g.get('T_obj_to_ee') is not None:
    return {'default': _profile_fields_from_dict(g)}
  return {}


def pick_teach_profile(
    T_base_obj: np.ndarray,
    profiles: dict[str, dict],
) -> tuple[str, float]:
  """按 FP 当前朝向，选最接近的一份示教档案。"""
  if not profiles:
    raise ValueError('没有示教档案')
  if len(profiles) == 1:
    name = next(iter(profiles))
    prof = profiles[name]
    R_t = prof.get('teach_R_base_obj')
    if R_t is None:
      return name, 0.0
    _, _, can, _ = canonicalize_object_rotation(
        T_base_obj[:3, :3], R_t, T_base_obj, prof['T_obj_to_ee'],
        prof.get('teach_ee_xyz'))
    return name, can
  R_now = T_base_obj[:3, :3]
  best_name, best_deg = '', 999.0
  for name, prof in profiles.items():
    R_t = prof.get('teach_R_base_obj')
    if R_t is None:
      continue
    _, _, can, _ = canonicalize_object_rotation(
        R_now, R_t, T_base_obj, prof['T_obj_to_ee'], prof.get('teach_ee_xyz'))
    if can < best_deg:
      best_name, best_deg = name, can
  if not best_name:
    best_name = next(iter(profiles))
    best_deg = 0.0
  return best_name, best_deg


def resolve_fp_teach_profile(
    cfg: dict,
    T_base_obj: np.ndarray,
    *,
    profile_override: Optional[str] = None,
) -> dict:
  """解析本次抓取用哪份示教档案，返回 T_obj_to_ee 等。"""
  profiles = list_teach_profiles(cfg)
  if not profiles:
    raise RuntimeError('还没示教：python3 hands/grasp/teach_grasp_frame.py --pose lying')
  if profile_override:
    key = profile_override.strip()
    if key not in profiles:
      known = ', '.join(profiles)
      raise RuntimeError(f'示教档案 {key!r} 不存在，已有: {known}')
    name, match_deg = key, 0.0
    if profiles[key].get('teach_R_base_obj') is not None:
      _, _, match_deg, _ = canonicalize_object_rotation(
          T_base_obj[:3, :3], profiles[key]['teach_R_base_obj'],
          T_base_obj, profiles[key]['T_obj_to_ee'],
          profiles[key].get('teach_ee_xyz'))
  else:
    name, match_deg = pick_teach_profile(T_base_obj, profiles)
  prof = profiles[name]
  label = _TEACH_PROFILE_LABELS.get(name, name)
  return {
      'profile': name,
      'profile_label': label,
      'match_deg': float(match_deg),
      **prof,
  }


def _rotation_delta_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
  dR = R_a @ R_b.T
  return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))


def _object_symmetry_matrices_obj_frame(
    half_extents: Optional[list[float]] = None,
) -> list[np.ndarray]:
  """
  茶叶罐方盒在物体系下的离散对称（右乘 R_now @ S）。
  只含 180° 翻面；长方体不加 90° 换边（会把长轴当短轴，夹爪抓到尾部空气）。
  """
  mats: list[np.ndarray] = [np.eye(3)]
  for diag in ((-1., -1., 1.), (-1., 1., -1.), (1., -1., -1.)):
    mats.append(np.diag(diag))
  if half_extents is not None and len(half_extents) >= 2:
    hx, hy = abs(float(half_extents[0])), abs(float(half_extents[1]))
    if abs(hx - hy) / max(hx, hy, 1e-6) < 0.12:
      for ang in (0.5 * np.pi, -0.5 * np.pi):
        c, s = float(np.cos(ang)), float(np.sin(ang))
        mats.append(np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]))
  return mats


def _grasp_pos_from_R(
    R_obj: np.ndarray,
    T_base_obj: np.ndarray,
    T_obj_to_ee: np.ndarray,
) -> np.ndarray:
  Tb = np.asarray(T_base_obj, dtype=float).reshape(4, 4).copy()
  Tb[:3, :3] = np.asarray(R_obj, dtype=float).reshape(3, 3)
  T_oe = np.asarray(T_obj_to_ee, dtype=float).reshape(4, 4)
  return (Tb @ T_oe)[:3, 3]


def canonicalize_object_rotation(
    R_now: np.ndarray,
    R_teach: np.ndarray,
    T_base_obj: Optional[np.ndarray] = None,
    T_obj_to_ee: Optional[np.ndarray] = None,
    teach_ee_xyz: Optional[np.ndarray] = None,
    *,
    half_extents: Optional[list[float]] = None,
    symmetry_flip_min_raw_deg: float = 40.0,
) -> tuple[np.ndarray, float, float, float]:
  """
  FP 方盒对称：在等价朝向里选最合适的一个。
  用示教抓取点全 3D 距离打分（不只比 z），避免 180° 翻面后夹爪跑到物体另一头抓空。
  返回 (R_use, 原始角度差°, 校正后角度差°, 抓取点偏移mm)
  """
  R_now = np.asarray(R_now, dtype=float).reshape(3, 3)
  R_teach = np.asarray(R_teach, dtype=float).reshape(3, 3)
  raw = _rotation_delta_deg(R_now, R_teach)
  teach_xyz = (np.asarray(teach_ee_xyz, dtype=float).reshape(3)
               if teach_ee_xyz is not None else None)
  T_oe = (np.asarray(T_obj_to_ee, dtype=float).reshape(4, 4)
          if T_obj_to_ee is not None else None)
  use_pos = T_base_obj is not None and T_oe is not None and teach_xyz is not None

  def _score(R_c: np.ndarray) -> tuple[float, float, float]:
    ang = _rotation_delta_deg(R_c, R_teach)
    if not use_pos:
      return ang, ang, 0.0
    gpos = _grasp_pos_from_R(R_c, T_base_obj, T_oe)
    pos_err_mm = float(np.linalg.norm(gpos - teach_xyz) * 1000.0)
    # 位置差权重大：长条物体翻 180° 后姿态角可能仍接近，但抓取点会偏一整截
    return ang + 2.5 * pos_err_mm, ang, pos_err_mm

  _, raw_ang, raw_pos_mm = _score(R_now)
  if raw_ang < float(symmetry_flip_min_raw_deg) and raw_pos_mm < 40.0:
    return R_now, raw, raw_ang, 0.0

  sym_mats = _object_symmetry_matrices_obj_frame(half_extents)
  R_best = R_now
  best_total, best_ang, best_pos = _score(R_now)
  gpos_now = (_grasp_pos_from_R(R_now, T_base_obj, T_oe) if use_pos else None)
  for S in sym_mats[1:]:
    R_c = R_now @ S
    total, ang, pos_mm = _score(R_c)
    if total < best_total - 0.5:
      R_best, best_total, best_ang, best_pos = R_c, total, ang, pos_mm
  shift_mm = 0.0
  if use_pos and gpos_now is not None:
    gpos_best = _grasp_pos_from_R(R_best, T_base_obj, T_oe)
    shift_mm = float(np.linalg.norm(gpos_best - gpos_now) * 1000.0)
  return R_best, raw, best_ang, shift_mm


def apply_symmetry_to_T_base_obj(
    T_base_obj: np.ndarray,
    R_teach: Optional[np.ndarray],
    T_obj_to_ee: Optional[np.ndarray] = None,
    teach_ee_xyz: Optional[np.ndarray] = None,
    *,
    half_extents: Optional[list[float]] = None,
    symmetry_flip_min_raw_deg: float = 40.0,
) -> tuple[np.ndarray, Optional[dict]]:
  """若给了示教朝向，对 T_base_obj 旋转做对称规范化。"""
  if R_teach is None:
    return T_base_obj, None
  R_can, raw_d, can_d, shift_mm = canonicalize_object_rotation(
      T_base_obj[:3, :3], R_teach, T_base_obj, T_obj_to_ee, teach_ee_xyz,
      half_extents=half_extents,
      symmetry_flip_min_raw_deg=symmetry_flip_min_raw_deg)
  changed = not np.allclose(R_can, T_base_obj[:3, :3], atol=1e-6)
  info = {'raw_deg': raw_d, 'canon_deg': can_d, 'applied': changed,
          'grasp_shift_mm': shift_mm}
  if not changed:
    return T_base_obj, info
  T_out = T_base_obj.copy()
  T_out[:3, :3] = R_can
  return T_out, info


def _pose6_close(a: list[float], b: list[float],
                 pos_mm: float = 5.0, rot_deg: float = 3.0) -> bool:
  a = np.asarray(a, dtype=float).reshape(6)
  b = np.asarray(b, dtype=float).reshape(6)
  if np.max(np.abs((a[:3] - b[:3]) * 1000.0)) > pos_mm:
    return False
  return np.max(np.abs(a[3:] - b[3:])) <= np.radians(rot_deg)


def _matrix_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-3) -> bool:
  return bool(np.max(np.abs(np.asarray(a, float).reshape(4, 4)
                            - np.asarray(b, float).reshape(4, 4))) <= tol)


def _handeye_mismatch_mm(taught: np.ndarray, current: np.ndarray) -> float:
  t = np.asarray(taught, float).reshape(4, 4)
  c = np.asarray(current, float).reshape(4, 4)
  return float(np.max(np.abs(t - c) * 1000.0))


def fp_teach_stale_reason(
    cfg: dict,
    T_base_cam: Optional[np.ndarray] = None,
    *,
    cfg_path: Optional[str] = None,
) -> Optional[str]:
  """
  检查 T_obj_to_ee 是否跟当前手眼、位置1 对得上。
  相机动过或位置1变了却没重示教，会返回中文原因；没问题返回 None。
  """
  g = cfg.get('arm', {}).get('fp_grasp', {})
  if g.get('orientation_mode', 'fp') != 'fp':
    return None
  if g.get('grasp_orientation') in ('nearest_face', 'top_face'):
    return None
  top = g.get('T_obj_to_ee')
  lying = (g.get('teach_profiles') or {}).get('lying', {}).get('T_obj_to_ee')
  if top is not None and lying is not None and not _matrix_close(top, lying):
    return ('顶层 T_obj_to_ee 与 teach_profiles.lying 不一致（示教后未同步）。'
            '请重跑: python3 hands/grasp/teach_grasp_frame.py --pose lying')
  if not list_teach_profiles(cfg):
    return '还没示教 T_obj_to_ee：运行 teach_grasp_frame.py --pose lying（或 standing）'
  cam_cfg = cfg.get('camera', {}) or {}
  mode = cam_cfg.get('hand_eye_mode', HAND_EYE_EYE_IN_HAND)
  if mode == HAND_EYE_EYE_IN_HAND:
    taught = g.get('teach_T_ee_cam')
    current = cam_cfg.get('camera_to_end_effector')
    if taught is None or current is None:
      return '还没示教 T_obj_to_ee：运行 teach_grasp_frame.py --pose lying（或 standing）'
    if not _matrix_close(np.asarray(current, float), np.asarray(taught, float)):
      mm = _handeye_mismatch_mm(taught, current)
      return (
          f'手眼标定已更新，但抓取示教还是旧的（相差约 {mm:.0f}mm）。'
          '请重跑: python3 hands/grasp/teach_grasp_frame.py --pose lying')
    return None
  taught_cam = g.get('teach_T_base_cam')
  if taught_cam is None:
    return '还没示教 T_obj_to_ee：运行 teach_grasp_frame.py --pose lying（或 standing）'
  if T_base_cam is None:
    T_base_cam = load_T_base_cam(cfg_path)
  if not _matrix_close(T_base_cam, np.asarray(taught_cam, float)):
    return '手眼标定已变（固定相机动过）：请重跑 teach_grasp_frame.py'
  return None


def validate_fp_grasp_plan(
    hover6: list[float],
    pre6: list[float],
    grasp6: list[float],
    lift6: list[float],
    init_pose6: list[float],
    obj_xyz: np.ndarray,
    R_init: np.ndarray,
    R_fp: np.ndarray,
    cfg: Optional[dict] = None,
    T_base_obj: Optional[np.ndarray] = None,
    *,
    max_object_rot_deg: float = 100.0,
    max_descent_m: float = 0.50,
    min_grasp_z_m: float = -0.35,
    half_extents: Optional[Sequence[float]] = None,
) -> None:
  """真抓前轻量检查（只打日志，不拦流程）。"""
  grasp_z = float(grasp6[2])
  pre_pos = np.asarray(pre6[:3], dtype=float)
  grasp_pos = np.asarray(grasp6[:3], dtype=float)
  pre_dist = float(np.linalg.norm(pre_pos - grasp_pos))
  if pre_dist < 0.02:
    print('  【提示】预抓取与抓取点很近，检查 approach_m')
  descent = float(pre_pos[2] - grasp_pos[2])
  if descent > max_descent_m:
    print(f'  【提示】预抓取→抓取竖直落差 {descent*1000:.0f}mm 偏大')
  if T_base_obj is not None and half_extents is not None:
    _, zmax = _object_bbox_z_range(T_base_obj, half_extents)
    pad_m = 0.20
    if cfg is not None:
      g = cfg.get('arm', {}).get('fp_grasp', {})
      pad_m = load_gripper_tcp_to_pad_m(cfg) + float(
          g.get('grasp_tcp_above_top_m', 0))
    if grasp_z < zmax + pad_m - 0.004:
      surf_z = float(zmax)
      if T_base_obj is not None:
        try:
          face = pick_top_grasp_face(T_base_obj, half_extents)
          n = np.asarray(face['outward_normal'], dtype=float).reshape(3)
          n = n / max(float(np.linalg.norm(n)), 1e-9)
          oc = T_base_obj[:3, 3]
          top_h = _fp_face_top_height_m(T_base_obj, half_extents, n)
          if abs(float(n[2])) > 1e-6:
            surf_z = (top_h - n[0] * float(oc[0]) - n[1] * float(oc[1])) / float(n[2])
        except Exception:
          pass
      if grasp_z < surf_z + pad_m - 0.004:
        print(f'  【警告】关爪法兰 z={grasp_z*1000:.0f}mm 低于物心顶面+外退 '
              f'{(surf_z+pad_m)*1000:.0f}mm，可能贴面太近')


def build_fp_staged_pose6(
    ob_in_cam,
    T_base_cam: np.ndarray,
    T_obj_to_ee: np.ndarray,
    approach_m: float,
    lift_m: float,
    grasp_above_m: float,
    grasp_z_offset_m: float,
    init_pose6: list[float],
    retreat_axis: str = 'pos_z',
    lift_axis: str = 'pos_z',
    R_init: Optional[np.ndarray] = None,
    R_teach_obj: Optional[np.ndarray] = None,
    teach_ee_xyz: Optional[np.ndarray] = None,
    teach_obj_xyz: Optional[np.ndarray] = None,
    R_teach_ee: Optional[np.ndarray] = None,
    grasp_orientation: str = 'teach',
    object_half_extents_m: Optional[list[float]] = None,
    symmetry_flip_min_raw_deg: float = 40.0,
    grasp_face_inset_m: float = 0.008,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[dict]]:
  """
  FP 6D 抓取路点。
  grasp_orientation=teach：位置=物体中心+dz，姿态=该档案示教末端（始终向下，不跟 FP 转侧面）。
  grasp_orientation=fp：T_grasp = T_base_obj @ T_obj_to_ee（完整跟转）。
  grasp_orientation=obj_z_parallel：法兰进给轴（工具 +Z）∥ FP 蓝轴（物体系 +Z）；
    躺放（蓝轴近水平）沿蓝轴从侧面进爪，不再用正上方示教 xyz。
  """
  T_base_obj = ob_in_cam_to_arm_T(ob_in_cam, T_base_cam)
  T_oe = np.asarray(T_obj_to_ee, dtype=float).reshape(4, 4)
  sym_info: Optional[dict] = None

  if grasp_orientation == 'obj_z_parallel':
    sym_info = {'mode': 'obj_z_parallel', 'applied': False}
    R_obj = T_base_obj[:3, :3]
    R_fp = R_obj @ T_oe[:3, :3]
    z_blue = R_obj[:, 2]
    T_grasp = np.eye(4, dtype=float)
    T_grasp[:3, :3] = _rotation_tool_z_parallel(z_blue, R_fp)
    half = object_half_extents_m or [0.0335, 0.065, 0.0265]
    if _obj_z_blue_is_mostly_horizontal(R_obj):
      # 躺放：物体在桌上哪都行，xyz 跟当前 FP+示教偏移；只改姿态（法兰 Z∥蓝轴）
      T_grasp[:3, 3] = (T_base_obj @ T_oe)[:3, 3]
      sym_info['approach'] = 'z_parallel_fp_xyz'
    else:
      T_grasp[:3, 3] = (T_base_obj @ T_oe)[:3, 3]
      sym_info['approach'] = 'teach_tcp'
    if abs(float(grasp_z_offset_m)) > 1e-9:
      T_grasp = _offset_along_ee_axis(T_grasp, grasp_z_offset_m, 'neg_z')
  elif (grasp_orientation == 'teach' and R_teach_ee is not None
      and teach_ee_xyz is not None and teach_obj_xyz is not None):
    sym_info = {'mode': 'teach_lock', 'applied': False}
    center = T_base_obj[:3, 3].copy()
    dz = float(np.asarray(teach_ee_xyz, dtype=float).reshape(3)[2]
               - np.asarray(teach_obj_xyz, dtype=float).reshape(3)[2])
    T_grasp = np.eye(4, dtype=float)
    T_grasp[:3, :3] = np.asarray(R_teach_ee, dtype=float).reshape(3, 3)
    T_grasp[2, 3] = float(center[2]) + dz + float(grasp_z_offset_m)
    T_grasp[0, 3] = float(center[0])
    T_grasp[1, 3] = float(center[1])
  else:
    T_base_obj, sym_info = apply_symmetry_to_T_base_obj(
        T_base_obj, R_teach_obj, T_oe, teach_ee_xyz,
        half_extents=object_half_extents_m,
        symmetry_flip_min_raw_deg=symmetry_flip_min_raw_deg)
    T_grasp = T_base_obj @ T_oe
    if abs(float(grasp_z_offset_m)) > 1e-9:
      T_grasp = _offset_along_ee_axis(T_grasp, grasp_z_offset_m, 'neg_z')
  T_pre = _offset_along_ee_axis(T_grasp, approach_m, retreat_axis)
  T_lift = _offset_along_ee_axis(T_grasp, lift_m, lift_axis)
  init_z = float(init_pose6[2])
  obj_c = T_base_obj[:3, 3]
  T_hover = np.eye(4, dtype=float)
  if grasp_orientation == 'obj_z_parallel':
    # 法兰 Z 已跟 FP 蓝轴对齐，直接进预抓取，不做「物体 xy + 位置1 高度」的视角位
    T_hover = T_pre.copy()
  elif grasp_orientation == 'fp' and R_init is not None:
    # 与眼在手外一致：先位置1姿态平移到物体上方，pre 再跟 FP 转腕
    T_hover[:3, :3] = np.asarray(R_init, dtype=float).reshape(3, 3)
    T_hover[:3, 3] = [float(obj_c[0]), float(obj_c[1]), init_z]
  else:
    T_hover[:3, :3] = T_grasp[:3, :3]
    T_hover[:3, 3] = [float(obj_c[0]), float(obj_c[1]), init_z]
  return T_hover, T_pre, T_grasp, T_lift, T_base_obj, sym_info


def _wrist_delta_deg(robot, pose_a: list[float], pose_b: list[float]) -> float:
  Ra = pose6_to_matrix(robot, pose_a)[:3, :3]
  Rb = pose6_to_matrix(robot, pose_b)[:3, :3]
  return _rotation_delta_deg(Ra, Rb)


def _align_pose6_at_hover(hover6: list[float], target6: list[float]) -> list[float]:
  """物体正上方原地转腕：xyz 保持 hover，rpy 跟 FP 目标位姿。"""
  return [float(hover6[i]) for i in range(3)] + [float(target6[i]) for i in range(3, 6)]


def _align_pose6_above_grasp(hover6: list[float], grasp6: list[float]) -> list[float]:
  """关爪点正上方转腕：xy 对准关爪点，z 保持观察高度，再纯竖直下去。"""
  return [float(grasp6[0]), float(grasp6[1]), float(hover6[2]),
          float(grasp6[3]), float(grasp6[4]), float(grasp6[5])]


def build_init_orient_grasp_pose6(
    obj_xyz: np.ndarray,
    init_pose6: list[float],
    approach_m: float,
    grasp_above_m: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
  """
  FP 只提供物体中心 xyz；姿态用位置1 init（竖直夹茶叶罐可解）。
  返回 hover(保持init高度)、pre、grasp、lift(pose6)。
  """
  rpy = list(init_pose6[3:6])
  hover_z = float(init_pose6[2])
  ox, oy, oz = float(obj_xyz[0]), float(obj_xyz[1]), float(obj_xyz[2])
  hover = [ox, oy, hover_z, *rpy]
  pre = [ox, oy, oz + approach_m, *rpy]
  grasp = [ox, oy, oz + grasp_above_m, *rpy]
  lift = pre.copy()
  return hover, pre, grasp, lift


def _offset_along_ee_axis(T_ee: np.ndarray, dist_m: float, axis: str) -> np.ndarray:
  """沿末端坐标轴平移 dist_m（米）。axis: neg_x/pos_x/neg_y/pos_y/neg_z/pos_z"""
  R = T_ee[:3, :3]
  dirs = {
      'neg_x': -R[:, 0], 'pos_x': R[:, 0],
      'neg_y': -R[:, 1], 'pos_y': R[:, 1],
      'neg_z': -R[:, 2], 'pos_z': R[:, 2],
  }
  if axis not in dirs:
    raise ValueError(f'未知 approach_axis: {axis}')
  out = T_ee.copy()
  out[:3, 3] = T_ee[:3, 3] + dirs[axis] * dist_m
  return out


_FACE_AXIS_LABELS = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')


def _enumerate_box_faces(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
) -> list[dict]:
  """FP 方盒六个面：面心、外法向、面积（物体系半轴 → 基座系）。"""
  R = T_base_obj[:3, :3]
  oc = T_base_obj[:3, 3]
  h = np.asarray(half_extents, dtype=float).reshape(3)
  faces = []
  for axis in range(3):
    for sign in (-1.0, 1.0):
      n_obj = np.zeros(3, dtype=float)
      n_obj[axis] = sign
      n = R @ n_obj
      n_norm = float(np.linalg.norm(n))
      if n_norm < 1e-9:
        continue
      n = n / n_norm
      fc = oc + R @ (np.eye(3)[axis] * sign * h[axis])
      area = float(4.0 * h[(axis + 1) % 3] * h[(axis + 2) % 3])
      idx = axis * 2 + (0 if sign > 0 else 1)
      faces.append({
          'face': _FACE_AXIS_LABELS[idx],
          'face_center': fc,
          'outward_normal': n,
          'area_m2': area,
          'axis': axis,
          'sign': sign,
          'up_dot': float(np.dot(n, np.array([0.0, 0.0, 1.0]))),
      })
  return faces


def _point_base_to_cam_xyz(p_base: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
  T = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
  p = np.asarray(p_base, dtype=float).reshape(3)
  ph = np.append(p, 1.0)
  return (np.linalg.inv(T) @ ph)[:3]


def teach_orientation_match_deg(
    T_base_obj: np.ndarray,
    cfg: dict,
) -> float:
  """FP 当前朝向与示教躺放档案的角度差（度）；小说明是视角位宽×高那类摆法。"""
  g = cfg.get('arm', {}).get('fp_grasp', {})
  R_t = g.get('teach_R_base_obj')
  if R_t is None:
    return 999.0
  _, _, can, _ = canonicalize_object_rotation(
      T_base_obj[:3, :3], np.asarray(R_t, float).reshape(3, 3),
      symmetry_flip_min_raw_deg=999.0)
  return float(can)


def pick_camera_facing_grasp_face(
    T_base_obj: np.ndarray,
    T_base_cam: np.ndarray,
    half_extents: Sequence[float],
    *,
    min_face_cam_dot: float = 0.45,
    depth_margin_m: float = 0.02,
    large_face_ratio: float = 0.85,
    reject_bottom_up_dot: float = -0.45,
) -> dict:
  """
  选抓取面（眼在手上 = 画面里朝向相机、离相机近、尽量大面）：
    1. 外法向必须朝向相机（cam_dot >= 阈值）
    2. 面心深度不得比物体中心更远（避免抓背面）
    3. 不要底面朝下的面
    4. 优先茶叶罐宽×高大类面（面积 >= 最大面 × ratio）
    5. 同分：更朝相机、更大、更近
  """
  T_bc = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
  cam = T_bc[:3, 3]
  obj_cam = _point_base_to_cam_xyz(T_base_obj[:3, 3], T_bc)
  obj_depth = float(obj_cam[2])

  cands = []
  for f in _enumerate_box_faces(T_base_obj, half_extents):
    fc = f['face_center']
    n = f['outward_normal']
    to_cam = cam - fc
    to_cam_norm = float(np.linalg.norm(to_cam))
    if to_cam_norm < 1e-6:
      continue
    to_cam = to_cam / to_cam_norm
    cam_dot = float(np.dot(n, to_cam))
    p_cam = _point_base_to_cam_xyz(fc, T_bc)
    face_depth = float(p_cam[2])
    cands.append({
        **f,
        'cam_dot': cam_dot,
        'cam_depth_m': face_depth,
        'depth_vs_obj_mm': (face_depth - obj_depth) * 1000.0,
    })
  if not cands:
    raise RuntimeError('无法枚举物体抓取面')

  max_area = max(c['area_m2'] for c in cands)

  def _pool(relax_large: bool) -> list[dict]:
    p = [c for c in cands if c['cam_dot'] >= float(min_face_cam_dot)]
    if not p:
      p = [c for c in cands if c['cam_dot'] > 0.0]
    if not p:
      p = list(cands)
    p = [c for c in p if float(c['up_dot']) >= float(reject_bottom_up_dot)]
    if not p:
      p = [c for c in cands if float(c['up_dot']) >= float(reject_bottom_up_dot)] or list(cands)
    p = [c for c in p if c['cam_depth_m'] <= obj_depth + float(depth_margin_m)]
    if not p:
      p = [c for c in cands if c['cam_depth_m'] <= obj_depth + float(depth_margin_m) + 0.03]
      if not p:
        p = list(cands)
    if not relax_large:
      large_min = max_area * float(large_face_ratio)
      lp = [c for c in p if c['area_m2'] >= large_min - 1e-9]
      if lp:
        p = lp
    return p

  pool = _pool(relax_large=False)
  if not pool:
    pool = _pool(relax_large=True)

  def _rank(c: dict) -> tuple:
    return (c['cam_dot'], c['area_m2'], -c['cam_depth_m'])

  pick = dict(max(pool, key=_rank))
  pick['pick_mode'] = 'camera'
  pick['obj_cam_depth_m'] = obj_depth
  return pick


def pick_nearest_face_to_ee(
    T_base_obj: np.ndarray,
    ee_xyz: np.ndarray,
    half_extents: Sequence[float],
) -> dict:
  """
  在物体六个面里，选离当前末端最近、且末端在外侧的那一面。
  返回面中心、外法向、物体系轴标签、平面距离等。
  """
  R = T_base_obj[:3, :3]
  oc = T_base_obj[:3, 3]
  ee = np.asarray(ee_xyz, dtype=float).reshape(3)
  h = np.asarray(half_extents, dtype=float).reshape(3)
  cands = []
  for axis in range(3):
    for sign in (-1.0, 1.0):
      n_obj = np.zeros(3, dtype=float)
      n_obj[axis] = sign
      n = R @ n_obj
      n_norm = float(np.linalg.norm(n))
      if n_norm < 1e-9:
        continue
      n = n / n_norm
      fc = oc + R @ (np.eye(3)[axis] * sign * h[axis])
      d_plane = float(np.dot(ee - fc, n))
      idx = axis * 2 + (0 if sign > 0 else 1)
      cands.append({
          'face': _FACE_AXIS_LABELS[idx],
          'face_center': fc,
          'outward_normal': n,
          'plane_dist_m': d_plane,
          'axis': axis,
          'sign': sign,
      })
  outside = [c for c in cands if c['plane_dist_m'] > 0.01]
  if outside:
    pick = min(outside, key=lambda c: c['plane_dist_m'])
  else:
    pick = max(cands, key=lambda c: c['plane_dist_m'])
  pick = dict(pick)
  # 底面朝下的面不抓（茶叶罐在桌上）
  if float(pick['outward_normal'][2]) < -0.45:
    pool = outside if outside else cands
    alt = [c for c in pool if float(c['outward_normal'][2]) >= -0.45]
    if alt:
      pick = dict(min(alt, key=lambda c: c['plane_dist_m']))
  pick['ee_dist_m'] = float(np.linalg.norm(ee - pick['face_center']))
  return pick


def pick_top_grasp_face(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
    ee_xyz: Optional[np.ndarray] = None,
    *,
    min_up_dot: float = 0.45,
) -> dict:
  """
  从 FP 方盒六个面里选最朝上的那一面（世界 +Z）。
  贴桌底面 up_dot<0 直接排除，避免去抓脚面。
  """
  cands = _enumerate_box_faces(T_base_obj, half_extents)
  upward = [c for c in cands if float(c['up_dot']) >= float(min_up_dot)]
  if not upward:
    upward = [c for c in cands if float(c['up_dot']) > 0.05]
  if not upward:
    raise RuntimeError(
        'FP 方盒没有朝上的面可抓（可能跟丢或朝向异常），先等绿框稳定')
  pool = upward

  def _rank(c: dict) -> tuple:
    ee_dist = 0.0
    if ee_xyz is not None:
      ee_dist = float(np.linalg.norm(np.asarray(ee_xyz, float).reshape(3) - c['face_center']))
    return (-c['up_dot'], -c['area_m2'], ee_dist)

  pick = dict(min(pool, key=_rank))
  pick['pick_mode'] = 'top'
  if ee_xyz is not None:
    pick['ee_dist_m'] = float(np.linalg.norm(
        np.asarray(ee_xyz, float).reshape(3) - pick['face_center']))
  return pick


def resolve_grasp_face(
    T_base_obj: np.ndarray,
    T_base_cam: np.ndarray,
    half_extents: Sequence[float],
    *,
    face_pick: str = 'camera',
    ee_xyz: Optional[np.ndarray] = None,
    face_pick_min_cam_dot: float = 0.45,
    face_pick_depth_margin_m: float = 0.02,
    face_pick_large_face_ratio: float = 0.85,
    hand_eye_mode: str = HAND_EYE_EYE_IN_HAND,
    lying_top_min_up_dot: float = 0.55,
) -> dict:
  """
  face_pick:
    auto   — 眼在手上：躺放优先朝上面，否则画面朝向相机的一面；眼在手外：最朝上一面
    camera — 画面里朝向相机、离相机近、优先大面
    top    — 最朝上一面（俯视抓）
    nearest — 离末端最近
  """
  mode = str(face_pick).strip().lower()
  if mode == 'auto':
    if hand_eye_mode == HAND_EYE_EYE_IN_HAND:
      top = pick_top_grasp_face(T_base_obj, half_extents, ee_xyz)
      all_faces = _enumerate_box_faces(T_base_obj, half_extents)
      max_area = max(f['area_m2'] for f in all_faces) if all_faces else 0.0
      large_min = max_area * float(face_pick_large_face_ratio)
      # 平躺：朝上的面是宽×长那种大顶面 → 从上往下；竖立/侧放：顶面小，改抓画面朝向相机的一面
      if (float(top['up_dot']) >= float(lying_top_min_up_dot)
          and float(top['area_m2']) >= large_min - 1e-9):
        top['pick_mode'] = 'auto_top_flat'
        return top
      face = pick_camera_facing_grasp_face(
          T_base_obj, T_base_cam, half_extents,
          min_face_cam_dot=face_pick_min_cam_dot,
          depth_margin_m=face_pick_depth_margin_m,
          large_face_ratio=face_pick_large_face_ratio)
      face['pick_mode'] = 'auto_camera'
      return face
    face = pick_top_grasp_face(T_base_obj, half_extents, ee_xyz)
    face['pick_mode'] = 'auto_top_eye_to_hand'
    return face
  if mode == 'top':
    return pick_top_grasp_face(T_base_obj, half_extents, ee_xyz)
  if mode == 'nearest':
    if ee_xyz is None:
      raise ValueError('face_pick=nearest 需要 ee_xyz')
    face = pick_nearest_face_to_ee(T_base_obj, ee_xyz, half_extents)
    face['pick_mode'] = 'nearest'
    return face
  return pick_camera_facing_grasp_face(
      T_base_obj, T_base_cam, half_extents,
      min_face_cam_dot=face_pick_min_cam_dot,
      depth_margin_m=face_pick_depth_margin_m,
      large_face_ratio=face_pick_large_face_ratio)


def _flange_tcp_parallel_to_face(
    face_center: np.ndarray,
    outward_normal: np.ndarray,
    flange_face_gap_m: float,
) -> np.ndarray:
  """
  法兰与抓取面平行时，TCP 在面外侧沿外法向 +n 退 flange_face_gap_m（默认 6cm）。
  """
  n = np.asarray(outward_normal, dtype=float).reshape(3)
  n = n / max(float(np.linalg.norm(n)), 1e-9)
  return np.asarray(face_center, dtype=float).reshape(3) + n * float(flange_face_gap_m)


def _ee_rotation_for_top_face(
    T_base_obj: np.ndarray,
    outward_normal: np.ndarray,
    R_prefer: Optional[np.ndarray] = None,
    half_extents: Optional[Sequence[float]] = None,
    approach_axis: str = 'z',
) -> np.ndarray:
  """
  与瑞尔曼 top_face 同规则：开合轴对齐顶面**长边** → 手指夹**短边**。
  瑞尔曼：+Z 进给、+X 开合（X∥长边）。
  高擎 tool_link：+X 进给、+Z 开合（Z∥长边；此前误用 Y 开合导致抓长边）。
  """
  z_in = -np.asarray(outward_normal, dtype=float).reshape(3)
  z_in = z_in / max(float(np.linalg.norm(z_in)), 1e-9)
  R_obj = np.asarray(T_base_obj[:3, :3], dtype=float).reshape(3, 3)
  h = np.asarray(half_extents if half_extents is not None else [1.0, 1.0, 1.0],
                 dtype=float).reshape(3)
  in_plane: list[tuple[float, np.ndarray]] = []
  for i in range(3):
    v = R_obj[:, i] - np.dot(R_obj[:, i], z_in) * z_in
    ln = float(np.linalg.norm(v))
    if ln < 1e-6:
      continue
    eff = float(h[i]) * ln
    in_plane.append((eff, v / ln))
  if not in_plane:
    return _ee_rotation_for_face_approach(outward_normal, R_prefer)
  in_plane.sort(key=lambda t: t[0])
  long_edge = in_plane[-1][1]  # 开合应对齐的方向
  y_rm = np.cross(z_in, long_edge)
  y_rm = y_rm / max(float(np.linalg.norm(y_rm)), 1e-9)
  # 瑞尔曼：列 = [开合X=长边, Y, 进给Z]
  R_rm = np.column_stack([long_edge, y_rm, z_in])
  R_rm_flip = R_rm @ np.diag([-1.0, -1.0, 1.0])
  if R_prefer is not None:
    d0 = _rotation_delta_deg(R_rm, R_prefer)
    d1 = _rotation_delta_deg(R_rm_flip, R_prefer)
    R_rm = R_rm if d0 <= d1 else R_rm_flip

  ax = str(approach_axis).strip().lower()
  if ax == 'x':
    # 高擎：X=进给(原Z)，Z=开合=长边(原X)，Y=右手系
    x_axis = R_rm[:, 2]
    z_axis = R_rm[:, 0]
    y_axis = np.cross(z_axis, x_axis)
    y_n = float(np.linalg.norm(y_axis))
    if y_n < 1e-6:
      return _ee_rotation_for_face_approach(outward_normal, R_prefer)
    y_axis = y_axis / y_n
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-9)
    R0 = np.column_stack([x_axis, y_axis, z_axis])
    R_flip = np.column_stack([x_axis, -y_axis, -z_axis])
    if R_prefer is None:
      return R0
    d0 = _rotation_delta_deg(R0, R_prefer)
    d1 = _rotation_delta_deg(R_flip, R_prefer)
    return R0 if d0 <= d1 else R_flip
  if R_prefer is None:
    return R_rm
  return R_rm


def _obb_corners_base(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
) -> np.ndarray:
  """FP 绿框 8 角点（基座系，米）。"""
  R = T_base_obj[:3, :3]
  c = T_base_obj[:3, 3]
  h = np.asarray(half_extents, dtype=float).reshape(3)
  pts = []
  for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
      for sz in (-1.0, 1.0):
        local = np.array([sx * h[0], sy * h[1], sz * h[2]], dtype=float)
        pts.append(c + R @ local)
  return np.asarray(pts, dtype=float)


def fp_box_extents_m(half_extents: Sequence[float]) -> list[float]:
  """物体系长宽高（米），与绿框一致。"""
  h = np.asarray(half_extents, dtype=float).reshape(3)
  return [float(2.0 * h[i]) for i in range(3)]


def fp_box_extents_base_m(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
) -> list[float]:
  """绿框在基座系三轴跨度（米），用于日志。"""
  pts = _obb_corners_base(T_base_obj, half_extents)
  spans = pts.max(axis=0) - pts.min(axis=0)
  return [float(x) for x in spans]


def sample_fp_box_top_base_z(
    ob_in_cam: np.ndarray,
    half_extents: Sequence[float],
    depth: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
) -> Optional[float]:
  """
  用冻结帧深度 + 绿框投影，估顶面在基座系的 z（米）。
  手眼绝对高度漂了时，用这个顶住「一直往下蹭地」。
  """
  T = np.asarray(ob_in_cam, dtype=float).reshape(4, 4)
  h = np.asarray(half_extents, dtype=float).reshape(3)
  corners_obj = []
  for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
      for sz in (-1.0, 1.0):
        corners_obj.append([sx * h[0], sy * h[1], sz * h[2]])
  corners_obj = np.asarray(corners_obj, dtype=float).T
  corners_cam = (T[:3, :3] @ corners_obj + T[:3, 3:4]).T
  fx, fy = float(K[0, 0]), float(K[1, 1])
  cx, cy = float(K[0, 2]), float(K[1, 2])
  H, W = depth.shape[:2]
  uvs = []
  for p in corners_cam:
    if float(p[2]) < 0.04:
      continue
    uvs.append([(float(p[0]) / float(p[2])) * fx + cx,
                (float(p[1]) / float(p[2])) * fy + cy])
  if not uvs:
    return None
  uvs = np.asarray(uvs, dtype=float)
  x1 = max(0, int(np.floor(uvs[:, 0].min())))
  y1 = max(0, int(np.floor(uvs[:, 1].min())))
  x2 = min(W, int(np.ceil(uvs[:, 0].max())) + 1)
  y2 = min(H, int(np.ceil(uvs[:, 1].max())) + 1)
  if x2 - x1 < 4 or y2 - y1 < 4:
    return None
  y_cut = y1 + max(2, int(0.45 * (y2 - y1)))
  patch = np.asarray(depth[y1:y_cut, x1:x2], dtype=float)
  valid = patch[(patch > 0.04) & (patch < 3.0)]
  if valid.size < 6:
    return None
  z_med = float(np.median(valid))
  u_mid = 0.5 * (x1 + x2)
  v_mid = 0.5 * (y1 + y_cut)
  p_cam = np.array([
      (u_mid - cx) * z_med / fx,
      (v_mid - cy) * z_med / fy,
      z_med,
      1.0,
  ], dtype=float)
  T_bc = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
  p_base = T_bc @ p_cam
  return float(p_base[2])


def sample_depth_surface_at_base_point(
    p_base: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    *,
    patch: int = 5,
) -> Optional[np.ndarray]:
  """
  把基座系一点投到深度图，取实测表面三维点（基座系）。
  躺放/竖放都适用，比只用一个 z 标量准。
  """
  T_bc = np.asarray(T_base_cam, dtype=float).reshape(4, 4)
  p = np.asarray(p_base, dtype=float).reshape(3)
  ph = np.append(p, 1.0)
  p_cam = np.linalg.inv(T_bc) @ ph
  if float(p_cam[2]) < 0.04:
    return None
  fx, fy = float(K[0, 0]), float(K[1, 1])
  cx, cy = float(K[0, 2]), float(K[1, 2])
  u = fx * float(p_cam[0]) / float(p_cam[2]) + cx
  v = fy * float(p_cam[1]) / float(p_cam[2]) + cy
  H, W = depth.shape[:2]
  ui, vi = int(round(u)), int(round(v))
  r = max(1, int(patch))
  x1, x2 = max(0, ui - r), min(W, ui + r + 1)
  y1, y2 = max(0, vi - r), min(H, vi + r + 1)
  patch_d = np.asarray(depth[y1:y2, x1:x2], dtype=float)
  valid = patch_d[(patch_d > 0.04) & (patch_d < 3.0)]
  if valid.size < 3:
    return None
  z_m = float(np.median(valid))
  p_cam_m = np.array([
      (u - cx) * z_m / fx,
      (v - cy) * z_m / fy,
      z_m,
      1.0,
  ], dtype=float)
  p_out = T_bc @ p_cam_m
  return p_out[:3].copy()


def _fp_face_top_height_m(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
    outward_normal: np.ndarray,
) -> float:
  """绿框该抓取面沿外法向的最外缘高度（米）；竖放时比单点深度稳。"""
  corners = _obb_corners_base(T_base_obj, half_extents)
  n = np.asarray(outward_normal, dtype=float).reshape(3)
  n = n / max(float(np.linalg.norm(n)), 1e-9)
  return float(np.max(corners @ n))


def _point_on_face_plane_at_height(
    face_center: np.ndarray,
    outward_normal: np.ndarray,
    height_along_n: float,
) -> np.ndarray:
  """把面心沿外法向推到指定高度（米）。"""
  fc = np.asarray(face_center, dtype=float).reshape(3)
  n = np.asarray(outward_normal, dtype=float).reshape(3)
  n = n / max(float(np.linalg.norm(n)), 1e-9)
  return fc + n * (float(height_along_n) - float(np.dot(fc, n)))


def _green_box_zmax_world(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
) -> float:
  """绿框 8 角点世界 Z 最高值（米），与画面绿框一致，不做二次估计。"""
  return float(_obb_corners_base(T_base_obj, half_extents)[:, 2].max())


def _gripper_extra_drop_world_z(R_ee: np.ndarray, cfg: dict) -> float:
  """
  开爪时指尖相对 TCP 原点的额外下探（米）。
  瑞尔曼：开合轴为工具 X → 用 |R[2,0]|；
  高擎 approach_axis=x：开合轴为工具 Z → 用 |R[2,2]|。
  """
  g = cfg.get('gripper', {})
  half_stroke = float(g.get('stroke_mm', g.get('specs', {}).get('stroke_mm', 90))) * 0.5 / 1000.0
  # 高擎内置夹爪开口很小，未配 stroke_mm 时不要按 90mm 估
  if 'stroke_mm' not in g and 'specs' not in g:
    half_stroke = float(g.get('finger_half_stroke_m', 0.02))
  R = np.asarray(R_ee, dtype=float).reshape(3, 3)
  ax = str(cfg.get('arm', {}).get('fp_grasp', {}).get('approach_axis', 'z')).strip().lower()
  col = 2 if ax == 'x' else 0  # 高擎 Z 开合；瑞尔曼 X 开合
  return abs(float(R[2, col])) * half_stroke


def _green_box_top_point_world(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
) -> tuple[np.ndarray, float]:
  """
  兼容旧调用：返回 (物心 xyz 且 z=顶面, zmax)。
  xy 必须用物心/顶面心，不能用「最高角点均值」——稍有俯仰会偏到物体后侧。
  """
  oc = np.asarray(T_base_obj[:3, 3], dtype=float).reshape(3)
  zmax = _green_box_zmax_world(T_base_obj, half_extents)
  pt = np.array([float(oc[0]), float(oc[1]), zmax], dtype=float)
  return pt, zmax


def _grasp_surface_from_fp_box(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
    face_center: np.ndarray,
    outward_normal: np.ndarray,
) -> np.ndarray:
  """关爪参考面：只用绿框该抓取面最外缘，不采深度（顶面不必是完整平面）。"""
  n = np.asarray(outward_normal, dtype=float).reshape(3)
  n = n / max(float(np.linalg.norm(n)), 1e-9)
  fc = np.asarray(face_center, dtype=float).reshape(3)
  top_h = _fp_face_top_height_m(T_base_obj, half_extents, n)
  return _point_on_face_plane_at_height(fc, n, top_h)


def _top_surface_point_above_center(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
    outward_normal: np.ndarray,
) -> np.ndarray:
  """
  物心正上方、绿框顶面上的点（与画面中心对齐）。
  沿外法向从物心推到顶面外缘，躺/竖放/微俯仰都适用。
  """
  n = np.asarray(outward_normal, dtype=float).reshape(3)
  n = n / max(float(np.linalg.norm(n)), 1e-9)
  oc = np.asarray(T_base_obj[:3, 3], dtype=float).reshape(3)
  top_h = _fp_face_top_height_m(T_base_obj, half_extents, n)
  return oc + n * (top_h - float(np.dot(oc, n)))


def _top_surface_z_at_xy(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
    outward_normal: np.ndarray,
    x: float,
    y: float,
) -> float:
  """物心 (x,y) 处绿框顶面世界 Z（米）；竖放时用法向平面，不用全局 zmax。"""
  n = np.asarray(outward_normal, dtype=float).reshape(3)
  n = n / max(float(np.linalg.norm(n)), 1e-9)
  top_h = _fp_face_top_height_m(T_base_obj, half_extents, n)
  if abs(float(n[2])) < 1e-6:
    return _green_box_zmax_world(T_base_obj, half_extents)
  return float((top_h - n[0] * float(x) - n[1] * float(y)) / n[2])


def load_gripper_tcp_to_pad_m(cfg: dict) -> float:
  """
  瑞尔曼末端法兰(pose6) 到物体顶面的外退（米）。
  不是夹爪法兰盘。优先 fp_grasp.flange_above_top_m；
  否则用 gripper.body_length_m + close_gap_to_top_m；再否则 tcp_to_pad_m。
  """
  g = cfg.get('arm', {}).get('fp_grasp', {})
  if g.get('flange_above_top_m') is not None:
    return float(g['flange_above_top_m'])
  grip = cfg.get('gripper', {})
  if grip.get('body_length_m') is not None and grip.get('close_gap_to_top_m') is not None:
    return float(grip['body_length_m']) + float(grip['close_gap_to_top_m'])
  return float(grip.get('tcp_to_pad_m', 0.035))


def _grasp_tcp_on_fp_face(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
    face: dict,
    grasp_face_inset_m: float,
    *,
    grasp_tcp_above_top_m: float = 0.01,
    gripper_tcp_to_pad_m: float = 0.035,
    table_z_m: float = 0.0,
    grasp_table_clearance_m: float = 0.015,
    R_ee: Optional[np.ndarray] = None,
    top_face_mode: bool = False,
    vertical_extra_m: float = 0.0,
    cfg: Optional[dict] = None,
    approach_axis: str = 'z',
) -> tuple[np.ndarray, dict]:
  """
  关爪时末端 TCP 位姿位置。
  approach_axis='z'：沿工具 -Z 外退（瑞尔曼法兰）；
  approach_axis='x'：沿工具 -X 外退（高擎 tool_link）。
  top_face：物心正上方落在绿框抓取面平面上，再沿进给轴外退。
  """
  n = np.asarray(face['outward_normal'], dtype=float).reshape(3)
  n = n / max(float(np.linalg.norm(n)), 1e-9)
  fc = np.asarray(face['face_center'], dtype=float).reshape(3)
  oc = np.asarray(T_base_obj[:3, 3], dtype=float).reshape(3)
  pad_m = float(gripper_tcp_to_pad_m)
  above_m = float(grasp_tcp_above_top_m)
  extra_m = float(vertical_extra_m)
  standoff = above_m + pad_m + extra_m
  corners = _obb_corners_base(T_base_obj, half_extents)
  zmax_fp = _green_box_zmax_world(T_base_obj, half_extents)
  zmin_fp = float(corners[:, 2].min())
  z_span = zmax_fp - zmin_fp
  up_dot = float(face.get('up_dot', abs(float(n[2]))))
  z_src = 'fp_green_box'
  ax = str(approach_axis).strip().lower()
  col = 0 if ax == 'x' else 2

  # 顶面抓取（含斜放）：面平面 + 沿进给轴外退。
  use_top_face_height = bool(top_face_mode) or abs(float(n[2])) > 0.65
  if use_top_face_height:
    surf = _top_surface_point_above_center(T_base_obj, half_extents, n)
    finger_drop = 0.0
    if R_ee is not None and cfg is not None:
      finger_drop = _gripper_extra_drop_world_z(R_ee, cfg)
    if R_ee is not None:
      R = np.asarray(R_ee, dtype=float).reshape(3, 3)
      grasp_pos = surf - R[:, col] * standoff
    else:
      grasp_pos = surf + n * standoff
    if finger_drop > 1e-9:
      grasp_pos = np.asarray(grasp_pos, dtype=float).copy()
      grasp_pos[2] += finger_drop
    along_n = float(np.dot(np.asarray(grasp_pos, dtype=float) - surf, n))
    face['surf_z_mode'] = 'face_plane_above_center'
    face['finger_drop_mm'] = round(finger_drop * 1000.0, 1)
    face['flange_above_top_mm'] = round(along_n * 1000.0, 1)
    face['grasp_z_formula'] = (
        f'面点({round(float(surf[0])*1000,1)},{round(float(surf[1])*1000,1)},'
        f'{round(float(surf[2])*1000,1)}) + 进给外退({round(pad_m*1000,1)})'
        f' axis={ax}'
        f' + finger_drop_z({round(finger_drop*1000,1)})'
        f' → z={round(float(grasp_pos[2])*1000,1)}')
  else:
    surf = _grasp_surface_from_fp_box(T_base_obj, half_extents, fc, n)
    if R_ee is not None:
      R = np.asarray(R_ee, dtype=float).reshape(3, 3)
      grasp_pos = surf - R[:, col] * standoff
    else:
      grasp_pos = surf + n * standoff
  face_clamped = False
  if use_top_face_height:
    floor_z = float(table_z_m) + float(grasp_table_clearance_m) + standoff
    if float(grasp_pos[2]) < floor_z:
      grasp_pos = np.asarray(grasp_pos, dtype=float).copy()
      grasp_pos[2] = floor_z
      face_clamped = True
      face['floor_clamp_mm'] = round(floor_z * 1000.0, 1)

  corners = _obb_corners_base(T_base_obj, half_extents)
  zmin = float(corners[:, 2].min())
  face = dict(face)
  face['zmax_source'] = z_src
  face['fp_top_h_mm'] = round(zmax_fp * 1000.0, 1)
  face['zmax_fp_mm'] = round(zmax_fp * 1000.0, 1)
  face['obj_center_mm'] = [round(float(x) * 1000.0, 1) for x in oc]
  face['face_center_mm'] = [round(float(x) * 1000.0, 1) for x in fc]
  face['grasp_z_clamped'] = face_clamped
  face['tcp_standoff_mm'] = round(standoff * 1000.0, 1)
  face['pad_mm'] = round(pad_m * 1000.0, 1)
  face['z_span_mm'] = round(z_span * 1000.0, 1)
  face['box_half_extents_m'] = [float(x) for x in np.asarray(half_extents, float).reshape(3)]
  face['box_extents_m'] = fp_box_extents_m(half_extents)
  face['box_extents_base_mm'] = [round(x * 1000.0, 1) for x in fp_box_extents_base_m(
      T_base_obj, half_extents)]
  face['box_zminmax_mm'] = [round(zmin * 1000.0, 1), round(zmax_fp * 1000.0, 1)]
  face['grasp_top_z_mm'] = round(float(surf[2]) * 1000.0, 1)
  face['grasp_close_z_mm'] = round(float(grasp_pos[2]) * 1000.0, 1)
  face['surface_z_mm'] = round(float(surf[2]) * 1000.0, 1)
  face['grasp_inset_mm'] = round(float(grasp_face_inset_m) * 1000.0, 1)
  face['up_dot'] = round(up_dot, 3)
  face['normal_z'] = round(float(n[2]), 3)
  face['standoff_world_z_mm'] = round(standoff * 1000.0, 1)
  face['approach_axis'] = ax
  if use_top_face_height:
    face['height_mode'] = 'face_plane'
  return grasp_pos, face


def _ee_rotation_for_face_approach(
    outward_normal: np.ndarray,
    R_prefer: Optional[np.ndarray] = None,
) -> np.ndarray:
  """末端 Z 指向物体内；绕工具 Z 翻转 180° 时选离位置1/示教腕姿更近的解，避免夹爪翻面。"""
  z_in = -np.asarray(outward_normal, dtype=float).reshape(3)
  z_in = z_in / max(np.linalg.norm(z_in), 1e-9)
  world_up = np.array([0.0, 0.0, 1.0], dtype=float)
  if abs(float(np.dot(world_up, z_in))) > 0.92:
    world_up = np.array([0.0, 1.0, 0.0], dtype=float)
  x = np.cross(world_up, z_in)
  x_norm = float(np.linalg.norm(x))
  if x_norm < 1e-6:
    world_up = np.array([1.0, 0.0, 0.0], dtype=float)
    x = np.cross(world_up, z_in)
    x_norm = float(np.linalg.norm(x))
  x = x / x_norm
  y = np.cross(z_in, x)
  y = y / max(np.linalg.norm(y), 1e-9)
  R0 = np.column_stack([x, y, z_in])
  if R_prefer is None:
    return R0
  R_pref = np.asarray(R_prefer, dtype=float).reshape(3, 3)
  R_flip = R0 @ np.diag([-1.0, -1.0, 1.0])
  d0 = _rotation_delta_deg(R0, R_pref)
  d1 = _rotation_delta_deg(R_flip, R_pref)
  return R0 if d0 <= d1 else R_flip


def _obj_z_blue_is_mostly_horizontal(R_obj: np.ndarray, thresh: float = 0.55) -> bool:
  """躺放：物体系蓝轴（+Z）近水平，不能再用正上方示教位。"""
  return abs(float(np.asarray(R_obj, float).reshape(3, 3)[2, 2])) < thresh


def _tcp_on_blue_approach_face(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
    face_inset_m: float,
) -> np.ndarray:
  """沿 FP 蓝轴（物体系 +Z）方向在 +Z 面外进爪，TCP 落在该面内侧 face_inset。"""
  R_obj = T_base_obj[:3, :3]
  center = T_base_obj[:3, 3]
  hz = float(np.asarray(half_extents, dtype=float).reshape(3)[2])
  z_blue = R_obj[:, 2]
  z_blue = z_blue / max(float(np.linalg.norm(z_blue)), 1e-9)
  face_outer = center + R_obj @ np.array([0.0, 0.0, hz], dtype=float)
  return face_outer - z_blue * float(face_inset_m)


def _rotation_tool_z_parallel(
    z_obj_in_base: np.ndarray,
    R_reference: Optional[np.ndarray] = None,
    *,
    R_obj: Optional[np.ndarray] = None,
    R_oe: Optional[np.ndarray] = None,
) -> np.ndarray:
  """
  末端工具 Z 轴（法兰进给轴）与物体系 +Z（FP 蓝线）平行。
  绕进给轴的滚转沿用 R_reference（R_base_obj @ R_obj_to_ee）。
  """
  z = np.asarray(z_obj_in_base, dtype=float).reshape(3)
  z = z / max(float(np.linalg.norm(z)), 1e-9)

  if R_obj is not None and R_oe is not None:
    Ro = np.asarray(R_obj, dtype=float).reshape(3, 3)
    Roe = np.asarray(R_oe, dtype=float).reshape(3, 3)
    z_ee_world = Ro @ Roe.T[:, 2]
    if float(np.dot(z, z_ee_world)) < 0.0:
      z = -z
    for idx in (0, 1):
      axis_obj = Roe.T[:, idx]
      axis_w = Ro @ axis_obj
      axis_w = axis_w - np.dot(axis_w, z) * z
      axis_norm = float(np.linalg.norm(axis_w))
      if axis_norm >= 1e-6:
        x = axis_w / axis_norm
        y = np.cross(z, x)
        y = y / max(float(np.linalg.norm(y)), 1e-9)
        return np.column_stack([x, y, z])

  if R_reference is not None:
    R_ref = np.asarray(R_reference, dtype=float).reshape(3, 3)
    if float(np.dot(R_ref[:, 2], z)) < 0.0:
      z = -z
    x = R_ref[:, 0] - np.dot(R_ref[:, 0], z) * z
    x_norm = float(np.linalg.norm(x))
    if x_norm < 1e-6:
      x = R_ref[:, 1] - np.dot(R_ref[:, 1], z) * z
      x_norm = float(np.linalg.norm(x))
    if x_norm >= 1e-6:
      x = x / x_norm
      y = np.cross(z, x)
      y = y / max(float(np.linalg.norm(y)), 1e-9)
      return np.column_stack([x, y, z])

  world_up = np.array([0.0, 0.0, 1.0], dtype=float)
  if abs(float(np.dot(world_up, z))) > 0.92:
    world_up = np.array([0.0, 1.0, 0.0], dtype=float)
  x = np.cross(world_up, z)
  x_norm = float(np.linalg.norm(x))
  if x_norm < 1e-6:
    world_up = np.array([1.0, 0.0, 0.0], dtype=float)
    x = np.cross(world_up, z)
    x_norm = float(np.linalg.norm(x))
  x = x / x_norm
  y = np.cross(z, x)
  y = y / max(float(np.linalg.norm(y)), 1e-9)
  return np.column_stack([x, y, z])


def _object_bbox_z_range(
    T_base_obj: np.ndarray,
    half_extents: Sequence[float],
) -> tuple[float, float]:
  """物体 OBB 在基座系下的最低/最高 z（米）。"""
  R = T_base_obj[:3, :3]
  oc = T_base_obj[:3, 3]
  h = np.asarray(half_extents, dtype=float).reshape(3)
  zs = []
  for sx in (-1.0, 1.0):
    for sy in (-1.0, 1.0):
      for sz in (-1.0, 1.0):
        local = np.array([sx * h[0], sy * h[1], sz * h[2]], dtype=float)
        zs.append(float((oc + R @ local)[2]))
  return min(zs), max(zs)


def build_nearest_face_staged_pose6(
    ob_in_cam,
    T_base_cam: np.ndarray,
    ee_xyz: np.ndarray,
    half_extents: Sequence[float],
    approach_m: float,
    init_pose6: list[float],
    retreat_axis: str = 'neg_z',
    lift_axis: str = 'neg_z',
    *,
    flange_face_gap_m: float = 0.06,
    grasp_face_inset_m: float = 0.008,
    robot=None,
    R_init: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
  """
  FP 6D + 当前末端 → 抓「离臂最近」那一面。
  hover 用位置1姿态只平移到物体上方；到 pre 再转成抓取姿态（避免 movel 失败）。
  """
  T_base_obj = ob_in_cam_to_arm_T(ob_in_cam, T_base_cam)
  face = pick_nearest_face_to_ee(T_base_obj, ee_xyz, half_extents)
  n = face['outward_normal']
  grasp_pos, face = _grasp_tcp_on_fp_face(
      T_base_obj, half_extents, face, grasp_face_inset_m)

  if R_init is None:
    if robot is None:
      raise ValueError('build_nearest_face_staged_pose6 需要 robot 或 R_init')
    R_init = pose6_to_matrix(robot, init_pose6)[:3, :3]
  else:
    R_init = np.asarray(R_init, dtype=float).reshape(3, 3)

  T_grasp = np.eye(4, dtype=float)
  # 跟物体朝向：只用法向建系，不往位置1腕姿上拽（否则会看起来夹爪一直不转）
  T_grasp[:3, :3] = _ee_rotation_for_face_approach(n, R_prefer=R_init)
  T_grasp[:3, 3] = grasp_pos

  T_pre = _offset_along_ee_axis(T_grasp, approach_m, retreat_axis)
  T_lift = _offset_along_ee_axis(T_grasp, approach_m, lift_axis)
  init_z = float(init_pose6[2])
  obj_xy = T_base_obj[:3, 3][:2]
  T_hover = np.eye(4, dtype=float)
  T_hover[:3, :3] = T_grasp[:3, :3].copy()
  T_hover[:3, 3] = [float(obj_xy[0]), float(obj_xy[1]), init_z]
  face['mode'] = 'nearest_face'
  face['hover_xy_from'] = 'object_center'
  return T_hover, T_pre, T_grasp, T_lift, T_base_obj, face


def build_top_face_grasp_staged_pose6(
    ob_in_cam,
    T_base_cam: np.ndarray,
    half_extents: Sequence[float],
    approach_m: float,
    lift_m: float,
    init_pose6: list[float],
    retreat_axis: str = 'neg_z',
    lift_axis: str = 'neg_z',
    *,
    flange_face_gap_m: float = 0.06,
    grasp_face_inset_m: float = 0.008,
    table_z_m: float = 0.0,
    grasp_table_clearance_m: float = 0.015,
    grasp_tcp_above_top_m: float = 0.01,
    gripper_tcp_to_pad_m: float = 0.035,
    grasp_vertical_extra_m: float = 0.0,
    robot=None,
    R_init: Optional[np.ndarray] = None,
    cfg: Optional[dict] = None,
    T_base_obj: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
  """
  唯一抓取规则：FP ob_in_cam → 基座系方盒 → 抓最朝上的面。
  实时跟踪可直接传 T_base_obj（基座系已钉死），避免旧相机姿×新末端乱跳。
  高擎：cfg.arm.fp_grasp.approach_axis=x（tool_link +X 进给）。
  """
  if T_base_obj is not None:
    T_base_obj = np.asarray(T_base_obj, dtype=float).reshape(4, 4)
  else:
    T_base_obj = ob_in_cam_to_arm_T(ob_in_cam, T_base_cam)
  face = pick_top_grasp_face(T_base_obj, half_extents)
  n = np.asarray(face['outward_normal'], dtype=float).reshape(3)
  n = n / max(float(np.linalg.norm(n)), 1e-9)
  all_faces = _enumerate_box_faces(T_base_obj, half_extents)
  max_area = max(float(f['area_m2']) for f in all_faces) if all_faces else 0.0
  narrow_top = float(face.get('area_m2', 0)) < 0.82 * max_area - 1e-9
  face['narrow_top'] = narrow_top

  if R_init is None:
    if robot is None:
      raise ValueError('需要 robot 或 R_init')
    R_init = pose6_to_matrix(robot, init_pose6)[:3, :3]
  else:
    R_init = np.asarray(R_init, dtype=float).reshape(3, 3)

  approach_axis = 'z'
  if cfg is not None:
    approach_axis = str(
        load_fp_grasp_cfg(cfg).get('approach_axis', 'z')).strip().lower()

  R_grasp = _ee_rotation_for_top_face(
      T_base_obj, n, R_prefer=R_init, half_extents=half_extents,
      approach_axis=approach_axis)
  grasp_pos, face = _grasp_tcp_on_fp_face(
      T_base_obj, half_extents, face, grasp_face_inset_m,
      grasp_tcp_above_top_m=grasp_tcp_above_top_m,
      gripper_tcp_to_pad_m=gripper_tcp_to_pad_m,
      table_z_m=table_z_m, grasp_table_clearance_m=grasp_table_clearance_m,
      R_ee=R_grasp, top_face_mode=True,
      vertical_extra_m=float(grasp_vertical_extra_m),
      cfg=cfg, approach_axis=approach_axis)

  T_grasp = np.eye(4, dtype=float)
  T_grasp[:3, :3] = R_grasp
  T_grasp[:3, 3] = grasp_pos
  # 与瑞尔曼一致：pre/lift 沿抓取面外法向，不改成世界 Z
  pre_pos = grasp_pos + n * float(approach_m)
  lift_pos = grasp_pos + n * float(lift_m)
  T_pre = T_grasp.copy()
  T_pre[:3, 3] = pre_pos
  T_lift = T_grasp.copy()
  T_lift[:3, 3] = lift_pos
  face['descent_mode'] = 'face_normal'
  face['grasp_close_z_m'] = float(grasp_pos[2])

  init_z = float(init_pose6[2])
  obj_xy = T_base_obj[:3, 3][:2]
  T_hover = np.eye(4, dtype=float)
  T_hover[:3, :3] = R_init
  T_hover[:3, 3] = [float(obj_xy[0]), float(obj_xy[1]), init_z]
  face['mode'] = 'top_face_fp'
  return T_hover, T_pre, T_grasp, T_lift, T_base_obj, face


def build_face_parallel_staged_pose6(
    ob_in_cam,
    T_base_cam: np.ndarray,
    ee_xyz: np.ndarray,
    half_extents: Sequence[float],
    approach_m: float,
    lift_m: float,
    init_pose6: list[float],
    retreat_axis: str = 'neg_z',
    lift_axis: str = 'neg_z',
    *,
    flange_face_gap_m: float = 0.06,
    grasp_face_inset_m: float = 0.008,
    face_pick: str = 'auto',
    face_pick_min_cam_dot: float = 0.45,
    face_pick_depth_margin_m: float = 0.02,
    face_pick_large_face_ratio: float = 0.85,
    lying_top_min_up_dot: float = 0.55,
    hand_eye_mode: str = HAND_EYE_EYE_IN_HAND,
    robot=None,
    R_init: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
  """
  FP 6D 方盒 → 选抓取面 → 法兰与该面平行，沿外法向进爪。
  关爪高度由绿框顶面 + grasp_face_inset_m 决定（与画面绿框一致）。
  """
  T_base_obj = ob_in_cam_to_arm_T(ob_in_cam, T_base_cam)
  face = resolve_grasp_face(
      T_base_obj, T_base_cam, half_extents, face_pick=face_pick, ee_xyz=ee_xyz,
      face_pick_min_cam_dot=face_pick_min_cam_dot,
      face_pick_depth_margin_m=face_pick_depth_margin_m,
      face_pick_large_face_ratio=face_pick_large_face_ratio,
      hand_eye_mode=hand_eye_mode,
      lying_top_min_up_dot=lying_top_min_up_dot)
  n = face['outward_normal']
  grasp_pos, face = _grasp_tcp_on_fp_face(
      T_base_obj, half_extents, face, grasp_face_inset_m)
  obj_cam = _point_base_to_cam_xyz(T_base_obj[:3, 3], T_base_cam)
  face_center_cam = _point_base_to_cam_xyz(face['face_center'], T_base_cam)
  face['flange_gap_mm'] = round(float(flange_face_gap_m) * 1000.0, 1)
  face['obj_cam_depth_mm'] = round(float(obj_cam[2]) * 1000.0, 1)
  face['face_cam_depth_mm'] = round(float(face_center_cam[2]) * 1000.0, 1)
  face['mode'] = 'face_parallel'

  if R_init is None:
    if robot is None:
      raise ValueError('build_face_parallel_staged_pose6 需要 robot 或 R_init')
    R_init = pose6_to_matrix(robot, init_pose6)[:3, :3]
  else:
    R_init = np.asarray(R_init, dtype=float).reshape(3, 3)

  T_grasp = np.eye(4, dtype=float)
  use_top_rot = (
      str(face_pick).strip().lower() == 'top'
      or float(face.get('up_dot', 0)) >= float(lying_top_min_up_dot))
  if use_top_rot:
    T_grasp[:3, :3] = _ee_rotation_for_top_face(
        T_base_obj, n, R_prefer=R_init, half_extents=half_extents)
  else:
    T_grasp[:3, :3] = _ee_rotation_for_face_approach(n, R_prefer=R_init)
  T_grasp[:3, 3] = grasp_pos

  T_pre = _offset_along_ee_axis(T_grasp, approach_m, retreat_axis)
  T_lift = _offset_along_ee_axis(T_grasp, lift_m, lift_axis)
  init_z = float(init_pose6[2])
  obj_xy = T_base_obj[:3, 3][:2]
  T_hover = np.eye(4, dtype=float)
  T_hover[:3, :3] = R_init
  T_hover[:3, 3] = [float(obj_xy[0]), float(obj_xy[1]), init_z]
  return T_hover, T_pre, T_grasp, T_lift, T_base_obj, face


def build_fp_grasp_transforms(
    ob_in_cam,
    T_base_cam: np.ndarray,
    T_obj_to_ee: np.ndarray,
    approach_m: float,
    lift_m: float,
    retreat_axis: str = 'pos_z',
    lift_axis: str = 'pos_z',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  FP 6D → 预抓取 / 抓取 / 抬起 三个末端 4×4（基座系）。
  物体怎么转，末端姿态跟着转（T_obj_to_ee 在物体坐标系下固定）。
  """
  T_base_obj = ob_in_cam_to_arm_T(ob_in_cam, T_base_cam)
  T_grasp = T_base_obj @ T_obj_to_ee
  T_pre = _offset_along_ee_axis(T_grasp, approach_m, retreat_axis)
  T_lift = _offset_along_ee_axis(T_grasp, lift_m, lift_axis)
  return T_pre, T_grasp, T_lift, T_base_obj


def explain_reach(cur_xyz: np.ndarray, obj_xyz: np.ndarray,
                  observe_xyz: Optional[np.ndarray] = None) -> str:
  cur = np.asarray(cur_xyz, dtype=float).reshape(3)
  obj = np.asarray(obj_xyz, dtype=float).reshape(3)
  d = float(np.linalg.norm(obj - cur)) * 1000.0
  dx = (obj[0] - cur[0]) * 1000.0
  lines = [
      f'物体目标基座坐标(mm): x={obj[0]*1000:.1f} y={obj[1]*1000:.1f} z={obj[2]*1000:.1f}',
      f'当前末端(mm):         x={cur[0]*1000:.1f} y={cur[1]*1000:.1f} z={cur[2]*1000:.1f}',
      f'直线距离约 {d:.0f} mm，Δx={dx:+.0f} mm',
  ]
  if observe_xyz is not None:
    obs = np.asarray(observe_xyz, dtype=float).reshape(3)
    lines.append(f'观察位(mm): x={obs[0]*1000:.1f} y={obs[1]*1000:.1f} z={obs[2]*1000:.1f}')
  return '\n'.join(lines)
