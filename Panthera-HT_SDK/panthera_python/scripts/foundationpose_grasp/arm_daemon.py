#!/usr/bin/env python3
# ============================================================================
# Panthera arm daemon: same-process "return to pose 1 → MIT hold ↔ grasp";
# never switch processes, avoid motor brake torque loss (no brake handoff).
# Protocol: TCP 127.0.0.1:9877  one JSON request line / one response line
#   {"cmd":"ping"}
#   {"cmd":"grasp","pose_file":"/tmp/xxx.json"}
#   {"cmd":"shutdown"}
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

_PKG = Path(__file__).resolve().parent
_SCRIPTS = _PKG.parent
if str(_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS))
if str(_PKG) not in sys.path:
  sys.path.insert(0, str(_PKG))

from panthera_client import (  # noqa: E402
    HOLD_KD,
    HOLD_KP,
    MAX_TQU,
    PantheraClient,
    hold_joints_mit,
)
from grasp.grasp_pipeline import run_grasp_cycle  # noqa: E402

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 9877
LOCK_PATH = Path('/tmp/panthera_fp_lock.json')
PID_PATH = Path('/tmp/panthera_arm_daemon.pid')


class ArmDaemon:
  def __init__(self, config: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    self.config = str(config)
    self.host = host
    self.port = int(port)
    self.client: Optional[PantheraClient] = None
    self.init_j = None
    self.state = 'starting'  # starting|hold|grasp|error
    self.last_error = ''
    self._stop = False
    self._server: Optional[socket.socket] = None

  def _write_lock(self) -> None:
    assert self.client is not None
    T = self.client.T_base_link6()
    ret, pose6 = self.client.current_pose6()
    if ret != 0 or not pose6:
      raise RuntimeError('failed to read pose6')
    blob = {
        'T_base_link6': np.asarray(T, float).tolist(),
        'pose6_tcp': [float(x) for x in pose6[:6]],
        'init_joints_rad': [float(x) for x in self.init_j],
        'config': self.config,
        'daemon_port': self.port,
        'daemon_pid': os.getpid(),
    }
    LOCK_PATH.write_text(json.dumps(blob), encoding='utf-8')
    PID_PATH.write_text(str(os.getpid()), encoding='utf-8')

  def startup(self) -> None:
    print('======== ArmDaemon start: return to pose 1 ========')
    print(f'config={self.config}  listen={self.host}:{self.port}')
    self.client = PantheraClient.from_config(self.config)
    self.client.connect(require_serial_free=True)
    self.client.ensure_init_pose6()
    self.init_j = list(self.client.cfg['arm']['init_joints_rad'])
    dur = float(self.client.cfg['arm'].get('move_duration_s', 4.0))
    print(f'moveJ pose 1: {[round(float(x), 3) for x in self.init_j]}')
    ret = self.client.movej(self.init_j, duration=dur)
    if ret != 0:
      raise RuntimeError(f'movej init failed ret={ret}')
    self.client.gripper_open()
    time.sleep(0.3)
    self._write_lock()
    print(f'cached observe pose → {LOCK_PATH}')
    self.state = 'hold'
    print('======== pose 1 locked (same-process hold, no brake handoff) ========')

  def _hold_once(self) -> None:
    q = list(self.init_j)
    vel = [0.0] * self.client.robot.motor_count
    tqe = np.asarray(self.client.robot.get_Gravity(q), float)
    tqe = np.clip(tqe, -np.asarray(MAX_TQU), np.asarray(MAX_TQU))
    self.client.robot.pos_vel_tqe_kp_kd(
        q, vel, tqe.tolist(), HOLD_KP, HOLD_KD)

  def _handle_grasp(self, pose_file: str) -> dict:
    assert self.client is not None
    self.state = 'grasp'
    self.last_error = ''
    try:
      with open(pose_file, encoding='utf-8') as f:
        meta = json.load(f)
      ob = np.asarray(meta['ob_in_cam'], dtype=float).reshape(4, 4)
      lock_T = None
      raw = meta.get('lock_T_base_link6') or meta.get('T_base_link6')
      if raw is not None:
        lock_T = np.asarray(raw, dtype=float).reshape(4, 4)
      class_name = str(meta.get('class_name', '') or '')
      meta_half = None
      if meta.get('object_half_extents_m'):
        meta_half = [float(x) for x in meta['object_half_extents_m'][:3]]
      objects_yaml = str(meta['objects_yaml']) if meta.get('objects_yaml') else None

      # Refresh lock once more before grasp (still at observe pose)
      if lock_T is None:
        lock_T = self.client.T_base_link6()
        self._write_lock()

      run_grasp_cycle(
          ob,
          config=self.config,
          dry_run=False,
          lock_T_base_link6=lock_T,
          class_name=class_name,
          meta_half_extents=meta_half,
          objects_yaml=objects_yaml,
          client=self.client,
          release_on_exit=False,
          skip_home_on_start=True,
      )
      self._write_lock()
      self.state = 'hold'
      return {'ok': True, 'state': self.state}
    except Exception as e:
      self.last_error = f'{e}'
      traceback.print_exc()
      # run_grasp_cycle already tries to return to pose 1; hard-hold briefly here
      try:
        hold_joints_mit(self.client.robot, self.init_j, seconds=0.5)
        self._write_lock()
      except Exception:
        pass
      self.state = 'hold'
      return {'ok': False, 'error': self.last_error, 'state': self.state}

  def _dispatch(self, req: dict) -> dict:
    cmd = str(req.get('cmd', '')).lower()
    if cmd == 'ping':
      return {
          'ok': True,
          'state': self.state,
          'pid': os.getpid(),
          'error': self.last_error,
      }
    if cmd == 'shutdown':
      self._stop = True
      return {'ok': True, 'state': 'shutdown'}
    if cmd == 'grasp':
      if self.state == 'grasp':
        return {'ok': False, 'error': 'busy', 'state': self.state}
      pose_file = req.get('pose_file')
      if not pose_file:
        return {'ok': False, 'error': 'missing pose_file'}
      return self._handle_grasp(str(pose_file))
    return {'ok': False, 'error': f'unknown cmd {cmd}'}

  def _serve_ready(self, conn: socket.socket) -> None:
    conn.settimeout(120.0)
    try:
      data = b''
      while b'\n' not in data:
        chunk = conn.recv(4096)
        if not chunk:
          break
        data += chunk
      line = data.split(b'\n', 1)[0].decode('utf-8', errors='replace').strip()
      if not line:
        return
      req = json.loads(line)
      resp = self._dispatch(req)
      conn.sendall((json.dumps(resp, ensure_ascii=False) + '\n').encode('utf-8'))
    except Exception as e:
      try:
        conn.sendall(
            (json.dumps({'ok': False, 'error': str(e)}) + '\n').encode('utf-8'))
      except Exception:
        pass
    finally:
      try:
        conn.close()
      except Exception:
        pass

  def run(self) -> None:
    self.startup()
    self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._server.bind((self.host, self.port))
    self._server.listen(4)
    self._server.setblocking(False)
    print(f'[daemon] listening {self.host}:{self.port}')

    def _sig(_s, _f):
      self._stop = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    try:
      while not self._stop:
        if self.state == 'hold':
          self._hold_once()
        # Non-blocking accept; grasp also occupies time inside _handle_grasp
        try:
          r, _, _ = select.select([self._server], [], [], 0.0)
        except Exception:
          r = []
        if r:
          try:
            conn, _addr = self._server.accept()
          except BlockingIOError:
            conn = None
          if conn is not None:
            self._serve_ready(conn)
        else:
          time.sleep(0.008)
    finally:
      print('[daemon] exit (will release serial; exit brake may unload torque if configured)')
      try:
        if self._server:
          self._server.close()
      except Exception:
        pass
      # Note: disconnect may still brake — only happens when the whole program ends
      if self.client is not None:
        try:
          # Do not call brake outside destructor paths; hold once more before dropping ref
          hold_joints_mit(self.client.robot, self.init_j, seconds=0.2)
        except Exception:
          pass
        # Skip disconnect() to reduce immediate brake; leave it to process exit
        self.client = None
      if PID_PATH.is_file():
        try:
          PID_PATH.unlink()
        except Exception:
          pass


def main():
  ap = argparse.ArgumentParser(
      description='Panthera arm daemon (same-process hold + grasp, no brake handoff)')
  ap.add_argument('--config', required=True)
  ap.add_argument('--host', default=DEFAULT_HOST)
  ap.add_argument('--port', type=int, default=DEFAULT_PORT)
  ns = ap.parse_args()
  ArmDaemon(ns.config, ns.host, ns.port).run()


if __name__ == '__main__':
  main()
