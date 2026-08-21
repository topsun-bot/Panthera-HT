#!/usr/bin/env python3
"""Subscribe to FP UDP poses and transform into Panthera base frame (debug)."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_SCRIPTS = _PKG.parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_PKG))

from grasp.coord_utils import (  # noqa: E402
    HAND_EYE_EYE_IN_HAND,
    load_hand_eye_mode,
    load_T_base_cam,
    ob_in_cam_to_arm_T,
)
from panthera_client import PantheraClient  # noqa: E402


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=8899)
  parser.add_argument(
      '--config',
      default=str(_PKG / 'config' / 'hardware_panthera.yaml'))
  parser.add_argument('--no-arm', action='store_true',
                      help='do not connect arm; print camera frame only')
  args = parser.parse_args()

  mode = load_hand_eye_mode(args.config)
  client = None
  if not args.no_arm and mode == HAND_EYE_EYE_IN_HAND:
    client = PantheraClient.from_config(args.config)
    client.connect()
    print('eye-in-hand: each UDP uses current link6 to map into base frame')

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  sock.bind(('0.0.0.0', args.port))
  print(f'listening UDP :{args.port} ...')

  while True:
    data, _ = sock.recvfrom(65535)
    msg = json.loads(data.decode('utf-8').strip())
    if 'ob_in_cam' not in msg:
      print(f"[{msg.get('frame_id')}] lost={msg.get('lost')}")
      continue
    T_cam = np.asarray(msg['ob_in_cam'], float).reshape(4, 4)
    t = T_cam[:3, 3]
    line = (f"[{msg.get('frame_id')}] {msg.get('class_name','')} "
            f"cam: {t[0]:.3f} {t[1]:.3f} {t[2]:.3f}")
    if client is not None:
      T_arm = ob_in_cam_to_arm_T(
          T_cam, cfg_path=args.config, T_base_ee=client.T_base_link6())
      p = T_arm[:3, 3]
      line += f" | arm: {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"
    elif mode != HAND_EYE_EYE_IN_HAND:
      T_arm = load_T_base_cam(args.config) @ T_cam
      p = T_arm[:3, 3]
      line += f" | arm: {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"
    print(line)


if __name__ == '__main__':
  main()
