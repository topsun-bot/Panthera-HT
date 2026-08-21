# ============================================================================
# FoundationPose pose UDP publisher: object geometric center ob_in_cam
# (matches on-screen green box and Space-key grasp).
# ============================================================================

import json
import logging
import socket
import time

import numpy as np


class PoseUdpPublisher:
  """Send poses as UDP JSON in real time to the arm-side program."""

  def __init__(self, target='', send_hz=15.0):
    self.target = target.strip()
    self.send_hz = max(float(send_hz), 0.0)
    self.enabled = bool(self.target)
    self._sock = None
    self._addr = None
    self._last_send_ts = 0.0
    if not self.enabled:
      return
    host, sep, port = self.target.partition(':')
    if sep == '' or (not host) or (not port.isdigit()):
      raise SystemExit(
          '--pose_udp_target format error, example: --pose_udp_target 127.0.0.1:8899')
    self._addr = (host, int(port))
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self._sock.setblocking(False)

  def _throttle_ok(self, ts_now):
    if self.send_hz <= 0:
      return True
    min_dt = 1.0 / self.send_hz
    if ts_now - self._last_send_ts < min_dt:
      return False
    self._last_send_ts = ts_now
    return True

  def send_pose(self, frame_id, cls_name, pose, K, lost=False):
    if not self.enabled:
      return
    ts_now = time.time()
    if not self._throttle_ok(ts_now):
      return
    payload = {
        'schema': 'foundationpose.pose.v1',
        'ts_unix': ts_now,
        'frame_id': int(frame_id),
        'class_name': cls_name or '',
        'lost': bool(lost),
        'cam_K': np.asarray(K, dtype=float).reshape(3, 3).tolist(),
    }
    if pose is not None and not lost:
      payload['ob_in_cam'] = np.asarray(pose, dtype=float).reshape(4, 4).tolist()
    try:
      body = (json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8')
      self._sock.sendto(body, self._addr)
    except Exception as e:
      logging.warning(f'pose UDP send failed: {e}')
