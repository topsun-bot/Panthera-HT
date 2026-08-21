#!/usr/bin/env python3
"""Contract: arm-daemon APIs exist and hold is no longer a cross-process handoff."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# tests/ -> foundationpose_grasp -> scripts -> panthera_python -> Panthera-HT_SDK -> repo root
_REPO = Path(__file__).resolve().parents[5]
_ADAPTER = _REPO / 'third_party' / 'FoundationPose' / 'hands' / 'panthera_adapter.py'
_DAEMON = (
    _REPO / 'Panthera-HT_SDK' / 'panthera_python' / 'scripts'
    / 'foundationpose_grasp' / 'arm_daemon.py')


def test_daemon_apis():
  assert _DAEMON.is_file()
  text = _DAEMON.read_text(encoding='utf-8')
  assert 'class ArmDaemon' in text
  assert 'release_on_exit=False' in text
  assert 'skip_home_on_start=True' in text

  spec = importlib.util.spec_from_file_location('panthera_adapter', _ADAPTER)
  ad = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(ad)
  for name in (
      'start_arm_daemon', 'stop_arm_daemon', 'startup_go_init',
      'trigger_daemon_grasp', 'prepare_grasp_handoff', 'DaemonGraspJob'):
    assert hasattr(ad, name), name
  # Old hold handoff must not stop the serial again; require same-process / no-brake wording
  assert 'same-process' in text.lower()
  assert 'no brake handoff' in text.lower() or 'avoid motor brake' in text.lower()


if __name__ == '__main__':
  test_daemon_apis()
  print('test_daemon_apis OK')
