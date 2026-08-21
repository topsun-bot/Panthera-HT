# Panthera-HT × FoundationPose Grasp (ported from Realman)

**Top-face grasp rules** ported from `10.10.154.183:~/FoundationPose/hands`, adapted for the Panthera arm.  
**Do not modify** existing calibration/click scripts or the Host.

## One-shot full pipeline (recommended)

Remote FoundationPose is copied under `third_party/FoundationPose/` and wired to Panthera:

```bash
# 1) Stop Host app.py first (serial port is exclusive)
# 2) Launch
bash third_party/FoundationPose/run_panthera_grasp.sh
# (run from the Panthera-HT repo root)
```

Flow:

1. Auto-return to the home pose (your joint angles) and cache hand-eye  
2. YOLO detection boxes  
3. **Space** → FP register & track  
4. **Space again** → grasp → lift → **return to pose 1**

Perception Python: `~/miniconda3/envs/internnav`  
Grasp Python: `~/venvs/panthera`

## Home / observe pose (current)


From the Host UI current pose:

- Joints (rad): `[1.56, 0.82, 1.07, 0.78, 0.0, 0.0]`
- TCP (m / rad): `xyz=[0, 0.1215, 0.355]`, `rpy≈[0.641, -0.025, 1.542]` (converted from UI degrees)

Grasp start first `moveJ`s back to these joints, then opens the gripper and plans.


1. `ob_in_cam` → base-frame box `T_base_obj` (eye-in-hand: `T_base_link6 @ T_link6_cam @ T_cam_obj`)
2. Pick the FP green-box face that is **most upward**
3. Panthera: `tool_link` **+X along −outward normal** for approach; jaw open/close along the top-face long edge
4. Waypoints: `pre` (along normal above) → `grasp` (close jaws) → `lift`

## Layout

```
foundationpose_grasp/
├── config/hardware_panthera.yaml   # hand-eye + fp_grasp
├── bridge/pose_udp.py              # reusable UDP protocol on the FP side
├── bridge/pose_subscriber.py       # local pose receive / debug
├── grasp/coord_utils.py            # grasp rules
├── grasp/grasp_pipeline.py         # grasp + lift
└── panthera_client.py              # Panthera driver adapter
```

## Usage

Stop Host first (serial exclusive):

```bash
# Confirm serial is free
fuser /dev/ttyACM0 2>/dev/null || echo free

source ~/venvs/panthera/bin/activate
cd Panthera-HT_SDK/panthera_python/scripts
# (from the Panthera-HT repo root)
```

### 1) Remote FP sends UDP; this machine only receives (debug)

On the remote (Realman) host when running FP, add:

```bash
--pose_udp_target <this-machine-IP>:8899
```

Locally:

```bash
python3 foundationpose_grasp/bridge/pose_subscriber.py --port 8899
```

### 2) Plan preview (recommended first)

Listen for 2 s of UDP poses, print waypoints, no motion:

```bash
python3 foundationpose_grasp/grasp/grasp_pipeline.py --listen-sec 2
```

Or a specific JSON:

```bash
python3 foundationpose_grasp/grasp/grasp_pipeline.py \
  --pose-file foundationpose_grasp/examples/sample_ob_in_cam.json
```

### 3) Real robot (grasp + lift)

```bash
python3 foundationpose_grasp/grasp/grasp_pipeline.py \
  --pose-file /path/to/pose.json --execute
```

Minimal `pose.json` fields:

```json
{
  "ob_in_cam": [[...4x4...]],
  "class_name": "TeaCaddy",
  "object_half_extents_m": [0.0265, 0.0335, 0.065],
  "lock_T_base_link6": [[...4x4...]]
}
```

Prefer writing `lock_T_base_link6` as the flange pose at Space-freeze time; if missing, use current `link6` at execute time.

## Notes

- Hand-eye matrix from `handeye_output/manual_20260819_150758`
- `flange_above_top_m` / `tcp_to_pad_m` need fine-tuning from measured gripper geometry
- First version defaults `live_track: false` (frozen frame); place/release not ported
- Perception (YOLO+FP) can still run remotely; this machine only executes
