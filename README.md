# Panthera-HT

[Contents](#contents) · [Requirements](#requirements) · [Quick start (Python SDK)](#quick-start-python-sdk) · [Quick start (Host digital twin)](#quick-start-host-digital-twin) · [Host usage notes](#host-usage-notes) · [Safety](#safety)


TOPSUN workspace for the **Panthera-HT** 6-DOF robotic arm.

This repository packages the official HighTorque Panthera-HT materials used in-house, plus Host digital-twin integration notes for live control and visualization.

## Contents

| Path | Description |
|------|-------------|
| `Panthera-HT_SDK/` | C++ / Python SDK, motor wheels, example scripts |
| `Panthera-HT_Host/` | Web digital twin Host (Three.js frontend + Flask backend) |
| `Panthera-HT_Main/` | Project overview and documentation assets |

Upstream references:

- [HighTorque-Robotics/Panthera-HT_SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK)
- [HighTorque-Robotics/Panthera-HT_Host](https://github.com/HighTorque-Robotics/Panthera-HT_Host)
- [HighTorque-Robotics/Panthera-HT_Main](https://github.com/HighTorque-Robotics/Panthera-HT_Main)

## Requirements

- Ubuntu 22.04 (recommended)
- Python 3.10+
- Node.js 18+ (for Host frontend)
- Serial access to the arm communication board (`/dev/ttyACM*`)

## Quick start (Python SDK)

```bash
python3 -m venv ~/venvs/panthera
source ~/venvs/panthera/bin/activate
cd Panthera-HT_SDK/panthera_python
pip install motor_whl/hightorque_robot-1.2.0-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt

# Grant serial permission (current session)
sudo chmod 666 /dev/ttyACM* /dev/ttyUSB0

cd scripts
python 0_robot_get_state.py
```

## Quick start (Host digital twin)

```bash
# Backend (live robot)
cd Panthera-HT_Host
./backend.sh --live

# Frontend (another terminal)
./frontend.sh
# Open http://localhost:3000/
```

Demo mode (no hardware):

```bash
./backend.sh --demo
```

## Host usage notes

1. Confirm the backend is **live** (not Demo) when you need the real arm pose.
2. Open `http://localhost:3000/` for the dark digital-twin UI.
3. After connecting, joint sliders and the 3D model follow live joint states.
4. In Position mode, drag a link to preview; release sends the final joint target.
5. Use **Send Position** to command the staged joint targets.

## Safety

- Keep the workspace clear before enabling motion.
- Support the arm when power is removed.
- Use E-stop / stop commands if motion is unexpected.

## License

Upstream packages are MIT-licensed by HighTorque Robotics unless otherwise noted in each subdirectory. TOPSUN-BOT additions follow the same license terms.
