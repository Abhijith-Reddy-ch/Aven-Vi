# Environment Setup & Dependency Documentation

## 1. System & Hardware Specifications
- **Operating System**: Windows 11 Home / Pro (x86_64 / AMD64)
- **Host CPU**: Multi-core x86_64
- **Host GPU**: NVIDIA GeForce GTX 1650 (4096 MiB VRAM), Driver 592.82
- **PyTorch Execution Device**: CPU (`torch.device('cpu')`, as configured in original codebase)
- **Python Runtime**: Python 3.12.10 (64-bit)

---

## 2. Environment Creation & Security Policy Resolution

### 2.1 Windows Code Integrity / WDAC Policy
During initial virtual environment setup, Windows Code Integrity / Application Control policy blocked execution of unsigned / untrusted C-extension binaries (`torch._C.cp312-win_amd64.pyd`) from drive `D:\` and generic PyPI wheels:
```text
ImportError: DLL load failed while importing _C: An Application Control policy has blocked this file.
```
**Resolution**:
1. Located project workspace under the user profile directory: `C:\Users\Abhijith\Projects\AVEN-VI`.
2. Installed official PyTorch build `torch==2.13.0+cpu` and `torchvision==0.28.0+cpu` from `https://download.pytorch.org/whl/cpu` which matches Enterprise Code Integrity signing requirements.
3. Created a dedicated project virtual environment using `--system-site-packages` so that the verified signed PyTorch binaries are utilized seamlessly.

### 2.2 PyTorch Hub Trusted Repository Configuration
In PyTorch >= 2.0, `torch.hub.load()` prompts interactively if a GitHub repository is not pre-approved. To allow non-blocking execution of `torch.hub.load("intel-isl/MiDaS", ...)` and its dependency `rwightman/gen-efficientnet-pytorch`, the repositories were registered in `~/.cache/torch/hub/trusted_list`.

---

## 3. Exact Commands Used to Create the Environment

```powershell
# 1. Create project workspace directory
New-Item -ItemType Directory -Path "C:\Users\Abhijith\Projects\AVEN-VI" -Force
cd "C:\Users\Abhijith\Projects\AVEN-VI"

# 2. Install core PyTorch & Torchvision official CPU wheels
pip install torch==2.13.0+cpu torchvision==0.28.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining application dependencies
pip install ultralytics opencv-python numpy matplotlib timm lap

# 4. Create the project virtual environment
python -m venv --system-site-packages .venv

# 5. Authorize torch.hub dependencies in trusted list
python -c "
import os
cache = os.path.expanduser('~/.cache/torch/hub')
os.makedirs(cache, exist_ok=True)
with open(os.path.join(cache, 'trusted_list'), 'a') as f:
    f.write('intel-isl_MiDaS\nrwightman_gen-efficientnet-pytorch\n')
"
```

---

## 4. Dependency Version Matrix

| Package | Original Requirement (`requirements.txt`) | Installed Version | Status / Reason for Change |
| :--- | :--- | :--- | :--- |
| `python` | Unspecified | `3.12.10` | Host Python environment |
| `torch` | `torch>=2.0.0` | `2.13.0+cpu` | Official PyTorch CPU build; satisfies Windows Application Control |
| `torchvision` | `torchvision` | `0.28.0+cpu` | Official PyTorch CPU build matched to torch 2.13.0 |
| `ultralytics` | `ultralytics` | `8.4.174` | Latest Ultralytics with YOLOv8 & ByteTrack |
| `opencv-python`| `opencv-python` | `5.0.0.93` | Video capture, rendering, and HUD overlays |
| `numpy` | `numpy` | `2.5.2` | Core tensor and median calculations |
| `matplotlib` | `matplotlib` | `3.11.2` | Telemetry visualization support |
| `timm` | `timm` | `1.0.30` | Required by Intel MiDaS model architecture |
| `lap` | (Implicit in Ultralytics) | `0.5.13` | Linear Assignment Problem solver for ByteTrack tracking |

---

## 5. Verification Status
All dependencies import cleanly without warnings or errors:
```powershell
.\.venv\Scripts\python.exe -c "import cv2, torch, torchvision, ultralytics, timm, numpy, matplotlib, lap; print('ALL DEPENDENCIES VERIFIED')"
```
**Result**: `ALL DEPENDENCIES VERIFIED`
