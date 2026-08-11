import torch
import numpy as np
import traceback
from torchvision.transforms.functional import to_tensor, to_pil_image
from pathlib import Path
import os
import gc 
from huggingface_hub import snapshot_download 

from .RIFE.RIFE_HDv3 import Model as RIFEBaseModel 
#from .message_manager import MessageManager 
import devicetorch

# Get the directory of the current script (rife_core.py)
_MODULE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) # __file__ gives path to current script

# MODEL_RIFE_PATH = "model_rife" # OLD - this is relative to CWD
MODEL_RIFE_PATH = _MODULE_DIR / "model_rife" # NEW - relative to this script's location
RIFE_MODEL_FILENAME = "flownet.pkl"

class RIFEHandler:
    def __init__(self):
        self.model_dir = Path(MODEL_RIFE_PATH)
        self.model_file_path = self.model_dir / RIFE_MODEL_FILENAME
        self.rife_model = None

    def _ensure_model_downloaded_and_loaded(self) -> bool:
        if self.rife_model is not None:
            return True

        os.makedirs(self.model_dir, exist_ok=True)

        if not self.model_file_path.exists():
            print("INFO: RIFE model weights not found. Downloading...")
            try:
                snapshot_download(
                    repo_id="AlexWortega/RIFE",
                    allow_patterns=["*.pkl", "*.pth"],
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
                if self.model_file_path.exists():
                    print("SUCCESS: RIFE model weights downloaded.")
                else:
                    print(f"ERROR: RIFE model download failed. {RIFE_MODEL_FILENAME} not found.")
                    return False
            except Exception as e:
                print(f"ERROR: Failed to download RIFE model weights: {e}")
                return False

        try:
            print(f"INFO: Loading RIFE model from {self.model_dir}...")
            self.rife_model = RIFEBaseModel(local_rank=-1)
            self.rife_model.load_model(str(self.model_dir), -1)
            self.rife_model.eval()
            print("SUCCESS: RIFE model loaded.")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load RIFE model: {e}")
            traceback.print_exc()
            self.rife_model = None
            return False

    def unload_model(self):
        if self.rife_model is not None:
            print("INFO: Unloading RIFE model...")
            del self.rife_model
            self.rife_model = None
            devicetorch.empty_cache(torch)
            gc.collect()
            print("SUCCESS: RIFE model unloaded and memory cleared.")

    def interpolate_between_frames(self, frame1_np: np.ndarray, frame2_np: np.ndarray) -> np.ndarray | None:
        if self.rife_model is None:
            print("ERROR: RIFE model not loaded for interpolation.")
            return None

        try:
            # Contiguous RGB uint8 → float tensor (avoid extra PIL copies)
            img0_tensor = to_tensor(frame1_np).unsqueeze(0)
            img1_tensor = to_tensor(frame2_np).unsqueeze(0)

            img0 = devicetorch.to(torch, img0_tensor)
            img1 = devicetorch.to(torch, img1_tensor)

            required_multiple = 32
            h_orig, w_orig = img0.shape[2], img0.shape[3]
            pad_h = (required_multiple - h_orig % required_multiple) % required_multiple
            pad_w = (required_multiple - w_orig % required_multiple) % required_multiple

            if pad_h > 0 or pad_w > 0:
                img0 = torch.nn.functional.pad(img0, (0, pad_w, 0, pad_h), mode='replicate')
                img1 = torch.nn.functional.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')

            # inference_mode is faster than no_grad; autocast on CUDA is typically ~1.5–2×
            # faster on 40-series with no visible quality loss for RIFE HD at scale=1.0
            use_cuda = bool(img0.is_cuda)
            with torch.inference_mode():
                if use_cuda:
                    try:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            middle_frame_tensor = self.rife_model.inference(img0, img1, scale=1.0)
                    except Exception:
                        # Rare op incompat — fall back to full precision once
                        middle_frame_tensor = self.rife_model.inference(img0, img1, scale=1.0)
                else:
                    middle_frame_tensor = self.rife_model.inference(img0, img1, scale=1.0)

            if pad_h > 0 or pad_w > 0:
                middle_frame_tensor = middle_frame_tensor[:, :, :h_orig, :w_orig]

            # Direct tensor → uint8 HWC (skip PIL round-trip)
            mid = middle_frame_tensor.squeeze(0).float().clamp(0.0, 1.0)
            mid = (mid * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
            return mid.cpu().numpy()

        except Exception as e:
            print(f"ERROR: Error during RIFE frame interpolation: {e}")
            if "out of memory" in str(e).lower():
                devicetorch.empty_cache(torch)
            else:
                traceback.print_exc()
            return None