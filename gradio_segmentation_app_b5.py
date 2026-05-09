"""
Gradio App for RANZCR / CMUH Catheter Segmentation
Matches the preprocessing pipeline from the original notebook.

Usage:
    python gradio_segmentation_app.py

Requirements:
    pip install gradio segmentation-models-pytorch albumentations timm opencv-python-headless torch torchvision

Model weights should be placed at:
    exp_result/Jun__2_00_34_39_2025/model/
      unet++b5_2cbce_1024T15tip_lr1e4_bs4_augv2_30epo_cvc_best_fold0.pth
      ... fold1.pth ... fold4.pth
"""

import os
import sys
import numpy as np
import cv2
import gradio as gr
import torch
import torch.nn as nn
import albumentations
import segmentation_models_pytorch as smp
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
IMAGE_SIZE  = 1024
KERNEL_TYPE = "unet++b5_2cbce_1024T15tip_lr1e4_bs4_augv2_30epo_cvc"
ENET_TYPE   = "timm-efficientnet-b5"
MODEL_DIR   = "C:/WorkArea/zoey/unet/model" #模型權重路徑(PTH)
N_FOLDS     = 5

# Use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Channel labels for the two output masks
CHANNEL_LABELS = ["Clip (catheter body)", "Tip (catheter tip)"]
CHANNEL_COLORS = [
    (0, 255, 0),   # green  → clip channel
    (255, 0, 0),   # red    → tip  channel
]

# ─── Preprocessing (mirrors transforms_val in notebook) ───────────────────────
transforms_val = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
])

USE_CLAHE = False   # set True to match clahe_enhanced=True branch


def preprocess(pil_image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """
    Replicate __getitem__ preprocessing for 'test' mode.
    Returns:
        original_np : H×W×3 uint8 RGB (resized to IMAGE_SIZE for display)
        tensor      : 1×3×H×W float32 ready for the model
    """
    # PIL → numpy BGR (as cv2 would load it) → convert to RGB
    image_np = np.array(pil_image.convert("RGB"))          # H×W×3 uint8

    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2, 2))
        enhanced = clahe.apply(image_np[:, :, 0])
        image_np = np.stack([enhanced] * 3, axis=2)

    res      = transforms_val(image=image_np)
    image_t  = res["image"].astype(np.float32).transpose(2, 0, 1) / 255.0  # 3×H×W

    original_resized = res["image"]                        # H×W×3 uint8 RGB
    tensor = torch.tensor(image_t).unsqueeze(0)            # 1×3×H×W
    return original_resized, tensor


# ─── Model ────────────────────────────────────────────────────────────────────
class SegModel(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.seg = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=None,   # weights loaded separately
            classes=2,
            activation=None,
        )

    def forward(self, x):
        feats    = self.seg.encoder(x)
        dec_out  = self.seg.decoder(feats)
        return self.seg.segmentation_head(dec_out)


def load_models() -> list[SegModel]:
    """Load all fold models. Returns empty list if weights are missing."""
    models = []
    for fold in range(N_FOLDS):
        weight_path = os.path.join(
            MODEL_DIR, f"{KERNEL_TYPE}_best_fold{fold}.pth"
        )
        if not os.path.exists(weight_path):
            print(f"[WARNING] Model not found: {weight_path}")
            continue
        model = SegModel(ENET_TYPE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
        model.to(DEVICE).eval()
        models.append(model)
        print(f"  Loaded fold {fold}: {weight_path}")
    return models


print("Loading model weights …")
MODELS = load_models()
if not MODELS:
    print("[WARNING] No model weights found — the app will run in DEMO mode "
          "(random predictions). Place .pth files in:", MODEL_DIR)


# ─── Inference ────────────────────────────────────────────────────────────────
def run_inference(tensor: torch.Tensor) -> np.ndarray:
    """
    Returns averaged sigmoid probabilities: shape (2, H, W) float32 in [0,1].
    Falls back to random noise when no weights are loaded (demo mode).
    """
    tensor = tensor.to(DEVICE)
    if MODELS:
        with torch.no_grad():
            preds = torch.stack(
                [m(tensor).sigmoid() for m in MODELS], dim=0
            ).mean(0)                          # 1×2×H×W
        return preds[0].cpu().numpy()          # 2×H×W
    else:
        # Demo: random smooth blob so UI is still useful
        rng  = np.random.default_rng(42)
        base = rng.random((2, IMAGE_SIZE // 8, IMAGE_SIZE // 8)).astype(np.float32)
        out  = np.array([
            cv2.resize(ch, (IMAGE_SIZE, IMAGE_SIZE)) for ch in base
        ])
        return (out > 0.55).astype(np.float32)


# ─── Overlay helpers ──────────────────────────────────────────────────────────
def make_mask_rgb(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Combine two-channel probability map into a colour mask image (RGB uint8).
    Channel 0 (clip) → green, Channel 1 (tip) → red.
    """
    h, w  = prob_map.shape[1], prob_map.shape[2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for ch_idx, color in enumerate(CHANNEL_COLORS):
        binary = (prob_map[ch_idx] >= threshold).astype(np.uint8)
        for c in range(3):
            canvas[:, :, c][binary == 1] = color[c]

    return canvas


def make_overlay(original_rgb: np.ndarray,
                 prob_map: np.ndarray,
                 threshold: float = 0.5,
                 alpha: float = 0.45) -> np.ndarray:
    """Blend mask colours onto the original image."""
    overlay = original_rgb.copy().astype(np.float32)
    for ch_idx, color in enumerate(CHANNEL_COLORS):
        binary = (prob_map[ch_idx] >= threshold)
        for c in range(3):
            overlay[:, :, c][binary] = (
                overlay[:, :, c][binary] * (1 - alpha) + color[c] * alpha
            )
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ─── Main Gradio handler ───────────────────────────────────────────────────────
def segment(pil_image, threshold_pct: float, alpha_pct: float):
    if pil_image is None:
        return None, None, None, "⚠️ Please upload an image."

    threshold = threshold_pct / 100.0
    alpha = alpha_pct / 100.0

    # 1. Preprocess
    original_np, tensor = preprocess(pil_image)

    # 2. Inference
    prob_map = run_inference(tensor)  # 2×H×W

    # 3. 取得二值矩陣並儲存 (用於後續辨識模型訓練)
    # 確保資料夾存在
    os.makedirs("output_masks", exist_ok=True)
    
    import time
    timestamp = int(time.time() * 1000) # 使用毫秒增加唯一性
    
    # 轉為 0 或 255 的二值圖
    clip_mask = (prob_map[0] >= threshold).astype(np.uint8) * 255
    tip_mask  = (prob_map[1] >= threshold).astype(np.uint8) * 255 
    
    # 儲存為訓練用的二值標籤
    cv2.imwrite(f"output_masks/clip_{timestamp}.png", clip_mask)
    cv2.imwrite(f"output_masks/tip_{timestamp}.png", tip_mask)

    # 4. Build output images (Gradio 畫面顯示用)
    mask_rgb  = make_mask_rgb(prob_map, threshold)
    overlay   = make_overlay(original_np, prob_map, threshold, alpha)

    # 5. Stats
    clip_pct = float((prob_map[0] >= threshold).mean() * 100)
    tip_pct  = float((prob_map[1] >= threshold).mean() * 100)
    mode_str = f"{len(MODELS)}-fold ensemble" if MODELS else "DEMO (no weights)"
    status = (
        f"**Mode:** {mode_str} | **Device:** {DEVICE}\n\n"
        f"**檔案已儲存至:** `output_masks/` 資料夾\n\n"
        f"**Clip mask coverage:** {clip_pct:.2f} % (green)\n\n"
        f"**Tip mask coverage:** {tip_pct:.2f} % (red)"
    )

    return (
        Image.fromarray(original_np),
        Image.fromarray(mask_rgb),
        Image.fromarray(overlay),
        status,
    )

# ─── UI ───────────────────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; }
.output-label { font-weight: bold; font-size: 0.95rem; }
"""

with gr.Blocks(css=CSS, title="Catheter Segmentation") as demo:
    gr.Markdown(
        "# 🏥 Catheter Segmentation\n"
        "Upload a chest X-ray. The model will predict **clip** (green) "
        "and **tip** (red) masks using a UNet++ EfficientNet-B5 ensemble.\n\n"
        "> **Preprocessing:** CLAHE optional → Albumentations `Resize(1024, 1024)` → normalize ÷255",
        elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="pil",
                label="Upload Chest X-Ray",
                image_mode="RGB",
            )
            with gr.Accordion("⚙️ Settings", open=False):
                threshold_slider = gr.Slider(
                    minimum=1, maximum=99, value=50, step=1,
                    label="Mask threshold (%)",
                    info="Sigmoid probability threshold to binarise the mask",
                )
                alpha_slider = gr.Slider(
                    minimum=10, maximum=90, value=45, step=5,
                    label="Overlay opacity (%)",
                    info="How strongly the mask colour is blended onto the image",
                )
            run_btn = gr.Button("▶ Run Segmentation", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                out_original = gr.Image(label="① Original (resized)", elem_classes="output-label")
                out_mask     = gr.Image(label="② Predicted Mask",      elem_classes="output-label")
                out_overlay  = gr.Image(label="③ Overlay",             elem_classes="output-label")
            status_box = gr.Markdown("*Results will appear here after running.*")

    run_btn.click(
        fn=segment,
        inputs=[img_input, threshold_slider, alpha_slider],
        outputs=[out_original, out_mask, out_overlay, status_box],
    )
    img_input.change(          # auto-run on upload
        fn=segment,
        inputs=[img_input, threshold_slider, alpha_slider],
        outputs=[out_original, out_mask, out_overlay, status_box],
    )

    gr.Examples(
        examples=[],           # add local image paths here for quick demos
        inputs=img_input,
    )

    gr.Markdown(
        "---\n"
        "**Legend:** 🟢 Green = catheter clip body &nbsp;|&nbsp; 🔴 Red = catheter tip\n\n"
        "Place model weights (`*_best_fold{0-4}.pth`) in `exp_result/Jun__2_00_34_39_2025/model/` "
        "to enable real inference."
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=12355)
