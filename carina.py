# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import gradio as gr #不需要撰寫複雜的 HTML、CSS 或 JavaScript，就能在 Python 中直接定義網頁元件
import torch
import torch.nn as nn #Neural Networks（神經網路）。 PyTorch 中專門建立、訓練、以及優化深度學習模型而設計的子庫（Sub-library）。
import albumentations #影像增強
import segmentation_models_pytorch as smp #基於 PyTorch 的開源套件，專門為影像分割任務提供了大量預先寫好的模型架構、編碼器與損失函數。
import torchxrayvision as xrv #定位 Carina (氣管分叉點)
from PIL import Image
import pandas as pd
import time

# ─── Configuration ────────────────
IMAGE_SIZE  = 1024
KERNEL_TYPE = "unet++b5_2cbce_1024T15tip_lr1e4_bs4_augv2_30epo_cvc" #模型架構與骨幹(unet++ EfficientNet-B5)、損失函數2-channel Binary Cross Entropy (2-channel BCE)、與影像1024
#繪製了一個半徑為 15 像素的圓形區域作為標籤、學習率1*10^{-4}、批次大小4、數據增強、訓練輪次 (Epochs)30

ENET_TYPE   = "timm-efficientnet-b5" #backbone :EfficientNet-B5
MODEL_DIR   = "C:/WorkArea/zoey/unet/model" 
N_FOLDS     = 5 #5-Fold 交叉驗證(參與集成預測的模型總數/資料切分的份數)
CSV_PATH    = "output_masks/coordinates_with_labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Preprocessing ────────────────────────────────────────────────────────────
transforms_val = albumentations.Compose([albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE)]) #縮放大小
#將原始的圖片檔案格式，轉換為模型可以運算的數學張量
def preprocess(pil_image: Image.Image) -> tuple[np.ndarray, torch.Tensor]: #型別提示(函式預期接收什麼樣的輸入，並會回傳什麼樣的結果。)
    image_np = np.array(pil_image.convert("RGB")) #將讀入的 PIL 影像強制轉換為 RGB 三通道模式，將 PIL 物件轉換為 NumPy 陣列。
    res = transforms_val(image=image_np) #統一影像尺寸
    image_t = res["image"].astype(np.float32).transpose(2, 0, 1) / 255.0 #將像素值從 8 位元整數（0-255）轉為 32 位元浮點數，
    tensor = torch.tensor(image_t).unsqueeze(0) #將NumPy 陣列轉換為 PyTorch 的張量（Tensor）物件，在第 0 維（最前面）增加一個大小為 1 的新維度。[bchw]
    return res["image"], tensor #調整尺寸後的原始影像矩陣。給 AI 模型（Unet++）進行數學運算的結構

# ─── Helper: Make Overlay ───────據預測的機率圖（prob_map）覆蓋上一層半透明的彩色遮罩──────────
def make_overlay(image_np, prob_map, threshold, alpha):
    overlay = image_np.copy() #複製影像
    mask_colors = [(0, 255, 0), (255, 0, 0)] # Green: Clip, Red: Tip
    for i in range(2): #迴圈會依序處理這兩項特徵
        mask = (prob_map[i] >= threshold).astype(np.uint8) #將機率圖（值為 0~1 的浮點數）與門檻值（例如 0.5）進行比較，產生一個布林值矩陣（True/False）。
        color_mask = np.zeros_like(image_np)  #建立一個跟原圖大小完全相同、但全是黑色（數值全為 0）的畫布。
        color_mask[:] = mask_colors[i] #把 mask_colors[i] 這個顏色，填滿 color_mask 畫布裡的每一個像素點
        idx = mask > 0 #找出遮罩中數值為 1 的所有位置（像素點）。
        overlay[idx] = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)[idx] #半透明疊加=目前的原始影像*原圖透明度+要疊加的顏色*透明度+不額外增亮
    return overlay

# ─── Model Class ──────────────────────────────────────────────────────────────
class SegModel(nn.Module): #繼承 nn.Module父類別管理神經網路模塊
    def __init__(self, backbone: str): #初始化定義這個模型結構
        super().__init__() #繼承父類別
        # 將接收到的 backbone 傳給 encoder_name
        self.seg = smp.UnetPlusPlus(encoder_name=backbone, encoder_weights=None, classes=2, activation=None) #U-Net++:捕捉 CVC 導管邊緣或尖端
    def forward(self, x):
        feats = self.seg.encoder(x) #特徵提取（編碼）
        dec_out = self.seg.decoder(feats) #特徵還原（解碼
        return self.seg.segmentation_head(dec_out) #最終分類（輸出頭）回傳logit
#用於集成學習、讀入5個權重檔
def load_models():
    models = [] #建立一個空的列表，用來存放載入完成的模型實例
    for fold in range(N_FOLDS):
        path = os.path.join(MODEL_DIR, f"{KERNEL_TYPE}_best_fold{fold}.pth") #構建權重檔案的路徑
        if os.path.exists(path):
            m = SegModel(ENET_TYPE) #實例SegModel（含有 U-Net++ 和 EfficientNet-B5 骨幹）
            m.load_state_dict(torch.load(path, map_location=DEVICE), strict=False) #讀取 .pth 權重字典。
            m.to(DEVICE).eval() #將模型切換至評估模式
            models.append(m) #將設定好的模型加入列表。
    return models

print("Loading UNet++ Models...")
MODELS = load_models()

print("Loading XRV Model (Universal Interface)...")
# 使用最通用的 DenseNet 模型，並指定使用具有 15 個特徵點輸出的權重
try:
    LANDMARK_MODEL = xrv.models.DenseNet(weights="all").to(DEVICE).eval() #指定載入在多個大型 X 光資料集（NIH, CheXpert, MIMIC 等）混合訓練出的最強通用權重。
    MODEL_MODE = "universal_densenet"
    print("Using Universal DenseNet (All weights).")
except Exception as e:
    print(f"Failed to load preferred model, trying fallback: {e}")
    # 最終備案：如果連權重都抓不到，則不執行解剖點偵測，避免程式崩潰
    LANDMARK_MODEL = None
    MODEL_MODE = "none"
    print("Warning: No Landmark Model loaded. Carina detection will be disabled.")

# ─── Logic Functions ─────mask-> points
def extract_path_points(mask, sample_step=15):
    """把預測出的綠色路徑（Clip Mask）轉化成一串座標點，並存入 CSV"""
    # 確保 mask 是二值化的
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) #將輸入的灰階遮罩（0-255）進行切分。像素值大於 127 的變成 255（純白），其餘變成 0（純黑）。回傳門檻值(不使用)、二值化影像矩陣
    # 找出所有非零像素點
    points = cv2.findNonZero(binary) #掃描整張影像，找出所有像素值為「白（非零）」的座標。
    if points is None:
        return "[]" #字串
    # 將[n,[x,y]]轉為列表格式 [(x, y), ...]
    pts = [p[0].tolist() for p in points]
    
    # 為了避免點太多導致 CSV 爆炸，進行等距採樣 (例如每 15 個點取一個)
    sampled_pts = pts[::sample_step] #list[start : stop : step]
    
    # 確保最後一個點 (通常是 Tip) 有被包含進去
    if pts[-1] not in sampled_pts:
        sampled_pts.append(pts[-1])
        
    return str(sampled_pts)

def detect_carina(image_np):
    """
    偵測 Carina 位置，整合 CLAHE 增強、模型預測、座標限制與解剖比例備案。
    """
    if LANDMARK_MODEL is None:
        # 備案：解剖學經驗位置 (水平置中，垂直約影像 40-45% 處)
        return int(0.42 * IMAGE_SIZE), int(0.5 * IMAGE_SIZE)

    try:
        # 1. 影像預處理：使用 CLAHE 增強氣管黑影對比度
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) #將彩色圖轉為灰階，因為解剖點偵測不需要顏色資訊。
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #限制對比度自適應直方圖均衡化
        enhanced_gray = clahe.apply(gray) #CLAHE 物件的方法（Method）。它負責執行實際的數學運算
        
        # 縮放與正規化
        resized = cv2.resize(enhanced_gray, (224, 224)).astype(np.float32) #將輸入的影像 enhanced_gray 強制調整為寬 224 像素、高 224 像素。
        xrv_img = (resized / 255.0) * 2048.0 - 1024.0 # 將0~255的像素值縮小到0~1（除以 255），接著映射到 -1024到 1024之間。
        xrv_tensor = torch.from_numpy(xrv_img).unsqueeze(0).unsqueeze(0).to(DEVICE) #將 NumPy 的 ndarray（矩陣）轉換成 PyTorch 的 Tensor（張量）。 再次在索引為 0的位置增加一個維度。
        with torch.no_grad(): #測（推論）階段，關閉梯度計算。
            preds = LANDMARK_MODEL(xrv_tensor) #是模型對該張 X 光片的特徵點預測
            
            # 2. 座標提取與邏輯檢查
            if preds.ndim == 3 and preds.shape[1] > 9:  # 確保是 Landmark 格式
                p_x = preds[0, 9, 0].item() #Carina（氣管分叉點） 座標
                p_y = preds[0, 9, 1].item() #把張量轉成 Python 的浮點數，方便後續計算。
                # 將 [-1, 1] 轉為 [0, 1] 比例
                car_x_norm = (p_x + 1.0) / 2.0 #將座標轉化為百分比（比例）。
                car_y_norm = (p_y + 1.0) / 2.0
                
                # 3. 座標範圍限制 (Coordinate Clipping):Carina 理論上應在影像中線附近 (0.4~0.6) 且在上半部 (0.3~0.55)
                #np.clip(目標值, 最小值, 最大值)
                car_x_norm = np.clip(car_x_norm, 0.35, 0.48) #在標準胸部 X 光中，氣管分叉點（Carina）通常位於影像中心線偏左一點
                car_y_norm = np.clip(car_y_norm, 0.48, 0.52) #Carina 大約位於胸腔高度的一半處（第四、五胸椎附近）。
                final_x = int(car_x_norm * IMAGE_SIZE) #將0~1的比例，乘上圖片的真實像素尺寸
                final_y = int(car_y_norm * IMAGE_SIZE)
                return final_x, final_y
            
            else:
                # 如果模型維度不符 (例如只輸出疾病分類)，回傳解剖比例位置
                print("Landmark shape mismatch, using fallback.")
                return int(0.42 * IMAGE_SIZE), int(0.5 * IMAGE_SIZE)
                
    except Exception as e:
        print(f"Landmark detection logic error: {e}")
        # 發生任何錯誤時，回傳最穩定的解剖參考點
        return int(0.42 * IMAGE_SIZE), int(0.5 * IMAGE_SIZE)
#計算導管尖端（Tip）與解剖基準點（Carina）的相對位置，並根據醫學臨床準則給出診斷標籤。
def check_malposition(tip_y, car_y, tip_x, car_x):
    if tip_y is None or car_y is None: return "Unknown", (128, 128, 128) #如果其中一個點沒抓到（None），系統會回傳 "Unknown" 並顯示灰色 (128, 128, 128)。
    y_diff = tip_y - car_y #縱向位移：決定深淺(>0導管在carina下方)
    x_diff = abs(tip_x - car_x) #橫向位移：決定是否誤入血管

    if x_diff > 180: #如果橫向位移>180像素（約 6 公分）
        return "Abnormal (Wrong Vessel)", (0, 0, 255) #回傳字串和RGB 數值
    if 0 <= y_diff <= 150: #如果縱向位移介於0~150(約為 Carina 下方4.5 ~5$ 公分)之間(https://www.droracle.ai/articles/1061589/is-a-right-central-venous-catheter-with-the-tip)
        return "Normal", (0, 255, 0)  
    elif (150 < y_diff <= 250) or (-50 <= y_diff < 0):
        return "Borderline", (0, 165, 255) 
    else:
        return "Abnormal (Too Deep/High)", (0, 0, 255)

def segment(pil_image, threshold_pct: float, alpha_pct: float):   #門檻值百分比、透明度百分比
    if not pil_image: return None, None, None, "⚠️ No image" #如果沒有影像回傳空值（Mask, Overlay, Label, Status）
    threshold, alpha = threshold_pct/100.0, alpha_pct/100.0 #將百分比轉為小數
    original_np, tensor = preprocess(pil_image) #將原始的圖片檔案格式，轉換為模型可以運算的數學張量(B,H, W, C)->(B, C,H, W)
#推論（Inference，即現在的診斷階段）不需計算梯度。
    with torch.no_grad():
        if MODELS: #UNet++ Models
        #輪流叫出每一個模型 m，進行預測，將一組原始數值（稱為 Logits）壓縮到 $[0, 1]$ 之間
        #將上述所有模型產生的機率圖垂直疊放在一起。進行平均計算((3, 1, 2, 1024, 1024)->(1, 2, 1024, 1024)->(2, 1024, 1024)
            prob_map = torch.stack([m(tensor.to(DEVICE)).sigmoid() for m in MODELS]).mean(0)[0].cpu().numpy()
        else:
            prob_map = np.zeros((2, IMAGE_SIZE, IMAGE_SIZE)) #建立一個數值全部都是0的 NumPy 矩陣。
    
    car_x, car_y = detect_carina(original_np) #偵測 Carina 位置，整合 CLAHE 增強、模型預測、座標限制與解剖比例備案。

    # 1. 提取尖端遮罩 (通道 1)
    tip_mask_raw = (prob_map[1] >= threshold).astype(np.uint8) * 255 #逐一檢查每個像素，回傳TRUE、FALSE->1、0， 乘以 255。
    clip_mask = (prob_map[0] >= threshold).astype(np.uint8) * 255 # 整個 CVC 導管的路徑
    
    # 2. 找出所有尖端連通域 (不只取最大的一個)
    contours, _ = cv2.findContours(tip_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Retrieve(檢索/取回) External(外部的) (Contours Only)、(簡單鏈式逼近演算法): 使用外部輪廓檢索(RETR_EXTERNAL)與鏈碼逼近(CHAIN_APPROX_SIMPLE)提取尖端候選區域
    valid_contours = [c for c in contours if cv2.contourArea(c) > 5]
    num_tips = len(valid_contours)
    
    clean_tip_mask = np.zeros_like(tip_mask_raw)
    all_tip_coords = [] # 用來存所有偵測到的尖端中心點

    for cnt in valid_contours:
        # 把所有有效的尖端都畫到 clean_tip_mask 上
        cv2.drawContours(clean_tip_mask, [cnt], -1, 255, -1)
        
        # 計算每個連通域的中心點
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            all_tip_coords.append([cx, cy])

    # 3. 診斷邏輯：如果有多個點，我們通常以「最深/最下方」的點作為與 Carina 比較的 Tip
    # 這裡假設 y 座標越大代表位置越深
    main_tip_x, main_tip_y = None, None
    if all_tip_coords:
        # 找 y 座標最大的點作為主要的尖端 (Tip)
        best_tip = max(all_tip_coords, key=lambda p: p[1])
        main_tip_x, main_tip_y = best_tip[0], best_tip[1]

    # 4. 提取路徑點位 (包含所有亮點區域的座標)
    full_data_path =extract_path_points(clip_mask, sample_step=20)

    # 診斷與視覺化
    label, color_bgr = check_malposition(main_tip_y, car_y, main_tip_x, car_x)
    overlay = make_overlay(original_np, prob_map, threshold, alpha)
    
    # 繪製 Carina
    if car_x and car_y:
        cv2.circle(overlay, (car_x, car_y), 10, (255, 255, 255), -1)
        cv2.circle(overlay, (car_x, car_y), 7, (0, 255, 255), -1) 
    
    # 繪製所有偵測到的尖端 (用藍色圈起來)
    for tx, ty in all_tip_coords:
        cv2.circle(overlay, (tx, ty), 10, (255, 255, 255), -1)
        cv2.circle(overlay, (tx, ty), 7, (255, 0, 0), -1) 

    mask_rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    mask_rgb[:, :, 1] = clip_mask  
    mask_rgb[:, :, 0] = clean_tip_mask  

    save_status = ""
    # 修改存檔條件：只要偵測到尖端就存檔，不論 1 個還是多個
    if num_tips >= 1:
        base_dir = r"C:\WorkArea\zoey\unet\output_masks"
        for d in ["masks", "overlays", "originals", "csv"]: os.makedirs(os.path.join(base_dir, d), exist_ok=True)
        ts = int(time.time() * 1000) #Python 的時間模組，回傳自 1970 年 1 月 1 日 00:00:00 (UTC) 以來的總秒數。
        mask_path = os.path.join(base_dir, "masks", f"mask_{ts}.png")
        over_path = os.path.join(base_dir, "overlays", f"overlay_{ts}_{label}.png")
        orig_path = os.path.join(base_dir, "originals", f"original_{ts}.png")
        csv_path = os.path.join(base_dir, "csv", "database.csv")
        
        cv2.imwrite(mask_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(over_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(orig_path, cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR))
        
        df_row = pd.DataFrame([{
            "StudyInstanceUID": ts, 
            "label": f"CVC - {label}", 
            "data": full_data_path, # 這裡會存入 [[x1,y1], [x2,y2]]
            "ts": ts, 
            "tx": main_tip_x, "ty": main_tip_y, # 主要 Tip
            "cx": car_x, "cy": car_y,
            "mask_file": mask_path, "overlay_file": over_path, "original_file": orig_path,
            "num_tips_detected": num_tips # 額外紀錄偵測到的點數
        }])
        df_row.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path), encoding='utf-8-sig')
        save_status = f"💾 已偵測到 {num_tips} 個尖端並存檔。"
    else:
        save_status = "⚠️ 未偵測到尖端，未存檔。"

    label_display = {f"CVC - {label}": 1.0}
    return Image.fromarray(mask_rgb), Image.fromarray(overlay), label_display, f"✅ 分析完成!\n{save_status}"

# ─── UI Layout ────────────────────────────────────────────────────────────────
 #[容器層開始] - 這是一張大地圖
def build_carina_page():
    gr.Markdown("# 🏥 CVC Position Auto-Diagnostic App")
    gr.Markdown("支援 **即時標籤顯示** 與 **自動分類存檔**")

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="上傳 CXR 影像")
            with gr.Row():
                sld_thr = gr.Slider(1, 99, value=50, label="閾值 Threshold (%)")
                sld_alp = gr.Slider(10, 90, value=45, label="透明度 Alpha (%)")
            run_btn = gr.Button("🔍 執行分析", variant="primary")

        with gr.Column(scale=1):
            out_label = gr.Label(label="診斷分類結果", num_top_classes=1)
            with gr.Tab("Overlay Result"):
                out_over = gr.Image(label="Overlay")
            with gr.Tab("Binary Mask"):
                out_mask = gr.Image(label="Mask")
            status = gr.Markdown("### 狀態: Ready.")

    run_btn.click(
        fn=segment,
        inputs=[img_in, sld_thr, sld_alp],
        outputs=[out_mask, out_over, out_label, status]
    )

    back_btn = gr.Button("⬅ 返回大廳", variant="secondary")
    return back_btn

if __name__ == "__main__":
    demo.launch(server_port=12356) # [啟動層] - 把地圖發布到網路上