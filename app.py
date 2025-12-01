"""
===============================================================================
Image Captioning and Segmentation System
===============================================================================

This project integrates two advanced computer vision tasks into a unified system:

1. **Image Captioning**  
   Uses a Transformer-based decoder model trained on COCO captions to generate
   natural-language descriptions of input images. The model processes extracted
   CNN features and sequentially predicts caption tokens based on a learned
   vocabulary.

2. **Semantic Segmentation + Object Detection**  
   Utilizes a Mask R-CNN (ResNet50 FPN) model to detect objects in an image,
   generate instance-level segmentation masks, and draw bounding boxes around
   top-scoring detections.

The goal of this system is to demonstrate how deep learning models can work
together to provide a richer understanding of visual data—combining pixel-level
segmentation, object identification, and high-level caption generation.

A FastAPI backend is used to run inference with both models, and a modern web
dashboard allows users to upload images and view results such as:
- Original image
- Segmentation mask
- Bounding box image
- Generated caption
- Top detected classes with confidence scores

This project is built for educational, research, and demonstration purposes.

-------------------------------------------------------------------------------
Author
-------------------------------------------------------------------------------
Name: **Supriya Mandal, Madana Venkatesh & Biki Haldar**  
GitHub: https://github.com/MSupriya4223  
Year: 2025  

If you use this code or extend it, please give proper credit to the author.
===============================================================================
"""
# app.py
import os
import io
import math
from pathlib import Path
from typing import List, Tuple, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models

# ---------------------------
# COCO classes (index 0 = background)
# (same as in your working script)
# ---------------------------
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
    "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table",
    "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# ---------------------------
# Utilities (copied/kept from your working script)
# ---------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def generate_class_colors(class_names):
    np.random.seed(42)
    colors = {}
    for name in class_names:
        c = tuple(int(x) for x in np.random.randint(0, 256, 3))
        colors[name] = c
    return colors

def get_top_k_classes(pred_classes, pred_scores, K=10):
    class_score = {}
    for cls, score in zip(pred_classes, pred_scores):
        class_score[cls] = class_score.get(cls, 0.0) + float(score)
    sorted_items = sorted(class_score.items(), key=lambda x: x[1], reverse=True)
    return [cls for cls, _ in sorted_items[:K]], sorted_items

def create_colored_mask_single(mask_bin, color):
    h, w = mask_bin.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[mask_bin == 1] = color
    return colored

# ---------------------------
# Captioning model code (from your script)
# ---------------------------
class Vocabulary:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.avgpool(feats).view(feats.size(0), -1)
        emb = self.fc(pooled)
        return emb

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, nhead=8,
                 num_layers=4, max_len=30):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)

        layer = nn.TransformerDecoderLayer(embed_dim, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers)

        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.embed_dim = embed_dim

    def create_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), 1).bool()

    def forward(self, tgt, memory):
        B, T = tgt.size()
        emb = self.embed(tgt) * math.sqrt(self.embed_dim)
        emb = self.pos_enc(emb)
        mask = self.create_mask(T, tgt.device)
        out = self.transformer_decoder(emb, memory.unsqueeze(1), tgt_mask=mask)
        return self.fc_out(out)

class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderTransformer(vocab_size, embed_dim)

    def forward(self, images, captions):
        enc = self.encoder(images)
        return self.decoder(captions[:, :-1], enc)

def greedy_decode(model, img_tensor, vocab, max_len=30, device="cpu"):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        memory = model.encoder(img_tensor)

    sos = vocab.word2idx["<sos>"]
    eos = vocab.word2idx["<eos>"]

    cur = torch.tensor([[sos]], device=device)
    result = [sos]

    for _ in range(max_len):
        with torch.no_grad():
            logits = model.decoder(cur, memory)

        next_token = logits[0, -1].argmax().item()
        result.append(next_token)

        if next_token == eos:
            break

        cur = torch.cat([cur, torch.tensor([[next_token]], device=device)], dim=1)

    caption = [vocab.idx2word[t] for t in result if t not in [sos, eos, "<pad>"]]
    return " ".join(caption)

# ---------------------------
# FastAPI app + static
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# static folder & tmp
ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
STATIC_TMP = STATIC_DIR / "tmp"
ensure_dir(STATIC_TMP)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------
# Checkpoint paths (change if needed)
# ---------------------------
SEG_CHECKPOINT = "checkpoints/maskrcnn_epoch1.pth"
CAPTION_CHECKPOINT = "checkpoints/caption_epoch19.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Globals for models
_seg_model = None
_caption_model = None
_caption_vocab = None

_caption_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------------------------
# Model loaders
# ---------------------------
def load_seg_model(ckpt_path: str):
    """
    Load Mask R-CNN (maskrcnn_resnet50_fpn) and restore checkpoint robustly,
    using strict=False to allow minor mismatch keys (same as your working script).
    """
    global _seg_model
    print("Loading segmentation checkpoint:", ckpt_path)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    # robust load
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("[load] Missing keys (may be okay):", missing)
        if unexpected:
            print("[load] Unexpected keys (ignored):", unexpected)
        print("Segmentation checkpoint loaded (strict=False).")
    else:
        print("Segmentation checkpoint not found at:", ckpt_path, " — using uninitialized model.")
    model.to(device)
    model.eval()
    _seg_model = model

def load_captioning_model(ckpt_path: str):
    global _caption_model, _caption_vocab
    print("Loading captioning checkpoint:", ckpt_path)
    if not os.path.exists(ckpt_path):
        print("Caption checkpoint not found at:", ckpt_path)
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocabulary(ckpt["vocab"]["word2idx"])
    vocab_size = len(vocab.word2idx)
    model = CaptioningModel(vocab_size)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    _caption_model = model
    _caption_vocab = vocab
    print("Caption model loaded.")

# Load models at startup if available
@app.on_event("startup")
def startup_event():
    if os.path.exists(SEG_CHECKPOINT):
        try:
            load_seg_model(SEG_CHECKPOINT)
        except Exception as e:
            print("Error loading seg model:", e)
    else:
        print("Segmentation checkpoint not found:", SEG_CHECKPOINT)

    if os.path.exists(CAPTION_CHECKPOINT):
        try:
            load_captioning_model(CAPTION_CHECKPOINT)
        except Exception as e:
            print("Error loading caption model:", e)
    else:
        print("Caption checkpoint not found:", CAPTION_CHECKPOINT)

# ---------------------------
# Helper: write image to static/tmp with fixed names
# ---------------------------
def save_outputs_to_static_tmp(original_pil: Image.Image, colored_mask_np: np.ndarray, boxes_np_rgb: np.ndarray, detection_summary_lines: List[str]):
    """
    Save outputs with the exact filenames expected by your frontend:
      - original.png
      - colored_mask.png
      - boxes.png
      - detection_summary.txt
    Overwrites previous files each run (keeps frontend simple).
    """
    # ensure dir exists
    STATIC_TMP.mkdir(parents=True, exist_ok=True)

    orig_path = STATIC_TMP / "original.png"
    colored_path = STATIC_TMP / "colored_mask.png"
    boxes_path = STATIC_TMP / "boxes.png"
    summary_path = STATIC_TMP / "detection_summary.txt"

    # Save original
    original_pil.save(str(orig_path))

    # Save colored_mask (np RGB) -> PIL
    colored_pil = Image.fromarray(colored_mask_np)
    colored_pil.save(str(colored_path))

    # Save boxes image (np RGB) -> PIL
    boxes_pil = Image.fromarray(boxes_np_rgb)
    boxes_pil.save(str(boxes_path))

    # Save detection summary lines
    with open(str(summary_path), "w") as f:
        f.write("Detected classes ranked by accumulated confidence (top first):\n")
        for line in detection_summary_lines:
            f.write(line + "\n")

    return {
        "original": orig_path.name,
        "colored_mask": colored_path.name,
        "boxes": boxes_path.name,
        "detection_summary": summary_path.name
    }

# ---------------------------
# Core segmentation logic integrated (from infer_seg_custom.py)
# ---------------------------
def run_segmentation_and_visualize(pil_img: Image.Image, seg_checkpoint: str, top_k: int = 5, conf_thresh: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Runs the Mask R-CNN model (already loaded in _seg_model) on the provided PIL image.
    Returns:
      - colored_mask (HxWx3 np.uint8) RGB
      - boxes_image (HxWx3 np.uint8) RGB (with bottom strip and top-K labels)
      - detection_summary_lines (list of "label\\tvalue" strings)
    """
    if _seg_model is None:
        raise RuntimeError("Segmentation model not loaded on server.")

    img_np = np.array(pil_img)  # HxWx3 RGB
    img_vis = img_np.copy()
    H, W = img_np.shape[:2]

    # preprocess (same as your script)
    transform = torchvision.transforms.ToTensor()
    tensor = transform(pil_img).to(device).unsqueeze(0)

    # forward
    with torch.no_grad():
        outputs = _seg_model(tensor)[0]

    boxes_all = outputs["boxes"].cpu().numpy()  # Nx4
    masks_all = outputs["masks"].cpu().numpy()  # Nx1HxW
    scores_all = outputs["scores"].cpu().numpy()
    labels_idx_all = outputs["labels"].cpu().numpy()
    labels_all = [COCO_CLASSES[i] if i < len(COCO_CLASSES) else str(i) for i in labels_idx_all]

    # filter by conf threshold
    keep_mask = scores_all >= conf_thresh
    boxes = boxes_all[keep_mask]
    masks = masks_all[keep_mask][:, 0, :, :] if masks_all.size else np.zeros((0, H, W), dtype=np.uint8)
    scores = scores_all[keep_mask]
    labels_idx = labels_idx_all[keep_mask]
    labels = [labels_all[i] for i, v in enumerate(keep_mask) if v]

    if len(boxes) == 0:
        # return blank colored mask and the original image with a small note
        blank_mask = np.zeros((H, W, 3), dtype=np.uint8)
        vis = img_vis.copy()
        # create small bottom strip with "No detections"
        strip_h = 80
        canvas = np.zeros((H + strip_h, W, 3), dtype=np.uint8)
        canvas[:H, :, :] = vis
        cv2.putText(canvas, "No detections above confidence threshold.", (10, H + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        # detection summary empty
        return blank_mask, canvas, ["No detections\t0.0000"]

    # rank classes and pick top-K classes
    top_classes, ranked_list = get_top_k_classes(labels, scores, K=top_k)

    # consistent class colors
    class_colors = generate_class_colors(COCO_CLASSES)

    # 1) Gray mask (merged) — not returned, but kept for parity
    gray_mask = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        binm = (m > 0.5).astype(np.uint8)
        gray_mask = np.maximum(gray_mask, binm * 255)

    # 2) Colored mask overlay (each instance colored by class; merged by max)
    colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for m, lbl_idx in zip(masks, labels_idx):
        binm = (m > 0.5).astype(np.uint8)
        cls_name = COCO_CLASSES[lbl_idx] if lbl_idx < len(COCO_CLASSES) else str(lbl_idx)
        color = class_colors.get(cls_name, (255, 255, 255))
        inst_col = create_colored_mask_single(binm, color)
        colored_mask = np.maximum(colored_mask, inst_col)

    # 3) Bounding boxes + labels for top-K classes (avoid overlapping labels)
    vis = img_vis.copy()
    placed_label_rects = []

    for box, lbl_idx, score in zip(boxes, labels_idx, scores):
        cls_name = COCO_CLASSES[lbl_idx] if lbl_idx < len(COCO_CLASSES) else str(lbl_idx)
        if cls_name not in top_classes:
            continue
        color = class_colors.get(cls_name, (0, 255, 0))
        x1, y1, x2, y2 = box.astype(int).tolist()
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)

    # Add black strip below image for class names (same layout as your script)
    strip_h = 80
    final_h = H + strip_h
    final_w = W
    canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    canvas[:H, :, :] = vis.copy()

    unique_top = list(dict.fromkeys(top_classes))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    x_cursor = 10
    y_text = H + 50

    for cls_name in unique_top:
        color = class_colors.get(cls_name, (255, 255, 255))
        (tw, th), _ = cv2.getTextSize(cls_name, font, font_scale, thickness)
        cv2.putText(canvas, cls_name, (x_cursor, y_text), font, font_scale, color, thickness, cv2.LINE_AA)
        x_cursor += tw + 40

    # Detection summary lines (label \t total_score)
    detection_summary_lines = []
    for cls_name, total in ranked_list:
        detection_summary_lines.append(f"{cls_name}\t{total:.4f}")

    return colored_mask, canvas, detection_summary_lines

# ---------------------------
# Endpoint: /predict
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5, conf: float = 0.5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await file.read()
    pil = Image.open(io.BytesIO(contents)).convert("RGB")

    # Save original directly to static/tmp/original.png
    # (frontend will load /static/tmp/original.png)
    original_path = STATIC_TMP / "original.png"
    pil.save(str(original_path))

    # Run segmentation (uses model loaded at startup)
    try:
        colored_mask_np, boxes_np_rgb, detection_summary_lines = run_segmentation_and_visualize(
            pil_img=pil,
            seg_checkpoint=SEG_CHECKPOINT,
            top_k=top_k,
            conf_thresh=conf
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    # Run captioning if available
    if _caption_model is None or _caption_vocab is None:
        caption_text = "Caption model not loaded on server."
    else:
        try:
            inp = _caption_transform(pil).to(device)
            caption_text = greedy_decode(_caption_model, inp, _caption_vocab, device=device)
        except Exception as e:
            caption_text = f"Captioning failed: {e}"

    # Save outputs to static/tmp with expected filenames
    saved = save_outputs_to_static_tmp(pil, colored_mask_np, boxes_np_rgb, detection_summary_lines)

    # Parse detection_summary_lines into structured list
    top_classes_list: List[Dict[str, float]] = []
    for line in detection_summary_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            # last token is score (tab separated in file but we split by whitespace)
            score_str = parts[-1]
            label = " ".join(parts[:-1])
            try:
                score = float(score_str)
            except:
                score = 0.0
        else:
            label = parts[0] if parts else "unknown"
            score = 0.0
        top_classes_list.append({"label": label, "score": score})

    # Return basenames (frontend should request /static/tmp/<basename>) and parsed top_classes
    return JSONResponse({
        "original": saved["original"],
        "colored_mask": saved["colored_mask"],
        "boxes": saved["boxes"],
        "detection_summary": saved["detection_summary"],
        "top_classes": top_classes_list,
        "caption": caption_text
    })

# ---------------------------
# Simple index route (optional)
# ---------------------------

@app.post("/stage_tmp")
async def stage_tmp(data: dict):
    return {"message": "OK"}

@app.get("/")
def index():
    # If you have a static index page, you can return it. Otherwise, just a simple message.
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Segmentation + Captioning API running."}

