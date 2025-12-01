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
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision

# ---------------------------
# COCO classes (index 0 = background)
# ---------------------------
COCO_CLASSES = [
    "__background__",     # 0
    "person",             # 1
    "bicycle",            # 2
    "car",                # 3
    "motorcycle",         # 4
    "airplane",           # 5
    "bus",                # 6
    "train",              # 7
    "truck",              # 8
    "boat",               # 9
    "traffic light",      # 10
    "fire hydrant",       # 11
    "street sign",        # 12
    "stop sign",          # 13
    "parking meter",      # 14
    "bench",              # 15
    "bird",               # 16
    "cat",                # 17
    "dog",                # 18
    "horse",              # 19
    "sheep",              # 20
    "cow",                # 21
    "elephant",           # 22
    "bear",               # 23
    "zebra",              # 24
    "giraffe",            # 25
    "hat",                # 26  (not used)
    "backpack",           # 27
    "umbrella",           # 28
    "shoe",               # 29  (not used)
    "eye glasses",        # 30  (not used)
    "handbag",            # 31
    "tie",                # 32
    "suitcase",           # 33
    "frisbee",            # 34
    "skis",               # 35
    "snowboard",          # 36
    "sports ball",        # 37
    "kite",               # 38
    "baseball bat",       # 39
    "baseball glove",     # 40
    "skateboard",         # 41
    "surfboard",          # 42
    "tennis racket",      # 43
    "bottle",             # 44
    "plate",              # 45  (not used)
    "wine glass",         # 46
    "cup",                # 47
    "fork",               # 48
    "knife",              # 49
    "spoon",              # 50
    "bowl",               # 51
    "banana",             # 52
    "apple",              # 53
    "sandwich",           # 54
    "orange",             # 55
    "broccoli",           # 56
    "carrot",             # 57
    "hot dog",            # 58
    "pizza",              # 59
    "donut",              # 60
    "cake",               # 61
    "chair",              # 62
    "couch",              # 63
    "potted plant",       # 64
    "bed",                # 65
    "mirror",             # 66  (not used)
    "dining table",       # 67
    "window",             # 68  (not used)
    "desk",               # 69  (not used)
    "toilet",             # 70
    "door",               # 71  (not used)
    "tv",                 # 72
    "laptop",             # 73
    "mouse",              # 74
    "remote",             # 75
    "keyboard",           # 76
    "cell phone",         # 77
    "microwave",          # 78
    "oven",               # 79
    "toaster",            # 80
    "sink",               # 81
    "refrigerator",       # 82
    "blender",            # 83  (not used)
    "book",               # 84
    "clock",              # 85
    "vase",               # 86
    "scissors",           # 87
    "teddy bear",         # 88
    "hair drier",         # 89
    "toothbrush"          # 90
]

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def generate_class_colors(class_names):
    """Deterministic color per class name."""
    np.random.seed(42)
    colors = {}
    for name in class_names:
        c = tuple(int(x) for x in np.random.randint(0, 256, 3))
        colors[name] = c
    return colors

def get_top_k_classes(pred_classes, pred_scores, K=10):
    """Accumulate total score per class and return top-K class names."""
    class_score = {}
    for cls, score in zip(pred_classes, pred_scores):
        class_score[cls] = class_score.get(cls, 0.0) + float(score)
    sorted_items = sorted(class_score.items(), key=lambda x: x[1], reverse=True)
    return [cls for cls, _ in sorted_items[:K]], sorted_items

def create_colored_mask_single(mask_bin, color):
    """Create an RGB image where mask_bin==1 gets color, else 0."""
    h, w = mask_bin.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[mask_bin == 1] = color
    return colored

def rects_overlap(a, b):
    """Check if rect a intersects rect b.
    rect = (x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def place_label_non_overlapping(existing_boxes, x1, y1, w_text, h_text, img_h):
    """
    Try to place label rectangle above the box; if it collides with any existing box
    move it downward step by step until it fits or reaches image bottom.
    Returns (tx, ty) top-left coordinate for text.
    """
    # try above
    tx = x1
    ty = max(0, y1 - h_text - 6)
    # clamp
    if ty < 0:
        ty = 0
    # candidate rect
    cand = (tx, ty, tx + w_text, ty + h_text)
    tries = 0
    while any(rects_overlap(cand, eb) for eb in existing_boxes) and tries < 20:
        ty = ty + h_text + 6  # move down
        if ty + h_text > img_h:
            ty = max(0, img_h - h_text - 1)
            break
        cand = (tx, ty, tx + w_text, ty + h_text)
        tries += 1
    return tx, ty, cand

# ---------------------------
# Main inference function
# ---------------------------
def infer_and_visualize(image_path: str, seg_checkpoint: str, out_dir: str, top_k: int = 5, conf_thresh: float = 0.5, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(out_dir)

    # build model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    model.to(device)

    # load checkpoint robustly
    ckpt = torch.load(seg_checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt
    # load with strict=False to allow minor mismatches (prints returned missing/unexpected)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[load] Missing keys (may be okay):", missing)
    if unexpected:
        print("[load] Unexpected keys (ignored):", unexpected)
    model.eval()

    # load image
    pil = Image.open(image_path).convert("RGB")
    img_np = np.array(pil)  # HxWx3 (RGB)
    img_vis = img_np.copy()
    H, W = img_np.shape[:2]

    # preprocess
    transform = torchvision.transforms.ToTensor()
    tensor = transform(pil).to(device).unsqueeze(0)

    # forward
    with torch.no_grad():
        outputs = model(tensor)[0]

    boxes_all = outputs["boxes"].cpu().numpy()  # Nx4
    masks_all = outputs["masks"].cpu().numpy()  # Nx1HxW
    scores_all = outputs["scores"].cpu().numpy()
    labels_idx_all = outputs["labels"].cpu().numpy()
    labels_all = [COCO_CLASSES[i] if i < len(COCO_CLASSES) else str(i) for i in labels_idx_all]

    # filter by conf threshold
    keep_mask = scores_all >= conf_thresh
    boxes = boxes_all[keep_mask]
    masks = masks_all[keep_mask][:, 0, :, :]  # NxHxW
    scores = scores_all[keep_mask]
    labels_idx = labels_idx_all[keep_mask]
    labels = [labels_all[i] for i, v in enumerate(keep_mask) if v]

    if len(boxes) == 0:
        print("No detections above confidence threshold.")
        return

    # rank classes and pick top-K classes
    top_classes, ranked_list = get_top_k_classes(labels, scores, K=top_k)

    print("\nDetected classes ranked by total confidence (top first):")
    for cls_name, total_score in ranked_list[: top_k]:
        print(f"  {cls_name}: {total_score:.3f}")
    print()

    # consistent class colors
    class_colors = generate_class_colors(COCO_CLASSES)

    # 1) Gray mask (merged)
    gray_mask = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        binm = (m > 0.5).astype(np.uint8)
        gray_mask = np.maximum(gray_mask, binm * 255)
    gray_out = os.path.join(out_dir, "output_gray_mask.png")
    cv2.imwrite(gray_out, gray_mask)
    print("Saved:", gray_out)

    # 2) Colored mask overlay (each instance colored by class; merged by max)
    colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for m, lbl_idx in zip(masks, labels_idx):
        binm = (m > 0.5).astype(np.uint8)
        cls_name = COCO_CLASSES[lbl_idx] if lbl_idx < len(COCO_CLASSES) else str(lbl_idx)
        color = class_colors.get(cls_name, (255, 255, 255))
        # create colored instance and merge (max)
        inst_col = create_colored_mask_single(binm, color)
        colored_mask = np.maximum(colored_mask, inst_col)
    colored_out = os.path.join(out_dir, "output_colored_mask.png")
    # convert RGB->BGR for cv2.imwrite
    cv2.imwrite(colored_out, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    print("Saved:", colored_out)

    # 3) Bounding boxes + labels for top-K classes (avoid overlapping labels)
    vis = img_vis.copy()
    placed_label_rects = []  # list of rectangles already placed (x1,y1,x2,y2)

    # prepare text sizing
    for box, lbl_idx, score in zip(boxes, labels_idx, scores):
        cls_name = COCO_CLASSES[lbl_idx] if lbl_idx < len(COCO_CLASSES) else str(lbl_idx)
        if cls_name not in top_classes:
            continue  # skip classes outside top-K

        color = class_colors.get(cls_name, (0, 255, 0))
        x1, y1, x2, y2 = box.astype(int).tolist()
        # draw rectangle
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)

        # # label text
        # label_text = f"{cls_name}: {score:.2f}"
        # # compute text size
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.6
        # thickness = 2
        # (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        # text_h = text_h + baseline

        # # place label on top-left of box, but avoid overlaps
        # tx, ty, rect = place_label_non_overlapping(placed_label_rects, x1, y1, text_w + 6, text_h + 4, H)
        # # draw filled rectangle as background for text (for readability)
        # bx1, by1, bx2, by2 = rect
        # # ensure int and inside image
        # bx1, by1, bx2, by2 = map(int, (max(0, bx1), max(0, by1), min(W - 1, bx2), min(H - 1, by2)))
        # cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, thickness=-1)  # filled
        # # put text in contrasting color (choose black/white based on brightness)
        # brightness = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
        # text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        # txt_x = bx1 + 3
        # txt_y = by1 + text_h - 2
        # cv2.putText(vis, label_text, (txt_x, txt_y), font, font_scale, text_color, thickness=1, lineType=cv2.LINE_AA)
        # placed_label_rects.append((bx1, by1, bx2, by2))


    # ---------------------------------------------
    # Add black strip below image for class names
    # ---------------------------------------------

    # Height of the strip (you can increase if needed)
    strip_h = 80

    # Create new image with extra black space
    final_h = H + strip_h
    final_w = W
    canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    # Place the original bbox image at the top
    canvas[:H, :, :] = vis.copy()

    # Prepare list of top-K class names (NO SCORES)
    unique_top = list(dict.fromkeys(top_classes))  # remove duplicates but keep order

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    x_cursor = 10
    y_text = H + 50  # middle of black strip

    for cls_name in unique_top:
        color = class_colors.get(cls_name, (255, 255, 255))

        # compute width of the label
        (tw, th), _ = cv2.getTextSize(cls_name, font, font_scale, thickness)

        # write name in same color as bbox
        cv2.putText(canvas, cls_name, (x_cursor, y_text),
                    font, font_scale, color, thickness, cv2.LINE_AA)

        x_cursor += tw + 40  # space between labels



    bbox_out = os.path.join(out_dir, "output_bbox.png")
    # cv2 expects BGR
    # cv2.imwrite(bbox_out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(bbox_out, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    print("Saved:", bbox_out)

    # also produce a small text file listing top detected classes and their scores
    info_path = os.path.join(out_dir, "detection_summary.txt")
    with open(info_path, "w") as f:
        f.write("Detected classes ranked by accumulated confidence (top first):\n")
        for cls_name, total in ranked_list:
            f.write(f"{cls_name}\t{total:.4f}\n")
    print("Saved:", info_path)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Advanced Mask R-CNN inference producing masks + top-K bbox labels")
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--seg_checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs_infer", help="Directory to save outputs")
    p.add_argument("--top_k", type=int, default=5, help="Keep labels only for top-K classes by accumulated score")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    p.add_argument("--device", type=str, default=None, help="device to run on, e.g., cuda or cpu")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    infer_and_visualize(
        image_path=args.image_path,
        seg_checkpoint=args.seg_checkpoint,
        out_dir=args.out_dir,
        top_k=args.top_k,
        conf_thresh=args.conf,
        device=args.device
    )

