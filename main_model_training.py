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
import sys
import json
import math
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO
from tqdm import tqdm

# Optional: nltk tokenization
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        import nltk
        nltk.download("punkt")
    except Exception:
        pass

# ---------------------------
# Global CONFIG & Utilities
# ---------------------------
DEFAULT_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "caption": {
        "max_len": 30,
        "embed_dim": 256,
        "nhead": 8,
        "num_layers": 4,
        "lr": 3e-4,
        "batch_size": 64,
    },
    "segmentation": {
        "lr": 1e-4,
        "batch_size": 4,
    },
    "num_workers": 4,
    "save_dir": "./checkpoints",
}

# Make deterministic-ish
random.seed(DEFAULT_CONFIG["seed"])
torch.manual_seed(DEFAULT_CONFIG["seed"])
np.random.seed(DEFAULT_CONFIG["seed"])


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(path: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None,
                    epoch: Optional[int] = None, extra: Optional[Dict] = None):
    data = {"model_state": model.state_dict()}
    if optimizer is not None:
        data["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch
    if extra is not None:
        data.update(extra)
    torch.save(data, path)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------
# Tokenizer & Vocabulary
# ---------------------------
class SimpleTokenizer:
    """A very simple tokenizer using nltk if available, else whitespace split."""
    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.strip()
        try:
            import nltk
            tokens = nltk.word_tokenize(text.lower())
        except Exception:
            tokens = text.lower().split()
        return tokens


class Vocabulary:
    """Vocabulary builder and converter."""

    def __init__(self, min_freq: int = 5, specials: List[str] = None):
        self.min_freq = min_freq
        if specials is None:
            specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.specials = specials
        self.word2idx = {w: i for i, w in enumerate(specials)}
        self.idx2word = {i: w for i, w in enumerate(specials)}
        self.freqs = Counter()
        self._frozen = False

    def add_sentence(self, sentence: str):
        tokens = SimpleTokenizer.tokenize(sentence)
        self.freqs.update(tokens)

    def build_vocab(self):
        if self._frozen:
            return
        for w, cnt in sorted(self.freqs.items(), key=lambda x: (-x[1], x[0])):
            if cnt >= self.min_freq:
                if w not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
        self._frozen = True

    def __len__(self):
        return len(self.word2idx)

    def numericalize(self, tokens: List[str]) -> List[int]:
        return [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]

    def detokenize(self, idxs: List[int]) -> List[str]:
        return [self.idx2word.get(i, "<unk>") for i in idxs]


# ---------------------------
# COCO Datasets
# ---------------------------
class COCOCaptionDataset(Dataset):
    """
    COCO caption dataset returning (image_tensor, caption_tensor)
    caption_tensor: LongTensor of length max_len containing <sos> ... <eos> padded with <pad>
    """

    def __init__(self, coco_root: str, images_folder: str, captions_json: str,
                 transforms_img=None, max_len: int = 30, subset_ids: Optional[List[int]] = None,
                 min_freq: int = 5, build_vocab: bool = True):
        self.coco_root = coco_root
        self.images_folder = images_folder
        self.captions_json = captions_json
        self.transforms_img = transforms_img
        self.max_len = max_len
        self.coco = COCO(captions_json)

        # choose image ids
        all_ids = list(self.coco.imgs.keys())
        if subset_ids is not None:
            # filter to only those IDs present in COCO (defensive)
            present = set(all_ids)
            self.ids = [iid for iid in subset_ids if iid in present]
        else:
            self.ids = all_ids

        # build caption map image_id -> [captions]
        self.imgid_to_caps = {}
        for ann in self.coco.anns.values():
            img_id = ann["image_id"]
            if img_id in self.ids:
                self.imgid_to_caps.setdefault(img_id, []).append(ann["caption"])

        # collect all captions to build vocab
        self.vocab = Vocabulary(min_freq)
        if build_vocab:
            for caps in self.imgid_to_caps.values():
                for c in caps:
                    self.vocab.add_sentence(c)
            self.vocab.build_vocab()

        # if some images have no captions, ensure they are excluded
        self.ids = [i for i in self.ids if i in self.imgid_to_caps]

    def __len__(self):
        return len(self.ids)

    def caption_to_ids(self, caption: str) -> torch.Tensor:
        tokens = SimpleTokenizer.tokenize(caption)
        tokens = tokens[: self.max_len - 2]  # reserve for sos and eos
        tokens2 = ["<sos>"] + tokens + ["<eos>"]
        ids = self.vocab.numericalize(tokens2)
        # pad
        if len(ids) < self.max_len:
            ids = ids + [self.vocab.word2idx["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # randomly pick one caption among 5
        caps = self.imgid_to_caps[img_id]
        caption = random.choice(caps)
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.coco_root, self.images_folder, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        if self.transforms_img:
            img = self.transforms_img(img)
        cap_ids = self.caption_to_ids(caption)
        return img, cap_ids, img_id


class COCOSegmentationDatasetForMaskRCNN(Dataset):
    """
    Prepares data for torchvision's Mask R-CNN. Each item returns (image_tensor, target_dict)
    target_dict keys: boxes (FloatTensor[N,4]), labels (Int64Tensor[N]), masks (UInt8Tensor[N,H,W]), image_id (Int64Tensor[1]), area, iscrowd
    """

    def __init__(self, coco_root: str, images_folder: str, instances_json: str,
                 transforms_img=None, subset_ids: Optional[List[int]] = None):
        self.coco_root = coco_root
        self.images_folder = images_folder
        self.instances_json = instances_json
        self.transforms_img = transforms_img
        self.coco = COCO(instances_json)

        all_ids = list(self.coco.imgs.keys())
        if subset_ids is not None:
            present = set(all_ids)
            self.ids = [iid for iid in subset_ids if iid in present]
        else:
            self.ids = all_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.coco_root, self.images_folder, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            if "bbox" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann.get("category_id", 1))
            mask = self.coco.annToMask(ann)  # H x W, binary
            masks.append(torch.as_tensor(mask, dtype=torch.uint8))
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            # Provide a dummy background box to avoid empty targets
            w_img, h_img = img.size
            boxes = [[0, 0, 1, 1]]
            labels = [0]
            masks = [torch.zeros((h_img, w_img), dtype=torch.uint8)]
            areas = [1]
            iscrowd = [0]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.stack(masks) if len(masks) > 0 else torch.zeros((0, info["height"], info["width"]), dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms_img:
            img = self.transforms_img(img)

        return img, target


# ---------------------------
# Models: Captioning (ResNet -> Transformer)
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :].to(x.device)


class EncoderCNN(nn.Module):
    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # use resnet34 by default (faster than resnet50)
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if hasattr(models, "ResNet34_Weights") else None) \
            if pretrained else models.resnet34(weights=None)
        modules = list(resnet.children())[:-2]  # remove avgpool and fc
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B,3,H,W)
        feats = self.backbone(images)  # (B,C,H',W')
        pooled = self.avgpool(feats).view(feats.size(0), -1)  # (B,C)
        emb = self.fc(pooled)  # (B, embed_dim)
        return emb


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, nhead: int = 8, num_layers: int = 4, max_len: int = 30):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def generate_square_subsequent_mask(self, size: int, device: torch.device):
        # returns mask with True in positions to be masked (PyTorch uses float mask in some variants; here we'll pass bool mask)
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
        return mask  # (T, T)

    def forward(self, tgt_seq: torch.Tensor, memory: torch.Tensor):
        # tgt_seq: (B, T) token ids
        B, T = tgt_seq.size()
        tgt_emb = self.embed(tgt_seq) * math.sqrt(self.embed_dim)
        tgt_emb = self.pos_enc(tgt_emb)  # (B,T,D)
        tgt_mask = self.generate_square_subsequent_mask(T, tgt_seq.device)
        # Transformer expects (B,T,D) with batch_first=True
        out = self.transformer_decoder(tgt=tgt_emb, memory=memory.unsqueeze(1), tgt_mask=tgt_mask)
        logits = self.fc_out(out)  # (B,T,vocab)
        return logits


class CaptioningModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, nhead: int = 8, num_layers: int = 4, max_len: int = 30, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim=embed_dim, pretrained=pretrained_encoder)
        self.decoder = DecoderTransformer(vocab_size=vocab_size, embed_dim=embed_dim, nhead=nhead, num_layers=num_layers, max_len=max_len)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        # images: (B,3,H,W), captions: (B,T)
        enc = self.encoder(images)  # (B, D)
        logits = self.decoder(captions[:, :-1], enc)  # predict next token for each position
        return logits  # (B, T-1, V)


# ---------------------------
# Greedy Decoding
# ---------------------------
def greedy_decode_caption(model: CaptioningModel, image_tensor: torch.Tensor, vocab: Vocabulary,
                          max_len: int = 30, device: Optional[str] = None) -> str:
    """
    image_tensor: (3,H,W) or (1,3,H,W)
    """
    device = device or DEFAULT_CONFIG["device"]
    model = model.to(device)
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    # encode once
    with torch.no_grad():
        enc = model.encoder(image_tensor)  # (1,D)
    idx2word = vocab.idx2word
    word2idx = vocab.word2idx
    cur = torch.tensor([[word2idx["<sos>"]]], dtype=torch.long, device=device)
    generated = [word2idx["<sos>"]]
    for i in range(max_len - 1):
        with torch.no_grad():
            logits = model.decoder(cur, enc)  # (1, T, V)
        next_token_logits = logits[0, -1]
        next_idx = next_token_logits.argmax().item()
        generated.append(next_idx)
        if idx2word.get(next_idx, "<unk>") == "<eos>":
            break
        cur = torch.cat([cur, torch.tensor([[next_idx]], dtype=torch.long, device=device)], dim=1)
    words = [idx2word.get(idx, "<unk>") for idx in generated]
    # strip special tokens and return
    filtered = [w for w in words if w not in ["<sos>", "<pad>", "<eos>"]]
    return " ".join(filtered)


# ---------------------------
# Training Routines
# ---------------------------
def get_subset_image_ids_from_coco(json_path: str, n: Optional[int]) -> List[int]:
    coco = COCO(json_path)
    all_ids = list(coco.imgs.keys())
    if n is None or n >= len(all_ids):
        return all_ids
    return random.sample(all_ids, n)


def train_captioning(
    coco_root: str,
    images_folder: str,
    captions_json: str,
    save_dir: str = "./checkpoints",
    subset: int = 30000,
    epochs: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
    lr: float = 3e-4,
    max_len: int = 30,
    embed_dim: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    device: Optional[str] = None,
):
    device = device or DEFAULT_CONFIG["device"]
    print(f"[train_captioning] device={device}, subset={subset}, epochs={epochs}, batch_size={batch_size}")

    # transforms
    transform_cap = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # subset ids
    subset_ids = get_subset_image_ids_from_coco(captions_json, subset)
    print(f"[train_captioning] sampled {len(subset_ids)} image ids")

    # dataset
    dataset = COCOCaptionDataset(
        coco_root=coco_root,
        images_folder=images_folder,
        captions_json=captions_json,
        transforms_img=transform_cap,
        max_len=max_len,
        subset_ids=subset_ids,
        min_freq=5,
        build_vocab=True
    )

    print(f"[train_captioning] vocab size = {len(dataset.vocab)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = CaptioningModel(vocab_size=len(dataset.vocab), embed_dim=embed_dim, nhead=nhead, num_layers=num_layers, max_len=max_len).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word2idx["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ensure_dir(save_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Cap Epoch {epoch}/{epochs}")
        for imgs, caps, _ in loop:
            imgs = imgs.to(device)
            caps = caps.to(device)  # (B, T)
            logits = model(imgs, caps)  # (B, T-1, V)
            B, Tm1, V = logits.shape
            # target is caps[:,1:]
            loss = criterion(logits.reshape(-1, V), caps[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(dataloader)
        print(f"[train_captioning] Epoch {epoch} avg loss: {avg_loss:.4f}")
        ckpt_path = os.path.join(save_dir, f"caption_epoch{epoch}.pth")
        save_checkpoint(ckpt_path, model, optimizer, epoch, extra={"vocab": {"word2idx": dataset.vocab.word2idx, "idx2word": dataset.vocab.idx2word}})
        print(f"[train_captioning] Saved checkpoint: {ckpt_path}")
    print("[train_captioning] Training finished.")


def train_segmentation(
    coco_root: str,
    images_folder: str,
    instances_json: str,
    save_dir: str = "./checkpoints",
    subset: int = 30000,
    epochs: int = 5,
    batch_size: int = 4,
    num_workers: int = 4,
    lr: float = 1e-4,
    device: Optional[str] = None,
):
    device = device or DEFAULT_CONFIG["device"]
    print(f"[train_segmentation] device={device}, subset={subset}, epochs={epochs}, batch_size={batch_size}")

    transform_train = transforms.Compose([transforms.ToTensor()])

    subset_ids = get_subset_image_ids_from_coco(instances_json, subset)
    print(f"[train_segmentation] sampled {len(subset_ids)} image ids")

    dataset = COCOSegmentationDatasetForMaskRCNN(
        coco_root=coco_root,
        images_folder=images_folder,
        instances_json=instances_json,
        transforms_img=transform_train,
        subset_ids=subset_ids
    )

    def collate_fn(batch):
        return tuple(zip(*batch))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    ensure_dir(save_dir)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Seg Epoch {epoch}/{epochs}")
        for imgs, targets in loop:
            imgs = list(img.to(device) for img in imgs)
            # targets is a tuple of dicts
            processed_targets = []
            for t in targets:
                t_proc = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        t_proc[k] = v.to(device)
                    else:
                        t_proc[k] = v
                processed_targets.append(t_proc)
            loss_dict = model(imgs, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            loss_val = losses.item()
            total_loss += loss_val
            loop.set_postfix(loss=loss_val)
        avg_loss = total_loss / len(dataloader)
        print(f"[train_segmentation] Epoch {epoch} avg loss: {avg_loss:.4f}")
        ckpt_path = os.path.join(save_dir, f"maskrcnn_epoch{epoch}.pth")
        save_checkpoint(ckpt_path, model, optimizer, epoch)
        print(f"[train_segmentation] Saved checkpoint: {ckpt_path}")
    print("[train_segmentation] Training finished.")


# ---------------------------
# Inference Routines
# ---------------------------
def infer_caption_from_checkpoint(cap_checkpoint: str, image_path: str, device: Optional[str] = None, max_len: Optional[int] = None) -> str:
    device = device or DEFAULT_CONFIG["device"]
    ckpt = torch.load(cap_checkpoint, map_location=device)
    vocab_dict = ckpt.get("vocab")
    if vocab_dict is None:
        raise RuntimeError("Caption checkpoint doesn't contain 'vocab'. Make sure to save vocab in checkpoint.")
    # reconstruct vocab object (lightweight)
    vocab = Vocabulary(min_freq=1)
    vocab.word2idx = vocab_dict["word2idx"]
    vocab.idx2word = {int(k): v for k, v in vocab_dict["idx2word"].items()} if isinstance(vocab_dict["idx2word"], dict) and all(isinstance(k, str) for k in vocab_dict["idx2word"].keys()) else vocab_dict["idx2word"]
    # vocab may not be fully functional but we only need idx2word/word2idx
    vocab._frozen = True

    # Prepare model
    V = len(vocab.word2idx)
    mcfg = DEFAULT_CONFIG["caption"]
    max_len = max_len or mcfg["max_len"]
    model = CaptioningModel(vocab_size=V, embed_dim=mcfg["embed_dim"], nhead=mcfg["nhead"], num_layers=mcfg["num_layers"], max_len=max_len)
    model.load_state_dict(ckpt["model_state"])
    # load image
    transform_cap = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    timg = transform_cap(img)
    caption = greedy_decode_caption(model, timg, vocab, max_len=max_len, device=device)
    return caption


def infer_segmentation_from_checkpoint(seg_checkpoint: str, image_path: str, device: Optional[str] = None, threshold: float = 0.5, save_mask: Optional[str] = None) -> Tuple[int, Optional[str]]:
    device = device or DEFAULT_CONFIG["device"]
    ckpt = torch.load(seg_checkpoint, map_location=device)
    # Create a Mask R-CNN model with matching state dict shape
    model = maskrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open(image_path).convert("RGB")
    timg = transform(img).to(device)
    with torch.no_grad():
        preds = model([timg])
    pred = preds[0]
    keep = [i for i, s in enumerate(pred["scores"]) if s > threshold]
    masks = pred["masks"][keep]  # each mask is [1,H,W]
    boxes = pred["boxes"][keep]
    labels = pred["labels"][keep]
    scores = pred["scores"][keep]
    # Optionally save masks as PNGs
    saved_dir = None
    if save_mask is not None:
        saved_dir = Path(save_mask)
        saved_dir.mkdir(parents=True, exist_ok=True)
        base = Path(image_path).stem
        for i, mask in enumerate(masks):
            m = mask[0].mul(255).byte().cpu().numpy()
            Image.fromarray(m).save(saved_dir / f"{base}_mask{i}.png")
    return len(masks), saved_dir


# ---------------------------
# CLI
# ---------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="COCO Captioning + Segmentation Single-file pipeline")
    p.add_argument("--mode", type=str, required=True, choices=["train_cap", "train_seg", "infer_cap", "infer_seg", "infer_both"], help="Operation mode")
    # dataset paths
    p.add_argument("--coco_root", type=str, default=None, help="COCO root folder (parent of images_folder)")
    p.add_argument("--images_folder", type=str, default="train2017", help="Images folder name inside coco_root")
    p.add_argument("--captions_json", type=str, default=None, help="Path to captions_train2017.json")
    p.add_argument("--instances_json", type=str, default=None, help="Path to instances_train2017.json")

    # training hyperparams
    p.add_argument("--subset", type=int, default=30000, help="Number of images to sample for subset (use none/all by providing a large number)")
    p.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--save_dir", type=str, default=DEFAULT_CONFIG["save_dir"])

    # inference
    p.add_argument("--image_path", type=str, default=None)
    p.add_argument("--cap_checkpoint", type=str, default=None)
    p.add_argument("--seg_checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])

    return p


def main_cli(argv=None):
    p = build_argparser()
    args = p.parse_args(argv)

    mode = args.mode
    device = args.device or DEFAULT_CONFIG["device"]
    save_dir = args.save_dir
    ensure_dir(save_dir)

    if mode == "train_cap":
        if not args.coco_root or not args.captions_json:
            print("Error: --coco_root and --captions_json are required for train_cap", file=sys.stderr)
            sys.exit(1)
        epochs = args.epochs or 10
        batch_size = args.batch_size or DEFAULT_CONFIG["caption"]["batch_size"]
        lr = args.lr or DEFAULT_CONFIG["caption"]["lr"]
        train_captioning(
            coco_root=args.coco_root,
            images_folder=args.images_folder,
            captions_json=args.captions_json,
            save_dir=save_dir,
            subset=args.subset,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=args.num_workers,
            lr=lr,
            max_len=DEFAULT_CONFIG["caption"]["max_len"],
            embed_dim=DEFAULT_CONFIG["caption"]["embed_dim"],
            nhead=DEFAULT_CONFIG["caption"]["nhead"],
            num_layers=DEFAULT_CONFIG["caption"]["num_layers"],
            device=device
        )

    elif mode == "train_seg":
        if not args.coco_root or not args.instances_json:
            print("Error: --coco_root and --instances_json are required for train_seg", file=sys.stderr)
            sys.exit(1)
        epochs = args.epochs or 5
        batch_size = args.batch_size or DEFAULT_CONFIG["segmentation"]["batch_size"]
        lr = args.lr or DEFAULT_CONFIG["segmentation"]["lr"]
        train_segmentation(
            coco_root=args.coco_root,
            images_folder=args.images_folder,
            instances_json=args.instances_json,
            save_dir=save_dir,
            subset=args.subset,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=args.num_workers,
            lr=lr,
            device=device
        )

    elif mode == "infer_cap":
        if not args.image_path or not args.cap_checkpoint:
            print("Error: --image_path and --cap_checkpoint are required for infer_cap", file=sys.stderr)
            sys.exit(1)
        print(f"[infer_cap] device={device}, image={args.image_path}, checkpoint={args.cap_checkpoint}")
        cap = infer_caption_from_checkpoint(args.cap_checkpoint, args.image_path, device=device)
        print("Generated caption:", cap)

    elif mode == "infer_seg":
        if not args.image_path or not args.seg_checkpoint:
            print("Error: --image_path and --seg_checkpoint are required for infer_seg", file=sys.stderr)
            sys.exit(1)
        print(f"[infer_seg] device={device}, image={args.image_path}, checkpoint={args.seg_checkpoint}")
        n_masks, saved_dir = infer_segmentation_from_checkpoint(args.seg_checkpoint, args.image_path, device=device, save_mask=save_dir)
        print(f"Found {n_masks} masks; saved to {saved_dir}")

    elif mode == "infer_both":
        if not args.image_path or not args.cap_checkpoint or not args.seg_checkpoint:
            print("Error: --image_path, --cap_checkpoint and --seg_checkpoint are required for infer_both", file=sys.stderr)
            sys.exit(1)
        print("[infer_both] Running segmentation...")
        n_masks, saved_dir = infer_segmentation_from_checkpoint(args.seg_checkpoint, args.image_path, device=device, save_mask=save_dir)
        print(f"[infer_both] Saved {n_masks} segmentation masks to {saved_dir}")
        print("[infer_both] Running captioning...")
        cap = infer_caption_from_checkpoint(args.cap_checkpoint, args.image_path, device=device)
        print(f"[infer_both] Caption: {cap}")

    else:
        print("Unknown mode", mode, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()

