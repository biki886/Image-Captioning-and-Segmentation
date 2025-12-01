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
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import math

# ------------------------------
# Vocabulary
# ------------------------------
class Vocabulary:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

# ------------------------------
# Positional Encoding  (matches checkpoint: pos_enc)
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ------------------------------
# Encoder (matches training)
# ------------------------------
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

# ------------------------------
# Decoder (EXACTLY matched to checkpoint)
# ------------------------------
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

# ------------------------------
# Full model wrapper
# ------------------------------
class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderTransformer(vocab_size, embed_dim)

    def forward(self, images, captions):
        enc = self.encoder(images)
        return self.decoder(captions[:, :-1], enc)

# ------------------------------
# Greedy decoding
# ------------------------------
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

# ------------------------------
# MAIN
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab = Vocabulary(ckpt["vocab"]["word2idx"])
    vocab_size = len(vocab.word2idx)

    # Build model EXACTLY like training
    model = CaptioningModel(vocab_size)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image).convert("RGB")
    img_tensor = transform(img)

    caption = greedy_decode(model, img_tensor, vocab, device=device)
    print("\nCaption:", caption)


if __name__ == "__main__":
    main()

