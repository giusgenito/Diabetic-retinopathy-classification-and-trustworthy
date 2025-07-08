import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import gc
from werkzeug.utils import secure_filename
import sys
import json

try:
    hub_dir = torch.hub.get_dir(); dinov2_dir = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
    if dinov2_dir not in sys.path: sys.path.append(dinov2_dir)
    import dinov2.layers.attention as dino_attention
except Exception:
    class Dummy: XFORMERS_AVAILABLE = True
    dino_attention = Dummy()

class ClassificationHead(nn.Module):
    def __init__(self, backbone_model, num_classes_head=5):
        super(ClassificationHead, self).__init__(); self.backbone = backbone_model; feature_dim = 1536
        self.classifier = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes_head))
    def forward(self, x):
        features_dict = self.backbone.forward_features(x); return self.classifier(features_dict['x_norm_clstoken'])

def rollout(attentions, discard_ratio, head_fusion):
    if not attentions: raise ValueError("Lista attenzioni vuota.")
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean": attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max": attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min": attention_heads_fused = attention.min(axis=1)[0]
            else: raise Exception("Fusion type not supported")
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1); _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]; flat[0, indices] = 0
            I = torch.eye(attention_heads_fused.size(-1)); a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True); result = torch.matmul(a, result)
    num_register_tokens = 4; mask = result[0, 0, 1 + num_register_tokens:]
    width = int(mask.size(-1)**0.5)
    if width * width != mask.size(-1): raise ValueError("Mask size non è un quadrato perfetto.")
    mask = mask.reshape(width, width).numpy()
    if np.max(mask) > 0: mask = mask / np.max(mask)
    return mask

class VITAttentionRollout:
    def __init__(self, model, head_fusion="min", discard_ratio=0.9):
        self.model, self.head_fusion, self.discard_ratio = model, head_fusion, discard_ratio; self.attentions, self.hooks = [], []
        for name, module in self.model.backbone.named_modules():
            if 'attn.attn_drop' in name and isinstance(module, nn.Dropout): self.hooks.append(module.register_forward_hook(self.get_attention))
    def get_attention(self, module, input, output): self.attentions.append(input[0].cpu())
    def __call__(self, input_tensor):
        self.attentions = []; original_xformers_state = dino_attention.XFORMERS_AVAILABLE; dino_attention.XFORMERS_AVAILABLE = False
        self.model.train();
        with torch.no_grad(): self.model.backbone(input_tensor)
        self.model.eval(); dino_attention.XFORMERS_AVAILABLE = original_xformers_state
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
    def remove_hooks(self):
        for handle in self.hooks: handle.remove()

LESION_COLORS_BGR = {"hard_exudate": (0, 255, 255), "microaneurysm": (0, 0, 255), "retinal_hemorrhage": (0, 255, 0), "neovascularization": (255, 0, 0), "cotton_wool_spots": (255, 0, 255)}
def find_lesion_bounding_boxes(gt_mask_bgr):
    if gt_mask_bgr is None: return []
    h, w, _ = gt_mask_bgr.shape; total_lesion_mask = np.zeros((h, w), dtype=np.uint8)
    for color in LESION_COLORS_BGR.values():
        color_mask = cv2.inRange(gt_mask_bgr, np.array(color), np.array(color)); total_lesion_mask = cv2.bitwise_or(total_lesion_mask, color_mask)
    contours, _ = cv2.findContours(total_lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]

# --- FUNZIONI PYTHON PER COERENZA CON JAVASCRIPT ---
def jet_py(v):
    r, g, b = 1.0, 1.0, 1.0
    if 0.0 <= v < 0.125: r, g, b = 0, 0, 0.5 + 4 * v
    elif 0.125 <= v < 0.375: r, g, b = 0, 4 * (v - 0.125), 1
    elif 0.375 <= v < 0.625: r, g, b = 4 * (v - 0.375), 1, 1 - 4 * (v - 0.375)
    elif 0.625 <= v < 0.875: r, g, b = 1, 1 - 4 * (v - 0.625), 0
    else: r, g, b = 1, 0, 0 # Clamp a rosso puro
    return int(r*255), int(g*255), int(b*255)

def is_hot_color_py(r, g, b):
    # Regola coerente con quella usata in JavaScript
    return r > 150 and b < 100

def generate_final_explanation(uploaded_image_path, masks_base_dir, output_folder, model_input_dim=504):
    try:
        # Fasi preliminari (invariate)...
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', verbose=False).to(device)
        model = ClassificationHead(backbone, num_classes_head=5).to(device); model_path = os.path.join(os.path.dirname(__file__), 'best_dinov2_model_con_preprocessing.pth')
        model.load_state_dict(torch.load(model_path, map_location=device)); model.eval()
        transform = transforms.Compose([transforms.Resize((model_input_dim, model_input_dim)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        original_pil_img = Image.open(uploaded_image_path).convert('RGB'); input_tensor = transform(original_pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor); probabilities = torch.softmax(logits, dim=1); predicted_class = torch.argmax(probabilities, dim=1).item(); confidence = probabilities[0, predicted_class].item()
        attention_rollout = VITAttentionRollout(model, head_fusion='min', discard_ratio=0.9); attention_mask = attention_rollout(input_tensor); attention_rollout.remove_hooks()
        
        image_id = os.path.basename(uploaded_image_path).split('_')[0]
        mask_search_pattern = os.path.join(masks_base_dir, f"{image_id}*.png"); mask_files = glob.glob(mask_search_pattern)
        lesion_boxes = []
        if mask_files: gt_mask_bgr = cv2.imread(mask_files[0]); lesion_boxes = find_lesion_bounding_boxes(gt_mask_bgr)
        
        base_filename = secure_filename(os.path.splitext(os.path.basename(uploaded_image_path))[0])
        analysis_preview_path = os.path.join(output_folder, f"{base_filename}_analysis.png")
        attention_data_path = os.path.join(output_folder, f"{base_filename}_attention_data.json")
        lesion_data_path = os.path.join(output_folder, f"{base_filename}_lesion_data.json")

        original_cv_img_resized = cv2.resize(cv2.cvtColor(np.array(original_pil_img), cv2.COLOR_RGB2BGR), (model_input_dim, model_input_dim))
        attention_mask_resized = cv2.resize(attention_mask, (model_input_dim, model_input_dim))
        heatmap_colored = np.zeros_like(original_cv_img_resized)
        for i in range(attention_mask_resized.shape[0]):
            for j in range(attention_mask_resized.shape[1]):
                r, g, b = jet_py(attention_mask_resized[i, j])
                heatmap_colored[i, j] = [b, g, r]

        analysis_image = cv2.addWeighted(original_cv_img_resized, 0.6, heatmap_colored, 0.4, 0)
        
        # --- LOGICA DI DISEGNO DEI RETTANGOLI AGGIORNATA ---
        if lesion_boxes:
            ratio_threshold = 0.20  # Soglia del 30%
            
            for x, y, w, h in lesion_boxes:
                hot_pixel_count = 0
                total_pixel_count = w * h
                if total_pixel_count == 0: continue

                # Itera sui pixel all'interno del rettangolo
                for i in range(y, y + h):
                    for j in range(x, x + w):
                        if 0 <= i < model_input_dim and 0 <= j < model_input_dim:
                            r, g, b = jet_py(attention_mask_resized[i, j])
                            if is_hot_color_py(r, g, b):
                                hot_pixel_count += 1
                
                # Calcola il rapporto e determina il colore
                hot_ratio = hot_pixel_count / total_pixel_count
                is_hit = hot_ratio > ratio_threshold
                
                color = (0, 255, 0) if is_hit else (0, 0, 255)  # Verde se hit, rosso se miss
                cv2.rectangle(analysis_image, (x, y), (x+w, y+h), color, 2)
                coverage_text = f"{hot_ratio * 100:.1f}%"
                text_position = (x, max(y - 10, 0))
                cv2.putText(
                    analysis_image,
                    coverage_text,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,              # dimensione fon 
                    color,            # stesso colore del rettangolo
                    1,                # spessore del testo
                    cv2.LINE_AA)

           
        # --- FINE LOGICA AGGIORNATA ---

        cv2.imwrite(analysis_preview_path, analysis_image)
        with open(attention_data_path, 'w') as f: json.dump(attention_mask_resized.tolist(), f)
        with open(lesion_data_path, 'w') as f: json.dump(lesion_boxes, f)

        del model, backbone, input_tensor, logits, probabilities, attention_mask; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return predicted_class, confidence, analysis_preview_path, attention_data_path, lesion_data_path
    except Exception as e:
        print(f"ERRORE CRITICO in generate_final_explanation: {e}"); import traceback; traceback.print_exc()
        return None, None, None, None, None