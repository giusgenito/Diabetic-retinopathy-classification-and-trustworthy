# =============================================================================
#                        IMPORTS & INITIAL SETUP
# =============================================================================
import os
import glob
import sys
import json
import gc

# Third-party libraries
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename

# --- DINOv2 Specific Import ---
# This block dynamically adds the DINOv2 repository directory to the Python path
# to import its custom modules. It includes a fallback to prevent crashes if the
# DINOv2 code is not found or xformers is not installed.
try:
    hub_dir = torch.hub.get_dir()
    dinov2_dir = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
    if dinov2_dir not in sys.path:
        sys.path.append(dinov2_dir)
    # This import is primarily used to toggle hardware-accelerated attention (xformers)
    import dinov2.layers.attention as dino_attention
except Exception:
    # If the import fails, create a dummy class to ensure the script can run.
    # This is important for Attention Rollout, which needs the non-accelerated path.
    class Dummy:
        XFORMERS_AVAILABLE = True
    dino_attention = Dummy()


# =============================================================================
#                        MODEL & EXPLAINABILITY CLASSES
# =============================================================================

class ClassificationHead(nn.Module):
    """
    A simple classification head to be attached to a pre-trained backbone.
    It consists of a two-layer MLP with a ReLU activation and Dropout.
    """
    def __init__(self, backbone_model, num_classes_head=5):
        super(ClassificationHead, self).__init__()
        self.backbone = backbone_model
        
        # The feature dimension of the DINOv2 ViT-G/14 model output.
        feature_dim = 1536
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes_head)
        )

    def forward(self, x):
        """
        Performs a forward pass, extracting features from the backbone and
        classifying them.
        """
        # Use the backbone's `forward_features` to get the output dictionary.
        features_dict = self.backbone.forward_features(x)
        # Use the CLS token for classification, as is standard for Vision Transformers.
        return self.classifier(features_dict['x_norm_clstoken'])

def rollout(attentions, discard_ratio, head_fusion):
    """
    Generates a single attention map from a list of attention maps from a ViT,
    as described in the "Attention Rollout" paper.
    """
    if not attentions:
        raise ValueError("Attention list cannot be empty.")
    
    # Initialize the result as an identity matrix.
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            # 1. Fuse attention heads
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise Exception("Invalid head fusion type.")

            # 2. Discard low-attention tokens to reduce noise
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]  # Keep the CLS token
            flat[0, indices] = 0

            # 3. Add residual connection and re-normalize
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            
            # 4. Propagate attention through layers via matrix multiplication
            result = torch.matmul(a, result)

    # Extract the attention map corresponding to the CLS token.
    # The first (1 + num_register_tokens) tokens are special (CLS + registers).
    num_register_tokens = 4 
    mask = result[0, 0, 1 + num_register_tokens:]
    
    # Reshape the mask into a 2D grid.
    width = int(mask.size(-1) ** 0.5)
    if width * width != mask.size(-1):
        raise ValueError("Mask size is not a perfect square.")
    mask = mask.reshape(width, width).numpy()
    
    # Normalize the mask to the [0, 1] range for visualization.
    if np.max(mask) > 0:
        mask = mask / np.max(mask)
        
    return mask

class VITAttentionRollout:
    """
    A utility class to generate attention maps for a Vision Transformer (ViT)
    using the Attention Rollout method. It uses PyTorch hooks to capture attention
    weights from the specified layers during a forward pass.
    """
    def __init__(self, model, head_fusion="min", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.hooks = []
        
        # Register a forward hook on the attention dropout layer of each block
        # to capture the attention maps non-invasively.
        for name, module in self.model.backbone.named_modules():
            if 'attn.attn_drop' in name and isinstance(module, nn.Dropout):
                self.hooks.append(module.register_forward_hook(self.get_attention))

    def get_attention(self, module, input_tensor, output_tensor):
        """Hook function to capture the attention probabilities."""
        self.attentions.append(input_tensor[0].cpu())

    def __call__(self, input_tensor):
        """
        Executes the forward pass and computes the attention rollout.
        """
        self.attentions = []
        
        # Temporarily disable xformers, as it prevents access to raw attention maps.
        original_xformers_state = dino_attention.XFORMERS_AVAILABLE
        dino_attention.XFORMERS_AVAILABLE = False
        
        # Run a forward pass to trigger the hooks and capture attentions.
        # Set to train() mode to ensure dropout layers are active for hooks to fire.
        self.model.train()
        with torch.no_grad():
            self.model.backbone(input_tensor)
        self.model.eval()
        
        # Restore the original xformers state.
        dino_attention.XFORMERS_AVAILABLE = original_xformers_state
        
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)

    def remove_hooks(self):
        """Removes all registered hooks to clean up."""
        for handle in self.hooks:
            handle.remove()


# =============================================================================
#                        LESION & VISUALIZATION UTILITIES
# =============================================================================

# BGR color codes for different types of lesions.
LESION_COLORS_BGR = {
    "hard_exudate": (0, 255, 255), "microaneurysm": (0, 0, 255),
    "retinal_hemorrhage": (0, 255, 0), "neovascularization": (255, 0, 0),
    "cotton_wool_spots": (255, 0, 255)
}

def find_lesion_bounding_boxes(gt_mask_bgr):
    """
    Finds bounding boxes for all lesions in a ground truth mask image.
    """
    if gt_mask_bgr is None:
        return []
    
    h, w, _ = gt_mask_bgr.shape
    total_lesion_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Combine masks for all lesion types into a single binary mask.
    for color in LESION_COLORS_BGR.values():
        color_mask = cv2.inRange(gt_mask_bgr, np.array(color), np.array(color))
        total_lesion_mask = cv2.bitwise_or(total_lesion_mask, color_mask)
        
    # Find contours and convert them to bounding boxes.
    contours, _ = cv2.findContours(total_lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]

def jet_py(v):
    """Python implementation of the JET colormap for consistency with frontend."""
    r, g, b = 1.0, 1.0, 1.0
    if 0.0 <= v < 0.125: r, g, b = 0, 0, 0.5 + 4 * v
    elif 0.125 <= v < 0.375: r, g, b = 0, 4 * (v - 0.125), 1
    elif 0.375 <= v < 0.625: r, g, b = 4 * (v - 0.375), 1, 1 - 4 * (v - 0.375)
    elif 0.625 <= v < 0.875: r, g, b = 1, 1 - 4 * (v - 0.625), 0
    else: r, g, b = 1, 0, 0
    return int(r*255), int(g*255), int(b*255)

def is_hot_color_py(r, g, b):
    """Identifies 'hot' colors (yellows/reds) in the JET map."""
    return r > 150 and b < 100


# =============================================================================
#                        MAIN ANALYSIS PIPELINE
# =============================================================================

def generate_final_explanation(uploaded_image_path, masks_base_dir, output_folder, model_input_dim=504):
    """
    Main function to run the full analysis pipeline on a single image.
    It loads the model, generates a prediction and an attention map, compares
    the attention to ground truth lesions, and saves all outputs.
    
    Returns:
        A tuple containing the predicted class, confidence, and paths to output files.
    """
    try:
        # --- 1. Model Loading and Image Preprocessing ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', verbose=False).to(device)
        model = ClassificationHead(backbone, num_classes_head=5).to(device)
        model_path = os.path.join(os.path.dirname(__file__), 'best_dinov2_model_con_preprocessing.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((model_input_dim, model_input_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        original_pil_img = Image.open(uploaded_image_path).convert('RGB')
        input_tensor = transform(original_pil_img).unsqueeze(0).to(device)

        # --- 2. Inference and Attention Map Generation ---
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        attention_rollout = VITAttentionRollout(model, head_fusion='min', discard_ratio=0.9)
        attention_mask = attention_rollout(input_tensor)
        attention_rollout.remove_hooks() # Clean up hooks after use

        # --- 3. Load Ground Truth Lesion Data (if available) ---
        image_id = os.path.basename(uploaded_image_path).split('_')[0]
        mask_search_pattern = os.path.join(masks_base_dir, f"{image_id}*.png")
        mask_files = glob.glob(mask_search_pattern)
        lesion_boxes = []
        if mask_files:
            gt_mask_bgr = cv2.imread(mask_files[0])
            lesion_boxes = find_lesion_bounding_boxes(gt_mask_bgr)
        
        # --- 4. Prepare Output Paths and Images ---
        base_filename = secure_filename(os.path.splitext(os.path.basename(uploaded_image_path))[0])
        analysis_preview_path = os.path.join(output_folder, f"{base_filename}_analysis.png")
        attention_data_path = os.path.join(output_folder, f"{base_filename}_attention_data.json")
        lesion_data_path = os.path.join(output_folder, f"{base_filename}_lesion_data.json")
        
        original_cv_img_resized = cv2.resize(cv2.cvtColor(np.array(original_pil_img), cv2.COLOR_RGB2BGR), (model_input_dim, model_input_dim))
        attention_mask_resized = cv2.resize(attention_mask, (model_input_dim, model_input_dim))
        
        # Create a colored heatmap from the normalized attention mask.
        heatmap_colored = np.zeros_like(original_cv_img_resized)
        for i in range(attention_mask_resized.shape[0]):
            for j in range(attention_mask_resized.shape[1]):
                r, g, b = jet_py(attention_mask_resized[i, j])
                heatmap_colored[i, j] = [b, g, r]

        # Overlay the heatmap on the original image.
        analysis_image = cv2.addWeighted(original_cv_img_resized, 0.6, heatmap_colored, 0.4, 0)
        
        # --- 5. "Hit or Miss" Logic: Evaluate Attention on Lesions ---
        # This section draws bounding boxes on the ground truth lesions.
        # The color of the box indicates whether the model's attention
        # sufficiently focused on that lesion ('hit') or not ('miss').
        if lesion_boxes:
            ratio_threshold = 0.20  # 20% of pixels in the box must be 'hot'.
            
            for x, y, w, h in lesion_boxes:
                hot_pixel_count = 0
                total_pixel_count = w * h
                if total_pixel_count == 0: continue

                # Count the number of 'hot' attention pixels within the lesion box.
                for i in range(y, y + h):
                    for j in range(x, x + w):
                        if 0 <= i < model_input_dim and 0 <= j < model_input_dim:
                            r, g, b = jet_py(attention_mask_resized[i, j])
                            if is_hot_color_py(r, g, b):
                                hot_pixel_count += 1
                
                # Determine if the attention 'hit' the lesion based on the threshold.
                hot_ratio = hot_pixel_count / total_pixel_count
                is_hit = hot_ratio > ratio_threshold
                
                # Draw the bounding box: green for a hit, red for a miss.
                color = (0, 255, 0) if is_hit else (0, 0, 255)
                cv2.rectangle(analysis_image, (x, y), (x+w, y+h), color, 2)
                
                # Add a text label showing the percentage of hot pixel coverage.
                coverage_text = f"{hot_ratio * 100:.1f}%"
                text_position = (x, max(y - 10, 0)) # Position above the box
                cv2.putText(analysis_image, coverage_text, text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                            
        # --- 6. Save Outputs and Clean Up Memory ---
        cv2.imwrite(analysis_preview_path, analysis_image)
        with open(attention_data_path, 'w') as f:
            json.dump(attention_mask_resized.tolist(), f)
        with open(lesion_data_path, 'w') as f:
            json.dump(lesion_boxes, f)

        # Explicitly delete large objects and clear GPU cache to free memory.
        del model, backbone, input_tensor, logits, probabilities, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return predicted_class, confidence, analysis_preview_path, attention_data_path, lesion_data_path
    
    except Exception as e:
        # Robust error handling for the entire pipeline.
        print(f"CRITICAL ERROR in generate_final_explanation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None