import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def to_numpy_image(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def overlay_mask_on_image(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    color=(255, 0, 0),
    alpha=0.4,
) -> np.ndarray:
    """
    image_np: H x W x 3 uint8
    mask_np:  H x W bool or 0/1
    """
    out = image_np.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)

    mask_bool = mask_np.astype(bool)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_box_and_label(
    image_pil: Image.Image,
    box,
    label: str,
    color=(0, 255, 0),
    width=3,
):
    draw = ImageDraw.Draw(image_pil)
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    text_pos = (x1 + 4, max(0, y1 - 12))
    draw.text(text_pos, label, fill=color, font=font)


# -----------------------------
# Load model + processor
# -----------------------------
model = build_sam3_image_model()
processor = Sam3Processor(model)

# -----------------------------
# Load image
# -----------------------------
image = Image.open("cats.jpg").convert("RGB")
image_np = to_numpy_image(image)

# -----------------------------
# Run inference
# -----------------------------
inference_state = processor.set_image(image)

# Replace with your actual prompt
prompt_text = "cat"
output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

# -----------------------------
# Normalize outputs to CPU numpy
# -----------------------------
if torch.is_tensor(masks):
    masks = masks.detach().cpu()

if torch.is_tensor(boxes):
    boxes = boxes.detach().cpu()

if torch.is_tensor(scores):
    scores = scores.detach().cpu()

# Create base overlay image
annotated_np = image_np.copy()

# If multiple detections exist, draw all of them
num_preds = len(scores)

for i in range(num_preds):
    mask_i = masks[i]
    box_i = boxes[i]
    score_i = scores[i]

    # Handle mask shape: could be [H,W], [1,H,W], etc.
    if torch.is_tensor(mask_i):
        mask_i = mask_i.squeeze().numpy()
    else:
        mask_i = np.array(mask_i).squeeze()

    # Convert to binary mask if needed
    if mask_i.dtype != np.bool_:
        mask_i = mask_i > 0.5

    # Overlay mask
    annotated_np = overlay_mask_on_image(
        annotated_np,
        mask_i,
        color=(255, 0, 0),   # red mask
        alpha=0.35,
    )

# Convert back to PIL for box/text drawing
annotated_pil = Image.fromarray(annotated_np)

for i in range(num_preds):
    box_i = boxes[i]
    score_i = float(scores[i])

    if torch.is_tensor(box_i):
        box_i = box_i.numpy()

    label = f"{prompt_text}: {score_i:.3f}"
    draw_box_and_label(
        annotated_pil,
        box_i,
        label=label,
        color=(0, 255, 0),   # green box/text
        width=3,
    )

# Save final annotated image
output_path = "cats_annotated.png"
annotated_pil.save(output_path)
print(f"Saved annotated image to: {output_path}")