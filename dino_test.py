import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import requests

model_id = "IDEA-Research/grounding-dino-tiny"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Read image from disk and convert to RGB
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(image_url, stream=True).raw)

# Define labels as a list, then join them
labels = ["Prostate"]  # or ["a cat", "a remote control"]
text = ". ".join(labels) + "."  # Creates "monopolar curved scissors."

# Pass as a single string
inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

result = results[0]

# Create a copy of the image to draw on
annotated_image = image.copy()
draw = ImageDraw.Draw(annotated_image)

# Try to load a font, fallback to default if not available
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except:
    font = ImageFont.load_default()

# Draw boxes and labels
for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")
    
    # Draw rectangle
    draw.rectangle(box, outline="red", width=3)
    
    # Draw label with confidence
    label_text = f"{label}: {round(score.item(), 3)}"
    
    # Get text bounding box for background
    bbox = draw.textbbox((box[0], box[1] - 20), label_text, font=font)
    draw.rectangle(bbox, fill="red")
    draw.text((box[0], box[1] - 20), label_text, fill="white", font=font)

# Save annotated image
output_path = "annotated_image.jpg"
annotated_image.save(output_path)
print(f"\nAnnotated image saved to {output_path}")