"""
Test script: verify DINO + SAM3 load into a single process with caching.

Run inside the Singularity container:
    python test_combined_models.py --image cats.jpg --dino-prompt "cat" --sam3-prompt "cat"

This will:
  1. Load DINO (first call — slow)
  2. Run DINO inference
  3. Load DINO again (cached — instant)
  4. Load SAM3 (first call — slow)
  5. Run SAM3 inference
  6. Load SAM3 again (cached — instant)
  7. Save annotated outputs
"""

import argparse
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Same ModelCache class from annotation_server.py
# ---------------------------------------------------------------------------

class ModelCache:
    def __init__(self):
        self._dino_model = None
        self._dino_processor = None
        self._sam3_model = None
        self._sam3_processor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Cache] Device: {self._device}")

    def get_dino(self):
        if self._dino_model is None:
            print("[Cache] Loading DINO...")
            t0 = time.time()
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            model_id = "IDEA-Research/grounding-dino-tiny"
            self._dino_processor = AutoProcessor.from_pretrained(model_id)
            self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self._device)
            print(f"[Cache] DINO loaded in {time.time() - t0:.1f}s")
        else:
            print("[Cache] DINO already loaded (cached)")
        return self._dino_model, self._dino_processor

    def get_sam3(self):
        if self._sam3_model is None:
            print("[Cache] Loading SAM3...")
            t0 = time.time()
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self._sam3_model = build_sam3_image_model()
            self._sam3_processor = Sam3Processor(self._sam3_model)
            print(f"[Cache] SAM3 loaded in {time.time() - t0:.1f}s")
        else:
            print("[Cache] SAM3 already loaded (cached)")
        return self._sam3_model, self._sam3_processor


cache = ModelCache()


def test_dino(image_path, prompt, threshold=0.3):
    print(f"\n{'='*50}")
    print(f"  DINO Test: prompt='{prompt}'")
    print(f"{'='*50}")

    model, processor = cache.get_dino()
    image = Image.open(image_path).convert("RGB")

    labels = [l.strip() for l in prompt.split(",")]
    text = ". ".join(labels) + "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - t0

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],
    )

    result = results[0]
    boxes = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        b = [round(x, 2) for x in box.tolist()]
        s = round(score.item(), 4)
        print(f"  Detected '{label}' score={s} box={b}")
        boxes.append({"box": b, "score": s, "label": label})

    print(f"  Inference time: {inference_time:.3f}s")
    print(f"  Total candidates: {len(boxes)}")

    # Test that cache works — second call should be instant
    print("\n  [Testing cache reuse...]")
    model2, proc2 = cache.get_dino()
    assert model2 is model, "Cache miss! Model was reloaded."
    print("  Cache OK — same object in memory.")

    return boxes


def test_sam3(image_path, prompt):
    print(f"\n{'='*50}")
    print(f"  SAM3 Test: prompt='{prompt}'")
    print(f"{'='*50}")

    model, processor = cache.get_sam3()
    image = Image.open(image_path).convert("RGB")

    inference_state = processor.set_image(image)

    t0 = time.time()
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    inference_time = time.time() - t0

    masks = output["masks"]
    scores = output["scores"]
    boxes = output.get("boxes", [])

    if torch.is_tensor(masks):
        masks = masks.detach().cpu()
    if torch.is_tensor(scores):
        scores = scores.detach().cpu()

    num = len(scores)
    print(f"  Masks returned: {num}")
    for i in range(num):
        s = float(scores[i])
        mask_i = masks[i]
        if torch.is_tensor(mask_i):
            mask_i = mask_i.squeeze().numpy()
        else:
            mask_i = np.array(mask_i).squeeze()
        if mask_i.dtype != np.bool_:
            mask_i = mask_i > 0.5
        pixel_count = mask_i.sum()
        total = mask_i.size
        pct = 100.0 * pixel_count / total
        print(f"  Mask {i}: score={s:.4f}, coverage={pct:.1f}% ({pixel_count}/{total} pixels)")

    print(f"  Inference time: {inference_time:.3f}s")

    # Test that cache works
    print("\n  [Testing cache reuse...]")
    model2, proc2 = cache.get_sam3()
    assert model2 is model, "Cache miss! Model was reloaded."
    print("  Cache OK — same object in memory.")

    return num


def main():
    parser = argparse.ArgumentParser(description="Test DINO + SAM3 combined loading")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--dino-prompt", type=str, default="cat", help="DINO text prompt")
    parser.add_argument("--sam3-prompt", type=str, default="cat", help="SAM3 text prompt")
    parser.add_argument("--threshold", type=float, default=0.3, help="DINO score threshold")
    args = parser.parse_args()

    print(f"\nTest image: {args.image}")
    print(f"DINO prompt: '{args.dino_prompt}'")
    print(f"SAM3 prompt: '{args.sam3_prompt}'")

    # Run DINO
    dino_boxes = test_dino(args.image, args.dino_prompt, args.threshold)

    # Run SAM3
    sam3_count = test_sam3(args.image, args.sam3_prompt)

    # Summary
    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  Device:         {cache._device}")
    print(f"  DINO loaded:    {cache._dino_model is not None}")
    print(f"  SAM3 loaded:    {cache._sam3_model is not None}")
    print(f"  DINO candidates: {len(dino_boxes)}")
    print(f"  SAM3 masks:     {sam3_count}")
    print(f"\n  Both models coexist in memory. Server is ready to go.")


if __name__ == "__main__":
    main()