"""
Run the fine-tuned RT-DETR PPE model on an image file or a live webcam feed.

Usage:
    # single image
    python infer.py --model_dir ../runs/ppe_rtdetr_v1 --image path/to/photo.jpg

    # webcam (dev PC only -- swap for a CSI/USB camera call on the Jetson later)
    python infer.py --model_dir ../runs/ppe_rtdetr_v1 --webcam
"""

import argparse

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# classes considered "violations" for a red box vs. green box in the overlay
NEGATIVE_CLASSES = {"no_hardhat", "no_safety_vest", "no_gloves", "no_boots"}


def load_model(model_dir: str):
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


def run_on_frame(processor, model, device, pil_image, threshold=0.5):
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    return results


def draw_results(cv_image, results, id2label):
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = id2label[label_id.item()]
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        color = (0, 0, 255) if label in NEGATIVE_CLASSES else (0, 200, 0)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        cv2.putText(cv_image, text, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)
    return cv_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--image", help="path to a single image")
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    processor, model, device = load_model(args.model_dir)
    id2label = model.config.id2label

    if args.image:
        pil_image = Image.open(args.image).convert("RGB")
        results = run_on_frame(processor, model, device, pil_image, args.threshold)
        cv_image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        cv_image = draw_results(cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR), results, id2label)
        out_path = "prediction.jpg"
        cv2.imwrite(out_path, cv_image)
        print(f"Saved annotated result to {out_path}")
        for score, label_id in zip(results["scores"], results["labels"]):
            print(f"  {id2label[label_id.item()]}: {score:.2f}")

    elif args.webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0)")
        print("Press 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = run_on_frame(processor, model, device, pil_image, args.threshold)
            frame = draw_results(frame, results, id2label)
            cv2.imshow("PPE Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    else:
        parser.error("Provide either --image <path> or --webcam")


if __name__ == "__main__":
    main()
