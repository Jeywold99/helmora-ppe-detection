"""
Fine-tune RT-DETR (Hugging Face `transformers` implementation, Apache-2.0)
on the unified PPE dataset produced by prepare_dataset.py.

This deliberately does NOT use the `ultralytics` package -- that package
is AGPL-3.0 licensed regardless of which model architecture you run through
it, which would conflict with the commercial licensing of the Smart Hard
Hat product. This script uses transformers' native RTDetrForObjectDetection,
which is a clean-room Apache-2.0 reimplementation of the same official
RT-DETR research.

Base checkpoint options (all Apache-2.0, pretrained on Objects365 -> COCO):
    PekingU/rtdetr_r18vd   -- smallest/fastest, good for a first prototype
    PekingU/rtdetr_r34vd
    PekingU/rtdetr_r50vd   -- what's earmarked for the production model

Usage:
    python train_rtdetr.py \
        --train_dir ../data/unified/train \
        --val_dir   ../data/unified/valid \
        --classes   ../configs/classes.yaml \
        --model_name PekingU/rtdetr_r18vd \
        --output_dir ../runs/ppe_rtdetr_v1 \
        --epochs 30 --batch_size 4
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)


class CocoPpeDataset(Dataset):
    """Reads a Roboflow/COCO-style _annotations.coco.json + image folder."""

    def __init__(self, data_dir: str, image_processor):
        self.data_dir = Path(data_dir)
        self.image_processor = image_processor

        with open(self.data_dir / "_annotations.coco.json", "r") as f:
            coco = json.load(f)

        self.images_by_id = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images_by_id.keys())

        self.anns_by_image = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images_by_id[image_id]
        image = Image.open(self.data_dir / img_info["file_name"]).convert("RGB")

        anns = self.anns_by_image.get(image_id, [])
        formatted_anns = [
            {
                "image_id": image_id,
                "category_id": a["category_id"],
                "bbox": a["bbox"],  # COCO format: [x_min, y_min, w, h]
                "area": a["area"],
                "iscrowd": a.get("iscrowd", 0),
            }
            for a in anns
        ]
        target = {"image_id": image_id, "annotations": formatted_anns}

        encoding = self.image_processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"][0]
        labels = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--classes", required=True, help="path to classes.yaml")
    parser.add_argument("--model_name", default="PekingU/rtdetr_r18vd")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    with open(args.classes, "r") as f:
        class_cfg = yaml.safe_load(f)
    id2label = {int(k): v for k, v in class_cfg["classes"].items()}
    label2id = {v: k for k, v in id2label.items()}

    image_processor = AutoImageProcessor.from_pretrained(args.model_name)

    model = AutoModelForObjectDetection.from_pretrained(
        args.model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # replacing the pretrained COCO/Objects365 head
    )

    train_dataset = CocoPpeDataset(args.train_dir, image_processor)
    val_dataset = CocoPpeDataset(args.val_dir, image_processor)
    print(f"train images: {len(train_dataset)}  val images: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=20,
        remove_unused_columns=False,  # required for object detection collators
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model + processor to {args.output_dir}")


if __name__ == "__main__":
    main()
