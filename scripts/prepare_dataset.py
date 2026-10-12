"""
Merge one or more Roboflow COCO-format PPE datasets into a single dataset
using the unified class taxonomy defined in configs/classes.yaml.

Expected input layout per dataset (this is Roboflow's standard COCO export):

    <dataset_dir>/
        train/
            _annotations.coco.json
            <image files>
        valid/
            _annotations.coco.json
            <image files>
        test/                      (optional)
            _annotations.coco.json
            <image files>

When exporting from Roboflow, choose "COCO JSON" as the export format --
this is more direct to merge across datasets than YOLO txt format, since
COCO keeps one JSON with explicit category names we can remap.

Usage:
    python prepare_dataset.py \
        --config ../configs/classes.yaml \
        --dataset construction_site_safety:/path/to/ds1 \
        --dataset construction_ppe_skcet:/path/to/ds2 \
        --output ../data/unified

Output layout:
    <output>/train/_annotations.coco.json + images
    <output>/valid/_annotations.coco.json + images
    <output>/test/_annotations.coco.json  + images   (if present in inputs)
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml


def load_taxonomy(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    unified_classes = cfg["classes"]  # {id: name}
    name_to_id = {name: cid for cid, name in unified_classes.items()}
    return unified_classes, name_to_id, cfg["source_mappings"]


def merge_split(split: str, datasets: list, name_to_id: dict, source_mappings: dict, output_dir: Path):
    """datasets: list of (mapping_key, dataset_root_path)"""
    out_split_dir = output_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    merged_images = []
    merged_annotations = []
    next_image_id = 1
    next_ann_id = 1
    skipped_no_split = []

    for mapping_key, dataset_root in datasets:
        src_split_dir = Path(dataset_root) / split
        ann_path = src_split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            skipped_no_split.append((mapping_key, split))
            continue

        with open(ann_path, "r") as f:
            coco = json.load(f)

        mapping = source_mappings.get(mapping_key)
        if mapping is None:
            raise ValueError(
                f"No source_mappings entry for '{mapping_key}' in classes.yaml. "
                f"Add one before merging this dataset."
            )

        # source category_id -> unified category_id (only for mapped/kept classes)
        src_catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        src_catid_to_unified_id = {}
        for cid, cname in src_catid_to_name.items():
            unified_name = mapping.get(cname)
            if unified_name is not None:
                src_catid_to_unified_id[cid] = name_to_id[unified_name]
            # else: class intentionally dropped (not in mapping table)

        # remap image ids to avoid collisions across datasets, copy files
        old_to_new_image_id = {}
        for img in coco["images"]:
            new_id = next_image_id
            next_image_id += 1
            old_to_new_image_id[img["id"]] = new_id

            src_file = src_split_dir / img["file_name"]
            new_file_name = f"{mapping_key}__{img['file_name']}"
            dst_file = out_split_dir / new_file_name
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
            else:
                print(f"  [warn] missing image, skipping copy: {src_file}")

            merged_images.append({
                "id": new_id,
                "file_name": new_file_name,
                "width": img.get("width"),
                "height": img.get("height"),
            })

        # remap annotations, dropping any whose category wasn't in the mapping
        for ann in coco["annotations"]:
            src_cid = ann["category_id"]
            if src_cid not in src_catid_to_unified_id:
                continue
            merged_annotations.append({
                "id": next_ann_id,
                "image_id": old_to_new_image_id[ann["image_id"]],
                "category_id": src_catid_to_unified_id[src_cid],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            })
            next_ann_id += 1

        print(f"  merged {mapping_key}/{split}: {len(coco['images'])} images")

    if not merged_images:
        print(f"[skip] no data found for split '{split}' across given datasets "
              f"({skipped_no_split})")
        return

    unified_categories = [{"id": cid, "name": name} for name, cid in name_to_id.items()]
    unified_categories.sort(key=lambda c: c["id"])

    out_json = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": unified_categories,
    }
    with open(out_split_dir / "_annotations.coco.json", "w") as f:
        json.dump(out_json, f)

    print(f"[done] {split}: {len(merged_images)} images, {len(merged_annotations)} annotations "
          f"-> {out_split_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to classes.yaml")
    parser.add_argument(
        "--dataset", action="append", required=True,
        help="mapping_key:/path/to/dataset_root, repeatable. "
             "mapping_key must match a key under source_mappings in classes.yaml",
    )
    parser.add_argument("--output", required=True, help="output directory for merged dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    _, name_to_id, source_mappings = load_taxonomy(args.config)

    datasets = []
    for entry in args.dataset:
        key, path = entry.split(":", 1)
        datasets.append((key, path))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n== merging split: {split} ==")
        merge_split(split, datasets, name_to_id, source_mappings, output_dir)


if __name__ == "__main__":
    main()
