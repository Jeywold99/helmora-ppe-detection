# PPE Detection Prototype (Smart Hard Hat foundation)

Base model prototype: detect **person / hardhat / no-hardhat / safety-vest /
no-safety-vest / gloves / no-gloves / boots / no-boots** in images, as a
foundation for the Smart Hard Hat production model.

## Why RT-DETR via Hugging Face `transformers`, specifically

This stays consistent with the RT-DETR + Objects365 decision already made
for the production model, but with one adjustment worth flagging: **don't
run RT-DETR through the `ultralytics` pip package.** That package is
AGPL-3.0 regardless of which architecture you load through it, which would
undo the licensing rationale for choosing RT-DETR over YOLOv8 in the first
place. This scaffold instead uses `transformers`' native `RTDetrForObjectDetection`
-- a clean-room Apache-2.0 reimplementation of the same official RT-DETR
research (lyuwenyu/RT-DETR), pretrained on Objects365 -> COCO. Checkpoints
used here (`PekingU/rtdetr_r18vd`, `r34vd`, `r50vd`) are all Apache-2.0.

## Dataset

**Do not train the shippable model on the SH17 dataset** -- despite having
exactly the classes you want (gloves, boots, vest, helmet), it's licensed
CC BY-NC-SA 4.0 (non-commercial only). It's fine to poke at for sanity-checking
your pipeline works end-to-end, but not for weights you intend to ship.

For commercially usable data, combine two confirmed CC BY 4.0 Roboflow
Universe sets:

1. **Construction Site Safety Image Dataset** (person/hardhat/vest, ~2.8k images)
   https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
2. **Construction PPE** by skcet (adds gloves + boots, 8,845 images)
   https://universe.roboflow.com/skcet-g4h72/construction-ppe-rdhzo

On each project page: sign in (free) -> **Dataset** tab -> **Download Dataset**
-> export format **COCO JSON** (not the default YOLO txt -- COCO keeps
explicit category names, which `prepare_dataset.py` needs to remap classes).
You'll get `train/`, `valid/`, `test/` folders each with `_annotations.coco.json`.

Then merge:

```bash
python scripts/prepare_dataset.py \
    --config configs/classes.yaml \
    --dataset construction_site_safety:/path/to/downloaded/ds1 \
    --dataset construction_ppe_skcet:/path/to/downloaded/ds2 \
    --output data/unified
```

This remaps both datasets' labels into the single taxonomy in
`configs/classes.yaml`, drops out-of-scope classes (machinery, vehicles,
traffic cones, masks), and writes a merged `train/valid/test` COCO dataset.

> Edit the `source_mappings` block in `configs/classes.yaml` to match the
> exact category names in whichever datasets you actually download --
> Roboflow projects aren't perfectly consistent about capitalization/naming.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train_rtdetr.py \
    --train_dir data/unified/train \
    --val_dir   data/unified/valid \
    --classes   configs/classes.yaml \
    --model_name PekingU/rtdetr_r18vd \
    --output_dir runs/ppe_rtdetr_v1 \
    --epochs 30 --batch_size 4
```

Start with `rtdetr_r18vd` (smallest) to confirm the pipeline works and get a
fast baseline; move to `r50vd` (the checkpoint earmarked for production) once
the data pipeline and class balance look right.

## Run inference

```bash
python scripts/infer.py --model_dir runs/ppe_rtdetr_v1 --image test.jpg
python scripts/infer.py --model_dir runs/ppe_rtdetr_v1 --webcam
```

## Next steps toward production (not in this scaffold yet)

- Add a proper mAP evaluation loop (pycocotools) to `train_rtdetr.py` --
  currently only tracks training/val loss.
- Once you're ready to build your own dataset: SH17's class list (17
  categories, including ear protection and medical suits) is a useful
  reference for what a thorough PPE taxonomy looks like, even though the
  data itself isn't usable.
- Export to ONNX / TensorRT for the Jetson Orin Nano once accuracy on the
  dev-PC-trained model looks good -- `transformers` models export to ONNX
  via `optimum`, which keeps the same Apache-2.0 licensing chain.
- Swap in your own captured images from the field (helmet-mounted cameras)
  as the primary training set once you have a labeling pipeline -- open
  datasets here are a bootstrap, not the end state.
