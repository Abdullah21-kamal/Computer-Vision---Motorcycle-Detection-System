
import argparse
from pathlib import Path
import csv

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from tqdm import tqdm


# ---------- Helpers ----------
def is_image_file(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def get_font(img_w: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, max(12, img_w // 100))
        except Exception:
            pass
    return ImageFont.load_default()


def draw_labelled_box(draw: ImageDraw.ImageDraw, font, box, caption, stroke=3, fill_rgb=(0, 255, 0)):
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=fill_rgb, width=stroke)

    try:
        bbox = draw.textbbox((0, 0), caption, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(caption, font=font)

    pad = 2
    bg = [x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1]
    draw.rectangle(bg, fill=fill_rgb)
    draw.text((x1 + pad, bg[1] + pad), caption, fill=(0, 0, 0), font=font)


# ---------- Model ----------
def load_model_and_categories(device: torch.device):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.0)
    model.eval().to(device)

    preprocess = weights.transforms()

    categories = list(weights.meta.get("categories", []))
    if not categories or categories[0] != "__background__":
        categories = ["__background__"] + categories

    return model, preprocess, categories


# ---------- Detection ----------
def detect_and_draw_motorcycles(image: Image.Image, preds: dict, conf_thresh: float, categories):
    """Detect motorcycles and draw boxes. Returns count of motorcycles detected."""
    boxes = preds["boxes"].detach().cpu().numpy()
    scores = preds["scores"].detach().cpu().numpy()
    labels = preds["labels"].detach().cpu().numpy()

    font = get_font(image.width)
    draw = ImageDraw.Draw(image)

    # Find motorcycle label ID(s)
    moto_ids = {i for i, n in enumerate(categories) if isinstance(n, str) and n.lower() == "motorcycle"}
    if not moto_ids:
        moto_ids = {4}  # COCO fallback

    motorcycle_count = 0
    stroke = max(2, image.width // 300)

    for box, score, lbl in zip(boxes, scores, labels):
        li = int(lbl)
        if float(score) < conf_thresh:
            continue
        if li < 0 or li >= len(categories):
            continue
        if li not in moto_ids:
            continue

        name = categories[li] if li < len(categories) else "motorcycle"
        caption = f"{name} {float(score):.2f}"
        draw_labelled_box(draw, font, box, caption, stroke=stroke)
        motorcycle_count += 1

    return motorcycle_count


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Detect motorcycles in nested recording/person/image folder structure."
    )
    parser.add_argument("--input_dir", required=True, help="Root folder containing recording folders")
    parser.add_argument("--output_dir", required=True, help="Root folder for outputs")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default 0.5)")
    parser.add_argument(
        "--max_size",
        type=int,
        default=1536,
        help="Max image dimension for inference (0 = disabled)",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="motorcycle_detections.csv",
        help="CSV output filename",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"[ERROR] Input directory does not exist: {input_root}")
        return

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, preprocess, categories = load_model_and_categories(device)

    def maybe_resize(img: Image.Image):
        if args.max_size and args.max_size > 0:
            w, h = img.size
            scale = min(args.max_size / max(w, h), 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return img

    # Collect all images from nested structure
    # Structure: input_dir / recording_id / person_id / image_file
    all_images = []
    for recording_dir in sorted(input_root.iterdir()):
        if not recording_dir.is_dir():
            continue
        recording_id = recording_dir.name

        for person_dir in sorted(recording_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            person_id = person_dir.name

            for img_file in sorted(person_dir.iterdir()):
                if img_file.is_file() and is_image_file(img_file):
                    all_images.append({
                        "recording_id": recording_id,
                        "person_id": person_id,
                        "image_id": img_file.stem,
                        "image_path": img_file,
                        "relative_path": img_file.relative_to(input_root)
                    })

    if not all_images:
        print(f"[WARN] No images found in {input_root}")
        return

    print(f"Found {len(all_images)} images across recordings and persons")

    # CSV results
    csv_rows = []

    # Process each image
    for item in tqdm(all_images, desc="Processing images"):
        recording_id = item["recording_id"]
        person_id = item["person_id"]
        image_id = item["image_id"]
        img_path = item["image_path"]

        # Load and process image
        img = load_image(img_path)
        img = maybe_resize(img)

        x = preprocess(img).to(device)

        with torch.inference_mode():
            pred = model([x])[0]

        # Draw motorcycles on a copy
        canvas = img.copy()
        num_motorcycles = detect_and_draw_motorcycles(
            canvas, pred, conf_thresh=args.conf, categories=categories
        )

        # Save annotated image preserving folder structure
        # Output structure: output_dir / recording_id / person_id / image_file
        out_img_path = output_root / recording_id / person_id / img_path.name
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_img_path)

        # Record results
        csv_rows.append({
            "recording_id": recording_id,
            "person_id": person_id,
            "image_id": image_id,
            "num_motorcycles": num_motorcycles
        })

    # Write CSV report
    csv_path = output_root / args.csv_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["recording_id", "person_id", "image_id", "num_motorcycles"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary statistics
    total_images = len(csv_rows)
    images_with_motorcycles = sum(1 for row in csv_rows if row["num_motorcycles"] > 0)
    total_motorcycles = sum(row["num_motorcycles"] for row in csv_rows)

    print("\n=== Detection Summary ===")
    print(f"Total images processed:          {total_images}")
    print(f"Images with motorcycles:         {images_with_motorcycles} ({images_with_motorcycles/total_images*100:.2f}%)")
    print(f"Total motorcycles detected:      {total_motorcycles}")
    print(f"Average motorcycles per image:   {total_motorcycles/total_images:.3f}")
    if images_with_motorcycles > 0:
        print(f"Avg per image with motorcycles:  {total_motorcycles/images_with_motorcycles:.3f}")
    print("========================\n")
    print(f"CSV report saved to: {csv_path}")
    print(f"Annotated images saved to: {output_root.resolve()}")


if __name__ == "__main__":
    main()