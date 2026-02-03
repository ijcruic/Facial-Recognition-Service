#!/usr/bin/env python3
"""
Alignment Strategy Comparison on LFW

Compares different alignment approaches using ATFaceDetect:
1. Current (broken DFA fallback)
2. BRIAR-style (simple affine with 5-point landmarks)
3. MTCNN baseline
"""
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import cv2
import torch
from torchvision import transforms

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'standalone_face_detector'))

from app.representation import InceptionResnetV1


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)


def get_model_dir():
    docker_path = Path('/service/model_weights')
    if docker_path.exists():
        return docker_path
    return Path(__file__).parent.parent / 'models_weights'


# BRIAR-style reference points for 112x112
ARCFACE_REF_112 = np.float32([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041]   # right mouth
])


def align_face_briar_style(image, landmarks, output_size=160):
    """
    BRIAR-style alignment using simple affine transform.

    Args:
        image: BGR numpy array
        landmarks: 5 points as [(x,y), ...] for left_eye, right_eye, nose, left_mouth, right_mouth
        output_size: output face size

    Returns:
        Aligned face (BGR)
    """
    src_pts = np.float32(landmarks)

    # Scale reference points to output size
    scale = output_size / 112.0
    ref_pts = ARCFACE_REF_112 * scale

    # Compute similarity transform
    M = cv2.estimateAffinePartial2D(src_pts, ref_pts)[0]

    if M is None:
        # Fallback to just crop if transform fails
        return None

    # Apply transform
    aligned = cv2.warpAffine(image, M, (output_size, output_size))
    return aligned


class ATFaceDetectPipeline:
    """ATFaceDetect with BRIAR-style alignment."""

    def __init__(self, model_path, device='cuda', output_size=160):
        self.device = device
        self.output_size = output_size

        # Import the detector
        from at_face_detector import initialize_model, compute_detections_batch, PriorBox
        from at_face_detector import resize_to_target_size_with_padding

        self.model, self.cfg = initialize_model(str(model_path / 'ATFaceDetect_v3.pth'))
        self.prior_box = PriorBox(self.cfg)
        self.resize_func = resize_to_target_size_with_padding
        self.compute_detections_batch = compute_detections_batch

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_align(self, img_pil):
        """
        Detect face and return aligned tensor using BRIAR-style alignment.
        """
        # Convert PIL to numpy BGR
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Run detection
        boxes, scores, keypoints = self.compute_detections_batch(
            self.model,
            [img_rgb],
            self.prior_box,
            batch_size=1,
            resize_func=self.resize_func
        )

        if len(boxes[0]) == 0:
            return None, False, None

        # Get best detection
        best_idx = np.argmax(scores[0])
        kpts = keypoints[0][best_idx]

        # Extract 5-point landmarks
        # kpts format: [right_eye_x, right_eye_y, left_eye_x, left_eye_y, nose_x, nose_y,
        #               right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y]
        landmarks = [
            (kpts[2], kpts[3]),   # left eye
            (kpts[0], kpts[1]),   # right eye
            (kpts[4], kpts[5]),   # nose
            (kpts[8], kpts[9]),   # left mouth
            (kpts[6], kpts[7]),   # right mouth
        ]

        # Align using BRIAR style
        aligned_bgr = align_face_briar_style(img_bgr, landmarks, self.output_size)

        if aligned_bgr is None:
            return None, False, landmarks

        # Convert to RGB and tensor
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        aligned_pil = Image.fromarray(aligned_rgb)
        face_tensor = self.transform(aligned_pil)

        return face_tensor, True, landmarks


class ATFaceDetectCropOnly:
    """ATFaceDetect with simple crop (no alignment) - baseline."""

    def __init__(self, model_path, device='cuda', output_size=160):
        self.device = device
        self.output_size = output_size

        from at_face_detector import initialize_model, compute_detections_batch, PriorBox
        from at_face_detector import resize_to_target_size_with_padding

        self.model, self.cfg = initialize_model(str(model_path / 'ATFaceDetect_v3.pth'))
        self.prior_box = PriorBox(self.cfg)
        self.resize_func = resize_to_target_size_with_padding
        self.compute_detections_batch = compute_detections_batch

        self.transform = transforms.Compose([
            transforms.Resize((output_size, output_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_align(self, img_pil):
        """Detect and crop (no alignment)."""
        img_rgb = np.array(img_pil)

        boxes, scores, keypoints = self.compute_detections_batch(
            self.model,
            [img_rgb],
            self.prior_box,
            batch_size=1,
            resize_func=self.resize_func
        )

        if len(boxes[0]) == 0:
            return None, False, None

        # Get best detection
        best_idx = np.argmax(scores[0])
        box = boxes[0][best_idx]  # [x, y, w, h]

        # Crop with margin
        x, y, w, h = box
        margin = int(max(w, h) * 0.1)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_rgb.shape[1], x + w + margin)
        y2 = min(img_rgb.shape[0], y + h + margin)

        crop = img_pil.crop((x1, y1, x2, y2))
        face_tensor = self.transform(crop)

        return face_tensor, True, None


class MTCNNPipeline:
    """MTCNN baseline with built-in alignment."""

    def __init__(self, device='cuda', output_size=160):
        self.device = device
        self.output_size = output_size

        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(
            image_size=output_size,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            device=device,
            post_process=True
        )

    def detect_and_align(self, img_pil):
        """Detect and align using MTCNN."""
        try:
            face_tensor = self.mtcnn(img_pil)
            if face_tensor is None:
                return None, False, None
            return face_tensor, True, None
        except Exception:
            return None, False, None


class DirectResize:
    """No detection - just resize (worst case baseline)."""

    def __init__(self, output_size=160):
        self.transform = transforms.Compose([
            transforms.Resize((output_size, output_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_align(self, img_pil):
        return self.transform(img_pil), True, None


def parse_pairs_file(pairs_file):
    """Parse LFW pairs.txt file."""
    pairs = []
    with open(pairs_file, 'r') as f:
        lines = f.readlines()

    num_folds, num_pairs = map(int, lines[0].strip().split())
    idx = 1

    for fold in range(num_folds):
        for _ in range(num_pairs):
            parts = lines[idx].strip().split()
            name = parts[0]
            img1_idx = int(parts[1])
            img2_idx = int(parts[2])
            pairs.append({
                'name1': name, 'img1_idx': img1_idx,
                'name2': name, 'img2_idx': img2_idx,
                'label': 1
            })
            idx += 1

        for _ in range(num_pairs):
            parts = lines[idx].strip().split()
            name1 = parts[0]
            img1_idx = int(parts[1])
            name2 = parts[2]
            img2_idx = int(parts[3])
            pairs.append({
                'name1': name1, 'img1_idx': img1_idx,
                'name2': name2, 'img2_idx': img2_idx,
                'label': 0
            })
            idx += 1

    return pairs


def evaluate_pipeline(pipeline, embed_model, pairs, lfw_dir, device,
                     pipeline_name, max_pairs=None):
    """Evaluate a pipeline on LFW verification."""

    print(f"\n{'='*60}")
    print(f"Evaluating: {pipeline_name}")
    print(f"{'='*60}")

    embedding_cache = {}
    detected_count = 0
    total_images = 0

    unique_images = set()
    eval_pairs = pairs[:max_pairs] if max_pairs else pairs

    for pair in eval_pairs:
        img1 = f"{pair['name1']}_{pair['img1_idx']:04d}.jpg"
        img2 = f"{pair['name2']}_{pair['img2_idx']:04d}.jpg"
        unique_images.add(img1)
        unique_images.add(img2)

    print(f"Processing {len(unique_images)} unique images...")

    for img_name in tqdm(unique_images, desc="Extracting"):
        img_path = lfw_dir / img_name
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert('RGB')
        face_tensor, detected, _ = pipeline.detect_and_align(img)

        if face_tensor is None:
            continue

        total_images += 1
        if detected:
            detected_count += 1

        with torch.no_grad():
            emb = embed_model(face_tensor.unsqueeze(0).to(device))
            emb = emb / emb.norm(dim=1, keepdim=True)

        embedding_cache[img_name] = emb.cpu().numpy()[0]

    print(f"Detection rate: {detected_count}/{total_images} ({100*detected_count/max(1,total_images):.1f}%)")

    # Compute similarities
    similarities = []
    labels = []

    for pair in eval_pairs:
        img1 = f"{pair['name1']}_{pair['img1_idx']:04d}.jpg"
        img2 = f"{pair['name2']}_{pair['img2_idx']:04d}.jpg"

        if img1 not in embedding_cache or img2 not in embedding_cache:
            continue

        sim = cosine_similarity(embedding_cache[img1], embedding_cache[img2])
        similarities.append(sim)
        labels.append(pair['label'])

    similarities = np.array(similarities)
    labels = np.array(labels)

    # Find optimal threshold
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_acc = 0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (similarities >= thresh).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    # Compute metrics
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    tar_at_far_01 = np.interp(0.001, fpr, tpr)
    tar_at_far_1 = np.interp(0.01, fpr, tpr)

    results = {
        'Accuracy': best_acc * 100,
        'Threshold': best_thresh,
        'AUC': roc_auc,
        'TAR@FAR=0.1%': tar_at_far_01 * 100,
        'TAR@FAR=1%': tar_at_far_1 * 100,
        'Detection Rate': 100 * detected_count / max(1, total_images),
        'Pairs Evaluated': len(similarities)
    }

    print(f"\n{pipeline_name} Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare alignment strategies on LFW')
    parser.add_argument('--lfw-dir', type=str,
                       default='/data/lfw/images',
                       help='Path to LFW images')
    parser.add_argument('--pairs-file', type=str,
                       default='/data/lfw/pairs.txt',
                       help='Path to pairs.txt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-pairs', type=int, default=1000)
    parser.add_argument('--pipelines', nargs='+',
                       default=['mtcnn', 'atface_briar', 'atface_crop', 'resize'],
                       choices=['mtcnn', 'atface_briar', 'atface_crop', 'resize'],
                       help='Which pipelines to test')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    lfw_dir = Path(args.lfw_dir)
    model_dir = get_model_dir()

    # Load pairs
    pairs = parse_pairs_file(args.pairs_file)
    print(f"Loaded {len(pairs)} pairs")

    # Load embedding model
    print("Loading FaceNet embedding model...")
    embed_path = model_dir / 'representation' / '20180402-114759-facenet.pt'
    embed_model = InceptionResnetV1(weights_filename=str(embed_path), device=device).eval()

    all_results = {}

    # Test MTCNN
    if 'mtcnn' in args.pipelines:
        try:
            pipeline = MTCNNPipeline(device)
            results = evaluate_pipeline(
                pipeline, embed_model, pairs, lfw_dir, device,
                "MTCNN (facenet-pytorch)", max_pairs=args.max_pairs
            )
            all_results['MTCNN'] = results
        except Exception as e:
            print(f"MTCNN failed: {e}")

    # Test ATFaceDetect + BRIAR alignment
    if 'atface_briar' in args.pipelines:
        try:
            pipeline = ATFaceDetectPipeline(model_dir / 'detection', device)
            results = evaluate_pipeline(
                pipeline, embed_model, pairs, lfw_dir, device,
                "ATFaceDetect + BRIAR Align", max_pairs=args.max_pairs
            )
            all_results['ATFace+BRIAR'] = results
        except Exception as e:
            print(f"ATFace+BRIAR failed: {e}")
            import traceback
            traceback.print_exc()

    # Test ATFaceDetect crop only
    if 'atface_crop' in args.pipelines:
        try:
            pipeline = ATFaceDetectCropOnly(model_dir / 'detection', device)
            results = evaluate_pipeline(
                pipeline, embed_model, pairs, lfw_dir, device,
                "ATFaceDetect + Crop Only", max_pairs=args.max_pairs
            )
            all_results['ATFace+Crop'] = results
        except Exception as e:
            print(f"ATFace+Crop failed: {e}")
            import traceback
            traceback.print_exc()

    # Test direct resize
    if 'resize' in args.pipelines:
        pipeline = DirectResize()
        results = evaluate_pipeline(
            pipeline, embed_model, pairs, lfw_dir, device,
            "Direct Resize (No Detection)", max_pairs=args.max_pairs
        )
        all_results['DirectResize'] = results

    # Summary
    print("\n" + "="*80)
    print("ALIGNMENT COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Pipeline':<25} {'Accuracy':>10} {'AUC':>8} {'TAR@1%':>10} {'Det Rate':>10}")
    print("-"*80)
    for name, res in all_results.items():
        print(f"{name:<25} {res['Accuracy']:>9.2f}% {res['AUC']:>8.4f} "
              f"{res['TAR@FAR=1%']:>9.2f}% {res['Detection Rate']:>9.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
