#!/usr/bin/env python3
"""
LFW Detector Comparison: MTCNN vs ATFaceDetect+DFA

Compares detection pipelines on LFW verification benchmark.
LFW images are 250x250 - large enough for proper face detection.
"""
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.representation import InceptionResnetV1


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)


def get_model_dir():
    """Get model directory - handles both local and Docker paths."""
    docker_path = Path('/service/model_weights')
    if docker_path.exists():
        return docker_path
    return Path(__file__).parent.parent / 'models_weights'


class MTCNNPipeline:
    """MTCNN pipeline using facenet-pytorch."""

    def __init__(self, device='cuda'):
        self.device = device

        try:
            from facenet_pytorch import MTCNN as FacenetMTCNN
            self.mtcnn = FacenetMTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                device=device,
                post_process=True
            )
            self.available = True
        except ImportError:
            print("facenet-pytorch not available for MTCNN comparison")
            self.available = False

    def detect_and_crop(self, img):
        """Detect face and return cropped tensor."""
        if not self.available:
            return None, False

        try:
            face_tensor = self.mtcnn(img)
            if face_tensor is None:
                return None, False
            return face_tensor, True
        except Exception:
            return None, False


class NewPipeline:
    """ATFaceDetect + DFA alignment pipeline."""

    def __init__(self, device='cuda'):
        self.device = device

        from app.recognition import FaceDetector

        model_path = get_model_dir() / 'detection'

        if not (model_path / 'ATFaceDetect_v3.pth').exists():
            raise FileNotFoundError(f"ATFaceDetect model not found at {model_path}")

        self.detector = FaceDetector(
            image_size=160,
            keep_all=False,
            model_path=str(model_path),
            device=device
        )

    def detect_and_align(self, img):
        """Detect face and return aligned tensor."""
        try:
            face_tensor = self.detector(img)

            if face_tensor is None or (hasattr(face_tensor, 'numel') and face_tensor.numel() == 0):
                return None, False

            if len(face_tensor.shape) == 4:
                face_tensor = face_tensor[0]

            return face_tensor, True

        except Exception:
            return None, False


class FallbackPipeline:
    """Fallback: just resize image to 160x160."""

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process(self, img):
        """Just resize without detection."""
        return self.transform(img), True


def parse_pairs_file(pairs_file):
    """Parse LFW pairs.txt file."""
    pairs = []
    with open(pairs_file, 'r') as f:
        lines = f.readlines()

    # First line is number of folds and pairs per fold
    num_folds, num_pairs = map(int, lines[0].strip().split())

    idx = 1
    for fold in range(num_folds):
        # Matched pairs (same person)
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

        # Mismatched pairs (different person)
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


def get_embedding(pipeline, embed_model, img, device, fallback=None):
    """Get embedding for an image using pipeline."""
    if hasattr(pipeline, 'detect_and_crop'):
        face_tensor, detected = pipeline.detect_and_crop(img)
    else:
        face_tensor, detected = pipeline.detect_and_align(img)

    if not detected and fallback:
        face_tensor, _ = fallback.process(img)

    if face_tensor is None:
        return None, detected

    with torch.no_grad():
        emb = embed_model(face_tensor.unsqueeze(0).to(device))
        emb = emb / emb.norm(dim=1, keepdim=True)

    return emb.cpu().numpy()[0], detected


def evaluate_pipeline(pipeline, embed_model, pairs, lfw_dir, device,
                     pipeline_name, use_fallback=True, max_pairs=None):
    """Evaluate a pipeline on LFW verification."""

    print(f"\n{'='*60}")
    print(f"Evaluating: {pipeline_name}")
    print(f"{'='*60}")

    fallback = FallbackPipeline() if use_fallback else None

    # Cache embeddings
    embedding_cache = {}
    detected_count = 0
    total_images = 0

    # Get unique images
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
        emb, detected = get_embedding(pipeline, embed_model, img, device, fallback)

        if emb is not None:
            embedding_cache[img_name] = emb
            total_images += 1
            if detected:
                detected_count += 1

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

    # Compute TAR@FAR
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    # TAR @ FAR=0.1%
    tar_at_far_01 = np.interp(0.001, fpr, tpr)
    # TAR @ FAR=1%
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
    parser = argparse.ArgumentParser(description='Compare detectors on LFW')
    parser.add_argument('--lfw-dir', type=str,
                       default='/home/rick/GitHub/mo-faces/data/lfw/images',
                       help='Path to LFW images')
    parser.add_argument('--pairs-file', type=str,
                       default='/home/rick/GitHub/mo-faces/data/lfw/pairs.txt',
                       help='Path to pairs.txt')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--pipelines', nargs='+',
                       default=['old', 'new'],
                       choices=['old', 'new', 'fallback'],
                       help='Which pipelines to test')
    parser.add_argument('--max-pairs', type=int, default=1000,
                       help='Max pairs to evaluate (0 for all)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    lfw_dir = Path(args.lfw_dir)
    if not lfw_dir.exists():
        print(f"ERROR: LFW directory not found: {lfw_dir}")
        return

    # Load pairs
    pairs = parse_pairs_file(args.pairs_file)
    print(f"Loaded {len(pairs)} pairs from {args.pairs_file}")

    max_pairs = args.max_pairs if args.max_pairs > 0 else None

    # Load embedding model
    print("Loading FaceNet embedding model...")
    docker_path = Path('/service/model_weights/representation/20180402-114759-facenet.pt')
    local_path = Path(__file__).parent.parent / 'models_weights' / 'representation' / '20180402-114759-facenet.pt'
    embed_model_path = docker_path if docker_path.exists() else local_path
    embed_model = InceptionResnetV1(weights_filename=str(embed_model_path), device=device).eval()

    all_results = {}

    # Test MTCNN pipeline
    if 'old' in args.pipelines:
        try:
            mtcnn_pipeline = MTCNNPipeline(device)
            if mtcnn_pipeline.available:
                results = evaluate_pipeline(
                    mtcnn_pipeline, embed_model, pairs, lfw_dir, device,
                    "MTCNN (facenet-pytorch)", max_pairs=max_pairs
                )
                all_results['MTCNN'] = results
            else:
                print("MTCNN skipped - facenet-pytorch not installed")
        except Exception as e:
            print(f"MTCNN pipeline failed: {e}")

    # Test new pipeline
    if 'new' in args.pipelines:
        try:
            new_pipeline = NewPipeline(device)
            results = evaluate_pipeline(
                new_pipeline, embed_model, pairs, lfw_dir, device,
                "ATFaceDetect + DFA (New)", max_pairs=max_pairs
            )
            all_results['ATFaceDetect+DFA'] = results
        except Exception as e:
            print(f"New pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    # Test fallback
    if 'fallback' in args.pipelines:
        class FallbackWrapper:
            def __init__(self, fb):
                self.fb = fb
            def detect_and_crop(self, img):
                return self.fb.process(img)

        fallback = FallbackPipeline()
        results = evaluate_pipeline(
            FallbackWrapper(fallback), embed_model, pairs, lfw_dir, device,
            "Direct Resize (No Detection)", use_fallback=False, max_pairs=max_pairs
        )
        all_results['Direct Resize'] = results

    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Pipeline':<25} {'Accuracy':>10} {'AUC':>8} {'TAR@1%':>10} {'Det Rate':>10}")
    print("-"*80)
    for name, res in all_results.items():
        print(f"{name:<25} {res['Accuracy']:>9.2f}% {res['AUC']:>8.4f} "
              f"{res['TAR@FAR=1%']:>9.2f}% {res['Detection Rate']:>9.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()
