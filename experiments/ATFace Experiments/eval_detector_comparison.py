#!/usr/bin/env python3
"""
Detector Comparison: MTCNN vs ATFaceDetect+DFA

Compares the old MTCNN pipeline (crop+resize) against the new
ATFaceDetect+DFA pipeline (landmark-based alignment) on TinyFace benchmark.

Both use the same FaceNet embedding model, so we're testing detection + alignment.
"""
import sys
import os
import json
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


class MTCNNPipeline:
    """MTCNN-style pipeline using facenet-pytorch for comparison."""

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

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_crop(self, image_path):
        """Detect face and return cropped tensor."""
        if not self.available:
            return None, False

        img = Image.open(image_path).convert('RGB')

        try:
            # MTCNN returns cropped face tensor directly
            face_tensor = self.mtcnn(img)

            if face_tensor is None:
                return None, False

            return face_tensor, True

        except Exception as e:
            return None, False


def get_model_dir():
    """Get model directory - handles both local and Docker paths."""
    docker_path = Path('/service/model_weights')
    if docker_path.exists():
        return docker_path
    return Path(__file__).parent.parent / 'models_weights'


class NewPipeline:
    """New ATFaceDetect + DFA alignment pipeline using FaceDetector class."""

    def __init__(self, device='cuda'):
        self.device = device

        from app.recognition import FaceDetector

        model_path = get_model_dir() / 'detection'

        if not (model_path / 'ATFaceDetect_v3.pth').exists():
            raise FileNotFoundError(f"ATFaceDetect model not found at {model_path}")

        # Use the high-level FaceDetector class
        self.detector = FaceDetector(
            image_size=160,
            keep_all=False,  # Get single best face
            model_path=str(model_path),
            device=device
        )

    def detect_and_align(self, image_path):
        """Detect face and return aligned tensor using FaceDetector."""
        img = Image.open(image_path).convert('RGB')

        try:
            # Use the forward method which handles detection + alignment
            face_tensor = self.detector(img)

            if face_tensor is None or (hasattr(face_tensor, 'numel') and face_tensor.numel() == 0):
                return None, False

            # face_tensor is already [C, H, W] normalized
            if len(face_tensor.shape) == 4:
                face_tensor = face_tensor[0]  # Take first face

            return face_tensor, True

        except Exception as e:
            return None, False


class FallbackPipeline:
    """Fallback: just resize image to 160x160 (no detection)."""

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process(self, image_path):
        """Just resize without detection."""
        img = Image.open(image_path).convert('RGB')
        return self.transform(img), True


def evaluate_pipeline(pipeline, embed_model, probe_dir, gallery_dir, protocol,
                     pipeline_name, device, use_fallback=True):
    """Evaluate a detection pipeline on TinyFace."""

    print(f"\n{'='*60}")
    print(f"Evaluating: {pipeline_name}")
    print(f"{'='*60}")

    fallback = FallbackPipeline() if use_fallback else None

    # Extract gallery embeddings
    print("Extracting gallery embeddings...")
    gallery_embeddings = []
    gallery_detected = 0

    for item in tqdm(protocol['gallery'][:500], desc="Gallery"):  # Limit for speed
        img_path = gallery_dir / item['image']
        if not img_path.exists():
            continue

        if hasattr(pipeline, 'detect_and_crop'):
            face_tensor, detected = pipeline.detect_and_crop(str(img_path))
        else:
            face_tensor, detected = pipeline.detect_and_align(str(img_path))

        if detected:
            gallery_detected += 1
        elif fallback:
            face_tensor, _ = fallback.process(str(img_path))
        else:
            continue

        with torch.no_grad():
            emb = embed_model(face_tensor.unsqueeze(0).to(device))
            emb = emb / emb.norm(dim=1, keepdim=True)

        gallery_embeddings.append({
            'embedding': emb.cpu().numpy()[0],
            'identity': item['identity']
        })

    print(f"Gallery detection rate: {gallery_detected}/{len(gallery_embeddings)} "
          f"({100*gallery_detected/max(1,len(gallery_embeddings)):.1f}%)")

    # Evaluate probes
    print("Evaluating probes...")
    r1, r5, r10 = 0, 0, 0
    total = 0
    probe_detected = 0

    for item in tqdm(protocol['probes'][:500], desc="Probes"):  # Limit for speed
        img_path = probe_dir / item['image']
        if not img_path.exists():
            continue

        if hasattr(pipeline, 'detect_and_crop'):
            face_tensor, detected = pipeline.detect_and_crop(str(img_path))
        else:
            face_tensor, detected = pipeline.detect_and_align(str(img_path))

        if detected:
            probe_detected += 1
        elif fallback:
            face_tensor, _ = fallback.process(str(img_path))
        else:
            continue

        with torch.no_grad():
            emb = embed_model(face_tensor.unsqueeze(0).to(device))
            emb = emb / emb.norm(dim=1, keepdim=True)

        probe_emb = emb.cpu().numpy()[0]
        probe_id = item['identity']
        total += 1

        # Compute similarities
        sims = [(cosine_similarity(probe_emb, g['embedding']), g['identity'])
                for g in gallery_embeddings]
        sims.sort(reverse=True, key=lambda x: x[0])

        top10 = [s[1] for s in sims[:10]]
        if probe_id in top10[:1]:
            r1 += 1; r5 += 1; r10 += 1
        elif probe_id in top10[:5]:
            r5 += 1; r10 += 1
        elif probe_id in top10:
            r10 += 1

    results = {
        'Rank-1': 100 * r1 / max(1, total),
        'Rank-5': 100 * r5 / max(1, total),
        'Rank-10': 100 * r10 / max(1, total),
        'Probe Detection Rate': 100 * probe_detected / max(1, total),
        'Gallery Detection Rate': 100 * gallery_detected / max(1, len(gallery_embeddings))
    }

    print(f"\n{pipeline_name} Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare MTCNN vs ATFaceDetect+DFA')
    parser.add_argument('--tinyface-dir', type=str,
                       default='/home/rick/GitHub/smallfacesmoe/tinyface/Testing_Set',
                       help='Path to TinyFace Testing_Set')
    parser.add_argument('--protocol', type=str,
                       default='/home/rick/GitHub/smallfacesmoe/tinyface_protocol.json',
                       help='Path to protocol JSON')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--pipelines', nargs='+',
                       default=['new'],
                       choices=['old', 'new', 'fallback'],
                       help='Which pipelines to test')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load protocol
    with open(args.protocol, 'r') as f:
        protocol = json.load(f)

    probe_dir = Path(args.tinyface_dir) / 'Probe'
    gallery_dir = Path(args.tinyface_dir) / 'Gallery_Match'

    # Load embedding model (shared by all pipelines)
    print("Loading FaceNet embedding model...")
    # Check Docker path first, then local
    docker_path = Path('/service/model_weights/representation/20180402-114759-facenet.pt')
    local_path = Path(__file__).parent.parent / 'models_weights' / 'representation' / '20180402-114759-facenet.pt'
    embed_model_path = docker_path if docker_path.exists() else local_path
    embed_model = InceptionResnetV1(weights_filename=str(embed_model_path), device=device).eval()

    all_results = {}

    # Test MTCNN pipeline (using facenet-pytorch for comparison)
    if 'old' in args.pipelines:
        try:
            mtcnn_pipeline = MTCNNPipeline(device)
            if mtcnn_pipeline.available:
                results = evaluate_pipeline(
                    mtcnn_pipeline, embed_model, probe_dir, gallery_dir, protocol,
                    "MTCNN (facenet-pytorch)", device
                )
                all_results['MTCNN'] = results
            else:
                print("MTCNN comparison skipped - facenet-pytorch not installed")
        except Exception as e:
            print(f"MTCNN pipeline failed: {e}")

    # Test new pipeline (ATFaceDetect + DFA alignment)
    if 'new' in args.pipelines:
        try:
            new_pipeline = NewPipeline(device)
            results = evaluate_pipeline(
                new_pipeline, embed_model, probe_dir, gallery_dir, protocol,
                "ATFaceDetect + DFA (New)", device
            )
            all_results['ATFaceDetect+DFA (New)'] = results
        except Exception as e:
            print(f"New pipeline failed: {e}")
            import traceback
            traceback.print_exc()

    # Test fallback (just resize, no detection)
    if 'fallback' in args.pipelines:
        fallback = FallbackPipeline()
        # Wrap fallback to have same interface
        class FallbackWrapper:
            def __init__(self, fb):
                self.fb = fb
            def detect_and_crop(self, path):
                return self.fb.process(path)

        results = evaluate_pipeline(
            FallbackWrapper(fallback), embed_model, probe_dir, gallery_dir, protocol,
            "Direct Resize (No Detection)", device, use_fallback=False
        )
        all_results['Direct Resize'] = results

    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Pipeline':<30} {'Rank-1':>10} {'Rank-5':>10} {'Det Rate':>10}")
    print("-"*70)
    for name, res in all_results.items():
        det_rate = res.get('Probe Detection Rate', 0)
        print(f"{name:<30} {res['Rank-1']:>9.2f}% {res['Rank-5']:>9.2f}% {det_rate:>9.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
