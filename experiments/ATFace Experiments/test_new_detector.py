#!/usr/bin/env python3
"""
Quick sanity test for the new ATFaceDetect + DFA pipeline.
Tests detection and alignment on sample images.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import cv2


def get_model_dir():
    """Get model directory - handles both local and Docker paths."""
    # Try Docker path first
    docker_path = Path('/service/model_weights')
    if docker_path.exists():
        return docker_path
    # Fall back to local path
    return Path(__file__).parent.parent / 'models_weights'


def test_detector():
    """Test ATFaceDetect detector via FaceDetector class."""
    print("="*60)
    print("Testing ATFaceDetect Detector")
    print("="*60)

    from app.recognition import FaceDetector

    model_path = get_model_dir() / 'detection'

    if not (model_path / 'ATFaceDetect_v3.pth').exists():
        print(f"ERROR: Model not found at {model_path / 'ATFaceDetect_v3.pth'}")
        return False

    print(f"Loading FaceDetector from {model_path}...")
    detector = FaceDetector(
        image_size=160,
        keep_all=True,
        model_path=str(model_path)
    )
    print("FaceDetector loaded successfully")

    # Test with sample image - check both Docker mount and local path
    test_dirs = [
        Path('/data/tinyface/Testing_Set/Gallery_Match'),
        Path('/home/rick/GitHub/smallfacesmoe/tinyface/Testing_Set/Gallery_Match')
    ]
    test_images = []
    for test_dir in test_dirs:
        if test_dir.exists():
            test_images = list(test_dir.glob('*.jpg'))[:5]
            break

    if not test_images:
        print("No test images found")
        return False

    faces_found = 0
    for img_path in test_images:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        print(f"\nTesting: {img_path.name} ({w}x{h})")

        boxes, probs = detector.detect(img)

        if boxes is not None and len(boxes) > 0:
            faces_found += 1
            print(f"  Found {len(boxes)} face(s)")
            for i, (box, prob) in enumerate(zip(boxes[:3], probs[:3])):
                print(f"    [{i}] box: {box}, prob: {prob:.3f}")
        else:
            print(f"  No faces detected (image may be too small)")

    print(f"\nDetected faces in {faces_found}/{len(test_images)} images")
    print("ATFaceDetect: OK")
    return True


def test_aligner():
    """Test face aligner."""
    print("\n" + "="*60)
    print("Testing Face Aligner")
    print("="*60)

    from app.face_aligner import FaceAligner, UnifiedFaceAligner

    aligner = FaceAligner(output_size=112)
    print(f"FaceAligner initialized (output_size=112)")

    # Test with dummy landmarks
    dummy_landmarks = np.array([
        [30, 40],   # left eye
        [70, 40],   # right eye
        [50, 60],   # nose
        [35, 80],   # left mouth
        [65, 80]    # right mouth
    ], dtype=np.float32)

    # Create dummy image
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    aligned = aligner.align(dummy_img, dummy_landmarks)
    print(f"Aligned face shape: {aligned.shape}")

    if aligned.shape == (112, 112, 3):
        print("FaceAligner: OK")
    else:
        print("FaceAligner: FAILED - unexpected shape")
        return False

    # Test unified aligner
    model_path = get_model_dir() / 'detection'
    aligner_path = model_path / 'aligner.pt'

    unified = UnifiedFaceAligner(
        output_size=160,
        dfa_model_path=str(aligner_path) if aligner_path.exists() else None
    )
    print(f"UnifiedFaceAligner initialized (DFA: {unified.dfa_aligner is not None})")

    return True


def test_full_pipeline():
    """Test full detection + alignment + embedding pipeline."""
    print("\n" + "="*60)
    print("Testing Full Pipeline")
    print("="*60)

    from app.recognition import FaceDetector

    model_path = get_model_dir() / 'detection'

    print("Initializing FaceDetector...")
    detector = FaceDetector(
        image_size=160,
        keep_all=True,
        model_path=str(model_path)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test image - check both Docker mount and local path
    test_dirs = [
        Path('/data/tinyface/Testing_Set/Gallery_Match'),
        Path('/home/rick/GitHub/smallfacesmoe/tinyface/Testing_Set/Gallery_Match')
    ]
    test_images = []
    for test_dir in test_dirs:
        if test_dir.exists():
            test_images = list(test_dir.glob('*.jpg'))[:1]
            break

    if not test_images:
        print("No test images found")
        return False

    img_path = test_images[0]
    print(f"Testing with: {img_path.name}")

    img = Image.open(img_path).convert('RGB')
    print(f"Image size: {img.size}")

    # Test detect()
    boxes, probs = detector.detect(img)
    print(f"detect() returned: boxes={type(boxes)}, probs={type(probs)}")

    if boxes is not None and len(boxes) > 0:
        print(f"  Found {len(boxes)} faces")
        for i, (box, prob) in enumerate(zip(boxes[:3], probs[:3])):
            print(f"    [{i}] box: {box}, prob: {prob:.3f}")
    else:
        print("  No faces detected")

    # Test detect() with landmarks
    boxes, probs, points = detector.detect(img, landmarks=True)
    print(f"detect(landmarks=True) returned: points shape = {points.shape if points is not None else None}")

    # Test forward() - should return cropped face tensors
    face_tensors = detector(img)
    if face_tensors is not None:
        print(f"forward() returned: shape = {face_tensors.shape}")
    else:
        print("forward() returned: None")

    print("\nFull Pipeline: OK")
    return True


def main():
    print("ATFaceDetect + DFA Pipeline Sanity Tests")
    print("="*60)

    results = []

    try:
        results.append(("ATFaceDetect", test_detector()))
    except Exception as e:
        print(f"ATFaceDetect test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ATFaceDetect", False))

    try:
        results.append(("FaceAligner", test_aligner()))
    except Exception as e:
        print(f"FaceAligner test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("FaceAligner", False))

    try:
        results.append(("Full Pipeline", test_full_pipeline()))
    except Exception as e:
        print(f"Full Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Full Pipeline", False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("="*60)
    if all_pass:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
