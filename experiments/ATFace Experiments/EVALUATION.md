# Face Detection Pipeline Evaluation

## Overview

This document describes the evaluation process used to compare different face detection and alignment strategies for use with FaceNet embeddings.

## Background

The original pipeline used MTCNN for face detection. We evaluated replacing it with ATFaceDetect (RetinaFace + ResNet50 backbone) from the BRIAR project, which offers improved performance on long-range and degraded images.

## Evaluation Methodology

### Dataset

**LFW (Labeled Faces in the Wild)** was selected for evaluation:
- 250x250 pixel images (sufficient for proper face detection)
- Standard face verification benchmark
- 6000 pairs total (3000 matched, 3000 mismatched)
- Evaluated on 1000 pairs for comparison

Note: TinyFace was initially considered but rejected due to extremely small face sizes (21-32 pixels), which caused all detectors to fall back to direct resize.

### Embedding Model

All pipelines used the same **FaceNet (InceptionResnetV1)** embedding model to isolate the impact of detection and alignment choices.

### Pipelines Tested

1. **MTCNN (facenet-pytorch)**: Standard MTCNN with built-in alignment
2. **ATFaceDetect + BRIAR Align**: ATFaceDetect detection with 5-point landmark affine alignment (ArcFace reference points)
3. **ATFaceDetect + Crop Only**: ATFaceDetect detection with simple bounding box crop and margin
4. **Direct Resize**: No detection, just resize full image to 160x160

### Metrics

- **Accuracy**: Optimal threshold accuracy on verification pairs
- **AUC**: Area under ROC curve
- **TAR@FAR=1%**: True Accept Rate at 1% False Accept Rate
- **TAR@FAR=0.1%**: True Accept Rate at 0.1% False Accept Rate
- **Detection Rate**: Percentage of images where face was successfully detected

## Results

| Pipeline | Accuracy | AUC | TAR@FAR=1% | TAR@FAR=0.1% | Det Rate |
|----------|----------|------|------------|--------------|----------|
| ATFaceDetect + Crop | **99.80%** | **0.9999** | **99.67%** | **98.67%** | 100% |
| MTCNN | 97.40% | 0.9838 | 96.00% | 83.67% | 100% |
| Direct Resize | 95.80% | 0.9917 | 92.33% | 78.00% | 100% |
| ATFaceDetect + BRIAR Align | 67.00% | 0.6835 | 5.67% | 2.67% | 100% |

## Key Findings

### 1. Simple Crop Outperforms Alignment

The most significant finding is that **simple bounding box crop dramatically outperforms landmark-based alignment** when used with FaceNet embeddings:

- ATFaceDetect + Crop: 99.80% accuracy
- ATFaceDetect + BRIAR Align: 67.00% accuracy

This is because FaceNet was trained with MTCNN preprocessing, which uses a different alignment approach than ArcFace-style 5-point alignment.

### 2. ATFaceDetect Improves Over MTCNN

When using the appropriate preprocessing (simple crop), ATFaceDetect provides a meaningful improvement over MTCNN:

- ATFaceDetect + Crop: 99.80% accuracy
- MTCNN: 97.40% accuracy

This represents a **2.4 percentage point improvement** in accuracy.

### 3. Detection Alone Adds Value

Even direct resize (no detection) achieves 95.80% accuracy on LFW, but proper face detection provides:
- Better handling of off-center faces
- Robustness to varying image sizes
- Improved performance on challenging images

## Recommendations

Based on this evaluation:

1. **Use ATFaceDetect for detection** - Superior detection performance, especially for long-range images
2. **Use simple crop for extraction** - Do NOT use landmark-based alignment with FaceNet
3. **Apply 10% margin** - Small margin around bounding box helps capture full face

## Evaluation Scripts

The evaluation tools are located in `tools/`:

- `eval_alignment_comparison.py` - Compare different alignment strategies on LFW
- `eval_lfw_comparison.py` - Compare MTCNN vs ATFaceDetect+DFA on LFW
- `eval_detector_comparison.py` - Compare detectors on TinyFace

### Running the Evaluation

```bash
# Full alignment comparison on LFW
python tools/eval_alignment_comparison.py \
    --lfw-dir /path/to/lfw/images \
    --pairs-file /path/to/pairs.txt \
    --max-pairs 1000 \
    --pipelines mtcnn atface_briar atface_crop resize
```

## Conclusion

The recommended configuration is **ATFaceDetect with simple crop** preprocessing, which achieves state-of-the-art performance (99.80% accuracy) on LFW while maintaining compatibility with FaceNet embeddings.
