# ASANA-SENSE Machine Learning & Computer Vision Assets

This directory organizes all machine learning models, training datasets, notebooks, evaluation metrics, and standalone testing scripts for the ASANA-SENSE yoga posture correction platform.

## Directory Structure

```
ml/
├── models/                     # Trained model artifacts & weights
│   ├── yoga_pose_classifier.tflite    # Lightweight TFLite classifier used by PoseEngine
│   ├── best_yoga_model.keras          # Saved Keras functional model directory
│   ├── best_yoga_model_copy.keras     # HDF5/Keras model checkpoint
│   └── pose_landmarker_heavy.task     # MediaPipe Pose Landmarker Heavy Task asset
├── data/                       # Landmark datasets for training & reference poses
│   ├── train_landmarks.csv            # Normalized 33-joint coordinates for training poses
│   └── test_landmarks.csv             # Test evaluation dataset
├── notebooks/                  # Model training and experimentation
│   └── yoga_pose_classifier.ipynb    # Google Colab / Jupyter notebook for architecture & training
├── scripts/                    # Standalone computer vision test scripts
│   └── yoga_pose_detector.py          # Real-time webcam OpenCV + MediaPipe demo runner
└── evaluation/                 # Model validation reports & visual analytics
    ├── classification_report.csv      # Precision, recall, and F1-score across 8 yoga classes
    ├── confusion_matrix.png           # Multi-class confusion matrix
    ├── per_class_accuracy.png         # Bar plot of classification accuracy per asana
    ├── roc_curves.png                 # Receiver Operating Characteristic curves
    └── training_curves.png            # Loss & accuracy trajectories over training epochs
```

## Supported Asana Classes
1. `chair` (Utkatasana)
2. `cobra` (Bhujangasana)
3. `dog` (Adho Mukha Svanasana)
4. `shoulder_stand` (Sarvangasana)
5. `triangle` (Trikonasana)
6. `tree` (Vrikshasana)
7. `warrior` (Virabhadrasana)
8. `no_pose` (Transition / Out of frame)

## Integration with Backend
- **Pose Classification**: `backend/pose_engine.py` dynamically loads `ml/models/yoga_pose_classifier.tflite` (or root fallback).
- **Biomechanical Angle Verification**: `backend/reference_poses.py` computes mean/variance reference vectors from `ml/data/train_landmarks.csv` to evaluate joint deviations in real-time over WebSocket (`/ws/pose-detect`).
