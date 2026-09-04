# SafeFall AI

## Vision-Based Elderly Fall Detection and Activity Monitoring

<p align="center">
  <b>AI-powered human pose analysis for fall detection, activity classification, and elderly safety monitoring.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Machine%20Learning-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Computer%20Vision-YOLO11--Pose-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.13-yellow?style=for-the-badge" />
</p>

<p align="center">
  <b>FA-2 Project • Artificial Intelligence — Machine Learning and Deep Learning</b><br>
  Student: <b>A Jeyaditya</b> • Registration No: <b>1000406</b>
</p>

---

## Live Application

> **SafeFall AI Streamlit Dashboard**

**Live App:** *Add deployed Streamlit URL here*

The deployed dashboard supports image and video analysis, pose visualization, activity classification, fall alerts, model evaluation, and session analytics.

---

# Project Overview

**SafeFall AI** is a vision-based monitoring system designed to detect potentially dangerous falls and classify human activity using pose estimation and machine learning.

Instead of relying entirely on raw image appearance, the system analyzes the geometry of the human body. Each video frame is processed by a pose-estimation model that extracts skeletal landmarks. These keypoints are then converted into geometric features and passed to a trained classifier.

The system recognizes five activity classes:

| Class      | Description                               |
| ---------- | ----------------------------------------- |
| `fall`     | Person detected in a fall-like posture    |
| `walking`  | Active upright movement                   |
| `sitting`  | Seated posture                            |
| `standing` | Upright stationary posture                |
| `normal`   | General non-dangerous posture or activity |

The objective is to create an AI-assisted monitoring pipeline that can identify potentially dangerous events while providing interpretable, pose-based visual feedback.

---

# Project Objective

This project was developed as part of **Formative Assessment 2 — Model Selection, Training, Evaluation and Deployment**.

The system demonstrates an end-to-end machine-learning workflow:

```text
Video / Image
      ↓
Human Detection
      ↓
YOLO11-Pose
      ↓
17 Body Keypoints
      ↓
Geometric Feature Extraction
      ↓
55-Dimensional Feature Vector
      ↓
Machine Learning Classifier
      ↓
Activity Prediction
      ↓
Fall Detection Logic
      ↓
Streamlit Monitoring Dashboard
```

The final application combines:

* Human pose estimation
* Feature engineering
* Activity classification
* Fall detection
* Model evaluation
* Interactive visualization
* Streamlit deployment

---

# Core Features

## Human Pose Estimation

SafeFall AI uses **YOLO11n-Pose** to identify human body landmarks.

For each detected person, YOLO-Pose predicts **17 anatomical keypoints**, including landmarks corresponding to:

* Head
* Shoulders
* Elbows
* Wrists
* Hips
* Knees
* Ankles

These landmarks allow the system to analyze body geometry rather than relying only on image pixels.

---

## Pose-Based Activity Classification

The detected skeleton is converted into a **55-dimensional feature vector**.

The features include:

* Normalized X and Y coordinates for all 17 keypoints
* Keypoint confidence values
* Human bounding-box aspect ratio
* Torso inclination angle
* Average knee angle
* Left and right knee-angle asymmetry

These features help the classifier identify postural differences between falling, standing, sitting, walking, and normal activity.

---

## Fall Detection

When the classifier produces a sufficiently confident fall prediction, the dashboard displays a fall alert.

The system also includes event logic to reduce repeated alerts for consecutive frames belonging to the same fall sequence.

> SafeFall AI is an academic prototype and does not automatically contact emergency services.

---

## Video Monitoring

Users can upload surveillance-style video files for analysis.

Supported formats include:

```text
.avi
.mp4
.mov
```

During processing, SafeFall AI displays:

* Detected human skeleton
* Current activity prediction
* Prediction confidence
* Processing progress
* Fall detection status
* Session analytics

AVI support is particularly relevant because the original Le2i dataset is distributed in `.avi` format.

---

## Image Analysis

Individual images can also be analyzed.

The dashboard displays:

* Original uploaded image
* YOLO pose visualization
* Detected skeleton
* Predicted activity
* Confidence score
* Class probability information when available

---

## Monitoring Analytics

During a session, the application tracks and visualizes:

* Total activity predictions
* Fall predictions and events
* Normal or non-fall activities
* Activity distribution
* Prediction confidence
* Recent activity history

The analytics section is generated from actual inference results produced during the current session.

---

# System Architecture

SafeFall AI consists of four primary stages.

```text
┌──────────────────────────────┐
│       Input Image/Video      │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│      YOLO11n-Pose Model      │
│      Human Pose Detection    │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│     17 Skeletal Keypoints    │
│   Coordinates + Confidence   │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│    Pose Feature Engineering  │
│        55-D Feature Vector   │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│   Trained Activity Classifier│
│                              │
│ Fall • Walking • Sitting     │
│ Standing • Normal            │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│     SafeFall AI Dashboard    │
│ Predictions • Alerts • Stats │
└──────────────────────────────┘
```

---

# Dataset

## Le2i Fall Detection Dataset

SafeFall AI was developed using the **Le2i Fall Detection Dataset**, a surveillance-video dataset containing staged fall and non-fall activities recorded across multiple indoor environments.

**Dataset:**
https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data

### Environments

The dataset contains recordings from six environment folders:

```text
Home_01
Home_02
Coffee_room_01
Coffee_room_02
Lecture_room
Office
```

### Dataset Structure

Videos are provided primarily in the following format:

```text
.avi
```

Each video is paired with a corresponding text annotation file containing the labelled fall-frame interval.

Typical structure:

```text
video (1).avi
video (1).txt

video (2).avi
video (2).txt
```

For non-fall videos, annotation files may contain:

```text
0 0
```

This indicates that no annotated fall interval exists.

---

# Dataset Availability

The complete raw Le2i dataset is **not included in this repository**.

The dataset is approximately **10 GB**, making it unsuitable for direct inclusion in the GitHub repository.

To reproduce the preprocessing pipeline:

```bash
kaggle datasets download -d <dataset-slug>
unzip <dataset-name>.zip -d le2i_dataset/
```

A valid Kaggle API token must first be available at:

```text
~/.kaggle/kaggle.json
```

A small number of processed visual outputs are retained in the repository for demonstration and assessment evidence.

---

# FA-1 — Data Preparation and Pose Processing

The initial preprocessing stage is implemented primarily in:

```text
fa1_pipeline.py
```

The preprocessing workflow performs the following operations.

### 1. Dataset Discovery

Each video is paired with its corresponding annotation file across all six Le2i environments.

### 2. Annotation Parsing

The ground-truth fall interval is extracted from each `.txt` annotation file.

### 3. Intelligent Frame Sampling

Frames are sampled from each video.

Sampling is intentionally biased toward annotated fall intervals because fall events are often short compared with the total video duration. Without targeted sampling, important fall frames could be missed.

### 4. Pose Detection

YOLO11-Pose is executed on sampled frames.

When multiple people are detected, the pipeline retains the highest-confidence human detection for downstream processing.

### 5. Activity Labelling

Fall labels primarily originate from the dataset's annotated fall interval.

A geometric fallback based on body orientation may assist with frames near annotation boundaries.

Remaining non-fall activities are separated using pose geometry and postural heuristics.

### 6. Image Preparation

Relevant visual data is resized to:

```text
224 × 224
```

Preprocessing also includes normalization and augmentation operations.

### 7. Dataset Splitting

FA-1 preprocessing creates a stratified dataset split for exploratory analysis and later model preparation.

---

# FA-2 — Machine Learning Model

The model-training stage is implemented through:

```text
pose_utils.py
train_classifier.py
```

---

## Feature Engineering

Each pose is transformed into a **55-dimensional numerical feature vector**.

Instead of feeding raw image pixels directly to the activity classifier, SafeFall AI represents human posture mathematically.

### Feature Composition

The vector contains information derived from:

```text
17 × X coordinates
17 × Y coordinates
17 × confidence values
+
Bounding-box aspect ratio
Torso inclination angle
Average knee angle
Knee-angle asymmetry
```

These values are designed to describe posture while reducing dependence on the person's absolute location within the frame.

---

# Model Selection

Two classification approaches are compared during training.

## Random Forest

The configuration includes:

* 200 decision trees
* Balanced class weights
* Multiclass classification
* Pose-feature input

## Multi-Layer Perceptron

A compact neural network is also evaluated.

Architecture:

```text
Input
  ↓
64 neurons
  ↓
32 neurons
  ↓
5-class output
```

The models are compared using validation performance.

The selected model is determined primarily through **macro F1-score**, ensuring that performance across all activity classes is considered rather than relying only on overall accuracy.

---

# Dataset Split

The FA-2 classification dataset is divided using a stratified split:

```text
70% Training
15% Validation
15% Testing
```

Stratification helps preserve class representation across each subset.

---

# Evaluation Metrics

SafeFall AI evaluates model performance using standard classification metrics:

| Metric        | Purpose                                              |
| ------------- | ---------------------------------------------------- |
| **Accuracy**  | Overall proportion of correct predictions            |
| **Precision** | Reliability of positive class predictions            |
| **Recall**    | Ability to identify examples belonging to each class |
| **F1-Score**  | Balance between precision and recall                 |
| **Macro F1**  | Equal-weight evaluation across all activity classes  |

Additional evaluation artifacts include:

* Confusion matrix
* Model-comparison graph
* MLP training-loss curve
* Validation performance
* Test performance
* Prediction examples

These outputs are stored under:

```text
fa2_outputs/
```

---

# Final Model Results

> Add final values after running `train_classifier.py`.

| Evaluation         |              Result |
| ------------------ | ------------------: |
| Winning Model      | `Pending final run` |
| Test Accuracy      | `Pending final run` |
| Test Macro F1      | `Pending final run` |
| Classes            |                 `5` |
| Pose Keypoints     |                `17` |
| Feature Dimensions |                `55` |

The generated training metadata can be found in:

```text
fa2_outputs/models/model_info.json
```

---

# SafeFall AI Dashboard

The project is deployed through an interactive **Streamlit dashboard**.

The dashboard contains five major sections.

## Overview

The overview provides a high-level summary of the monitoring session, including:

* Total activity predictions
* Fall activity and event count
* Non-fall activity count
* Current prediction confidence
* Activity distribution

---

## Image Analysis

Users can upload a single image and run pose-based classification.

The processing flow is:

```text
Original Image
      ↓
YOLO Pose Detection
      ↓
Skeleton Visualization
      ↓
Feature Extraction
      ↓
Activity Classification
      ↓
Confidence Result
```

---

## Video Monitoring

Uploaded videos are processed sequentially, with inference results displayed throughout the session.

The monitoring interface includes:

* Annotated frame
* Human skeleton
* Predicted activity
* Confidence score
* Processing status
* Fall-warning state

Video processing is deliberately bounded to maintain compatibility with CPU-based cloud deployment.

---

## Analytics

The analytics interface summarizes activity predictions generated during the current session.

Possible outputs include:

* Activity distribution
* Prediction history
* Fall-related events
* Confidence statistics

---

## Model Information

The model information section provides transparency into the underlying AI system.

Information displayed includes:

* Pose model
* Number of keypoints
* Feature dimensions
* Activity classes
* Classifier architecture
* Dataset information
* Evaluation outputs
* Confusion matrix
* Model comparison
* Training and loss graphs when available

---

# Fall Alert Logic

SafeFall AI does not treat every individual fall-classified frame as an independent emergency.

A fall sequence may span multiple adjacent video frames. The application therefore uses confidence thresholds and event cooldown logic to help distinguish repeated predictions from separate fall events.

This produces a more meaningful monitoring interface while preserving the underlying frame-level predictions.

---

# Privacy-Oriented Design

SafeFall AI's classification pipeline focuses on **human skeletal geometry** rather than identity.

The classifier receives pose-derived numerical features rather than personally identifying facial information.

These pose features describe properties such as:

* Joint positions
* Body orientation
* Torso angle
* Knee geometry
* Bounding-box proportions

This approach demonstrates how activity-recognition systems can reduce dependence on identifiable image appearance.

> Uploaded images and videos are still processed by the application to perform pose estimation. Therefore, SafeFall AI should be considered privacy-oriented rather than completely anonymous.

---

# Performance Optimisation

Video inference can be computationally expensive, particularly when running computer-vision models on CPU-only cloud infrastructure.

The application therefore uses several safeguards:

* Cached model loading
* Inference-only execution
* Sequential frame processing
* Controlled frame sampling
* Reduced inference resolution
* Bounded processing history
* Temporary-file cleanup
* Garbage collection where appropriate
* CPU-compatible inference

These optimizations allow the project to remain practical on local hardware and Streamlit Community Cloud.

---

# Technology Stack

| Technology                  | Role                                   |
| --------------------------- | -------------------------------------- |
| **Python**                  | Main programming language              |
| **Ultralytics YOLO11-Pose** | Human pose estimation                  |
| **PyTorch**                 | YOLO inference backend                 |
| **OpenCV**                  | Image and video processing             |
| **NumPy**                   | Numerical computation                  |
| **Pandas**                  | Data processing and analytics          |
| **Scikit-learn**            | Machine-learning models and evaluation |
| **Joblib**                  | Saving and loading trained artifacts   |
| **Matplotlib**              | Evaluation visualizations              |
| **Streamlit**               | Interactive web dashboard              |
| **Pillow**                  | Image handling                         |
| **Kaggle API**              | Dataset retrieval                      |

---

# Repository Structure

```text
1000406_Jeyaditya_AIY2_Machine_learning_Fall_detector_FA2/
│
├── app.py
│   └── Streamlit SafeFall AI dashboard
│
├── fa1_pipeline.py
│   └── Dataset preprocessing and pose extraction
│
├── train_classifier.py
│   └── Model training, comparison and evaluation
│
├── pose_utils.py
│   └── Shared pose-feature extraction utilities
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── fa1_outputs/
│   └── FA-1 preprocessing evidence and screenshots
│
├── fa2_outputs/
│   ├── models/
│   │   └── Trained model artifacts and model_info.json
│   │
│   └── screenshots/
│       └── Evaluation graphs and prediction examples
│
└── .streamlit/
    └── config.toml
```

> Exact folders may vary depending on which generated training outputs are currently committed.

---

# Running SafeFall AI Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Jeyaditya/1000406_Jeyaditya_AIY2_Machine_learning_Fall_detector_FA2.git
```

Navigate into the repository:

```bash
cd 1000406_Jeyaditya_AIY2_Machine_learning_Fall_detector_FA2
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run FA-1 Preprocessing

Configure the dataset path inside the preprocessing script before execution.

```bash
python fa1_pipeline.py
```

---

## 4. Train and Evaluate the Classifier

```bash
python train_classifier.py
```

This generates the trained classifier and associated evaluation outputs.

---

## 5. Launch SafeFall AI

```bash
streamlit run app.py
```

Streamlit will provide a local browser URL.

Typically:

```text
http://localhost:8501
```

---

# Deployment

SafeFall AI is designed for deployment through **Streamlit Community Cloud**.

Deployment entry point:

```text
app.py
```

The deployed application loads the trained model artifacts from the repository and performs inference on uploaded images and videos.

The deployment remains CPU-compatible and does not require CUDA.

---

# Supported Upload Formats

### Images

Common image formats supported by the deployed interface include:

```text
.jpg
.jpeg
.png
```

### Videos

```text
.avi
.mp4
.mov
```

AVI support is retained because the Le2i dataset itself uses AVI surveillance recordings.

---

# Model Artifacts

The deployed application relies on generated machine-learning artifacts produced by the training pipeline.

These can include:

```text
Classifier
Scaler
Label Encoder
Model Metadata
Evaluation Metrics
```

Artifacts are loaded during application startup and cached where appropriate to avoid unnecessary repeated initialization.

---

# Project Evidence

Assessment evidence is distributed across the repository.

Examples include:

### Pose Estimation

* Human detection
* 17-keypoint pose output
* Skeleton visualization

### Dataset Processing

* Video-frame sampling
* Annotation matching
* Feature extraction

### Model Training

* Model comparison
* Training outputs
* Validation results

### Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Loss curve
* Prediction screenshots

### Deployment

* Streamlit dashboard
* Image inference
* Video inference
* Fall-alert demonstration
* Analytics interface

---

# Why Pose-Based Fall Detection?

Raw-image classification requires a model to learn both relevant and irrelevant visual information.

For fall detection, many useful cues are fundamentally geometric:

```text
Is the torso nearly horizontal?
How is the body oriented?
What are the knee angles?
How wide is the person's bounding box?
Where are the hips relative to the shoulders?
```

Pose estimation converts these visual questions into numerical measurements.

For example:

```text
Camera Frame
     ↓
Person Detection
     ↓
Skeleton
     ↓
Body Geometry
     ↓
Activity Class
```

This gives SafeFall AI a more interpretable foundation for activity recognition.

---

# Example Inference Flow

Consider a single video frame.

### Stage 1

YOLO detects a person.

```text
Person Confidence → 0.91
```

### Stage 2

YOLO-Pose estimates 17 skeletal landmarks.

### Stage 3

SafeFall AI computes geometric properties.

```text
Bounding-box aspect ratio
Torso inclination
Knee angles
Pose asymmetry
Normalized joint coordinates
```

### Stage 4

These values become a:

```text
55-dimensional feature vector
```

### Stage 5

The classifier predicts:

```text
FALL
Confidence: 92%
```

### Stage 6

The dashboard displays the result and evaluates whether it satisfies the existing fall-alert logic.

---

# Learning Outcomes

This project demonstrates practical implementation of:

* Computer vision
* Human pose estimation
* Machine learning
* Neural networks
* Feature engineering
* Data preprocessing
* Video analysis
* Model comparison
* Classification metrics
* Streamlit application development
* AI system deployment

More importantly, the project demonstrates how multiple AI components can be connected into a complete end-to-end system rather than evaluated only as isolated notebook experiments.

---

# Limitations

SafeFall AI is an academic prototype.

Performance may be affected by:

* Poor lighting
* Severe occlusion
* Multiple overlapping people
* Unusual camera angles
* Low-resolution footage
* Pose-estimation failure
* Activities not represented in the training data
* Ambiguous transitional movements
* Differences between staged falls and real-world falls

A fall prediction should therefore be interpreted as an **AI-generated risk indication**, not a guaranteed medical emergency.

---

# Future Improvements

Potential future development could include:

* Temporal sequence modelling
* LSTM or Transformer-based motion analysis
* Multi-person tracking
* Improved fall-event grouping
* Edge-device deployment
* Webcam and live-camera inference
* Better low-light pose estimation
* Additional elderly-activity datasets
* More diverse training environments
* Automatic alert integration with authorized monitoring systems
* Explainable-AI visualizations
* Long-term monitoring statistics

A temporal model would be particularly valuable because falls are motion events, not merely individual static poses.

---

# Responsible Use

SafeFall AI is intended for:

* Academic research
* Machine-learning education
* Computer-vision experimentation
* Prototype elderly-monitoring research

It is **not a certified medical device** and should not be used as the sole mechanism for emergency detection or healthcare decision-making.

Human supervision and appropriate safety systems remain necessary in real-world deployments.

---

# Project Status

```text
Dataset preparation
Video/annotation pairing
Frame sampling
YOLO11-Pose integration
Pose feature extraction
Activity classification pipeline
Model comparison
Model evaluation
Streamlit dashboard
Image analysis
Video monitoring
Pose visualization
Fall-alert interface
Session analytics
Cloud deployment preparation
```

### Current Stage

> **FA-2 — Model Selection, Training, Evaluation and Deployment**

---

# Author

### A Jeyaditya

**Registration Number:** `1000406`

Artificial Intelligence — Machine Learning and Deep Learning

Project:

> **SafeFall AI — Vision-Based Elderly Fall Detection and Activity Monitoring**

---

# Final Note

SafeFall AI began as a fall-detection classification task and developed into a complete computer-vision pipeline combining:

```text
Dataset
   +
Pose Estimation
   +
Feature Engineering
   +
Machine Learning
   +
Evaluation
   +
Interactive Deployment
```

The final result is an interpretable AI prototype capable of analyzing human posture, classifying activity, visualizing skeletal movement, and flagging potentially dangerous fall events through an interactive monitoring dashboard.

---

<p align="center">
  <b>SafeFall AI</b><br>
  <i>Pose. Predict. Protect.</i>
</p>
