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

**Live App:** [Link to travel to the live app](https://1000406jeyadityaaiy2machinelearningfalldetectorfa2-85wg5vqkbwx.streamlit.app/)

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
SafeFall AI: Vision-Based Elderly Fall Detection and Activity Monitoring

Live Application

SafeFall AI Dashboard: Streamlit Community Cloud Deployment (Update with deployed URL)

The deployed dashboard supports single-image analysis, surveillance-style video stream analysis, real-time skeleton overlay, 5-class posture classification, temporal fall incident grouping, model transparency inspectability, and session analytics.

Project Overview

Falls represent a critical safety hazard for elderly individuals, particularly when living independently or when immediate caregiver intervention is inaccessible. SafeFall AI is an end-to-end vision-based monitoring pipeline designed to automatically detect falls and classify routine human activities from visual input (images and video).

Instead of relying on direct pixel-level RGB classification—which easily overfits to background decor, illumination, furniture, or camera vantage points—SafeFall AI abstracts visual scenes into human skeletal geometry.

Pose. Predict. Protect.


The system locates persons and extracts 17 anatomical keypoints via YOLO11n-Pose, converts these landmarks into an engineered 55-dimensional feature representation, scales them, and classifies posture via an optimized Random Forest classifier before passing raw frame predictions into temporal event clustering logic.

Target Activity Classes

Class

Description

Clinical / Safety Context

fall

Person detected in a horizontal, sudden, or fallen posture

Immediate safety risk requiring event grouping and alert triggers

walking

Dynamic upright ambulation

Routine ambulatory movement

sitting

Seated posture (chair, sofa, or resting position)

Normal sedentary activity

standing

Upright stationary posture

Stable vertical equilibrium

normal

General non-hazardous postures or neutral body orientations

Baseline unclassified everyday activity

## Project Objectives

Developed under Formative Assessment 2 (FA-2) — Model Selection, Training, Evaluation, and Deployment, the primary goals were:

Human Detection & Pose Extraction: Detect human subjects and extract reliable 17-keypoint skeletons across varied camera angles.

Pose Feature Engineering: Derive scale- and position-invariant geometric features (angles, bounding-box ratios, confidence weights).

Model Selection & Evaluation: Benchmark classical ML (Random Forest) against Deep Learning (Multi-Layer Perceptron) using macro-weighted evaluation metrics.

Temporal Event Aggregation: Prevent alarm fatigue by distinguishing consecutive frame-level fall predictions from singular real-world fall incidents.

Production Engineering & Deployment: Build a hardened, interactive Streamlit interface capable of bounded CPU inference, large file processing, and collision-free session handling.

## System Architecture

The pipeline strictly decouples visual perception, geometric feature extraction, machine learning classification, and downstream event alert logic:

                     IMAGE / VIDEO INPUT (.jpg, .png, .mp4, .avi, .mov)
                                     │
                                     ▼
                      YOLO11n-Pose Pretrained Model
                                     │
                                     ▼
                          17 COCO Skeletal Keypoints
                    (X, Y Normalized Coords + Confidences)
                                     │
                                     ▼
                         Pose Feature Engineering
                     (Joint Angles, Aspect Ratio, Tilt)
                                     │
                                     ▼
                            55-D Feature Vector
                                     │
                                     ▼
                         StandardScaler Normalization
                                     │
                                     ▼
                          Random Forest Classifier
                                (200 Trees)
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
    Activity Classification   Confidence Score       Temporal Fall Logic
      (5 Target Classes)       (Per-Frame %)       (Cooldown & Deduplication)
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                   Interactive Streamlit Monitoring Dashboard
               (Overlay, Analytics, Video Timeline, Telemetry)


## Core Features & Engineering Details

1. 55-Dimensional Geometric Feature Representation

Rather than operating on arbitrary coordinate grids, the skeletal representation is mapped to a 55-dimensional feature vector invariant to subject distance from camera and frame position:

$$\text{Total Features} = 17_X + 17_Y + 17_{\text{conf}} + 4_{\text{geometric}} = 55$$

Keypoint Coordinates ($17 \times 2 = 34$ features): Normalized $X$ and $Y$ coordinates for all 17 COCO keypoints relative to bounding box boundaries.

Keypoint Confidences ($17$ features): Model confidence scores for each individual joint detection.

Geometric & Postural Features ($4$ features):

Bounding-Box Aspect Ratio ($\frac{\text{Width}}{\text{Height}}$): Captures the sudden horizontal shift that accompanies ground contact.

Torso Inclination Angle: Angular deviation of the spine (vector between hip center and shoulder center) relative to the vertical axis.

Average Knee Angle: Joint flexion across both legs to separate seated postures from recumbent postures.

Knee-Angle Asymmetry: Differential angle between left and right knees to capture gait and uneven collapse.

2. Machine Learning Pipeline & Model Selection

Two distinct architectures were developed and benchmarked during FA-2:

Random Forest (Champion Model):

200 decision trees.

Balanced class weights (class_weight='balanced') to mitigate severe class imbalance between daily movements and falls.

Inputs preprocessed using StandardScaler.

Multi-Layer Perceptron (MLP Benchmark):

Architecture: $\text{Input}(55) \to \text{Dense}(64, \text{ReLU}) \to \text{Dense}(32, \text{ReLU}) \to \text{Output}(5, \text{Softmax})$.

Evaluated across cross-entropy loss convergence and validation macro-F1.

3. Fall Event Grouping & Cooldown Logic

Frame-by-frame classifiers generate independent predictions for every frame. In a 30 FPS surveillance recording, an individual remaining on the floor for 5 seconds generates 150 consecutive fall outputs.

Cooldown Grouping: SafeFall AI applies an event-clustering threshold that binds proximate frame predictions into a single Grouped Fall Incident.

Alert Triggering: Fall alerts trigger only on confident cluster onsets, eliminating multiple alert dispatches for one prolonged fall.

4. Enterprise-Grade Robustness Safeguards

During stress testing, the application incorporated several operational safeguards:

Unique Temporary Filenames: Prevents file collisions between concurrent or sequential video uploads using dynamic temporary paths with automatic garbage collection.

Stream Rewinding: Uploaded file streams are explicitly rewound (.seek(0)) prior to disk writes and OpenCV frame decoding.

Unknown Frame Count Resilience: Robust handling of truncated or streaming video headers where OpenCV reports negative or zero total frame counts.

Deduplicated Image Processing: Caching hashes of uploaded imagery to prevent redundant model passes and duplicated session history.

Video Confidence Isolation: Video statistics compute confidence distributions strictly within the active media file context rather than polluting historical session state.

Large File Support: Configured for video files up to 150 MB (tested against raw 70 MB uncompressed .avi files). Uses headless OpenCV (opencv-python-headless) for headless cloud deployment.

## Dataset Specifications

SafeFall AI was trained and validated using the Le2i Fall Detection Dataset acquired from the University of Burgundy (IMVIA Laboratory) / Kaggle mirrors.

Primary Sources:

Le2i IMVIA Research Portal

Kaggle Dataset Mirror

Recorded Environments:

Coffee_room_01

Coffee_room_02

Home_01

Home_02

Lecture_room

Office

Data Organization:

Videos are provided in .avi format, accompanied by ground-truth annotation .txt files specifying start and end frames of fall events.

Non-fall control videos feature 0 0 annotation tags.

The raw dataset (~10 GB) is excluded from the repository.

## Development Stages: FA-1 to FA-2

                       FA-1: PREPROCESSING & EXTRACTION
 [Raw Videos & Annotations] ──► [Fall-Biased Sampling] ──► [YOLO11-Pose] ──► [55-D Keypoint Extraction]
                                                                                        │
                                                                                        ▼
                         FA-2: ML MODELING & DEPLOYMENT
 [Stratified Split (70/15/15)] ──► [StandardScaler] ──► [Random Forest (200)] ──► [Streamlit App Deployment]


FA-1: Data Preparation Pipeline (fa1_pipeline.py)

Dataset Discovery & Pairing: Maps every raw video to its ground-truth fall duration text file.

Annotation Parsing: Resolves exact frame intervals representing initial instability, collapse, and recumbency.

Fall-Biased Sampling: Implements non-uniform frame extraction, oversampling brief fall segments while subsampling static backgrounds.

YOLO Pose Extraction: Runs YOLO11n-Pose on extracted frames, selecting the highest-confidence bounding box if multiple individuals appear.

Postural Heuristics: Assigns activity classes using dataset labels augmented with geometric fallback heuristics for boundary frames.

Feature Matrix Formulation: Structures vectors into normalized .npy / .csv arrays for model consumption.

FA-2: Machine Learning & Validation (train_classifier.py)

Stratified Splitting: 70% Train, 15% Validation, 15% Test, preserving class frequency distributions.

Model Benchmark: Optimization of Random Forest hyperparameters vs. MLP training curves.

Exportable Artifacts: Automated compilation of model.joblib, scaler.joblib, and model_info.json.

Evaluation Metrics & Validation

## Performance Metrics

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}}, \quad \text{Macro F1} = \frac{1}{N}\sum_{i=1}^{N}\text{F1}_i$$

Evaluated primarily on Macro F1-Score due to class imbalance between frequent activities (standing/walking) and infrequent events (falling).

Metric

Target / Specification

Evaluated Classes

5 (fall, walking, sitting, standing, normal)

Input Representation

55 Numerical Features (17 Keypoints + 4 Geometric Metrics)

Primary Classifier

Random Forest Classifier (200 Estimators)

Validation Stratification

70% Training / 15% Validation / 15% Testing

QA Verification Suite

65 checks executed, 0 failures

QA & Robustness Verification Summary

Prior to deployment, the application was verified across an exhaustive QA battery:

Static Code Verification: Syntax integrity and typing compliance.

Unit Tests: Geometric feature transformation logic and edge cases (e.g., zero keypoint detections).

Pipeline Real-Inference Tests: Validated with authentic image data, confirming end-to-end execution through StandardScaler and RandomForest without mock shortcuts.

Codec & Media Tests: Full validation across .mp4, .mov, and legacy uncompressed .avi surveillance feeds.

AppTest Flows: Headless simulation of Streamlit sessions, UI state retention, and cooldown transitions.

Technology Stack

Component

Library / Tool

Function

Language

Python 3.13

Core development platform

Pose Estimation

Ultralytics YOLO11n-Pose

Human landmark and bounding box extraction

Inference Backend

PyTorch / Torchvision

Neural execution backend for YOLO

Computer Vision

OpenCV (opencv-python-headless)

Frame decoding, resizing, and graphical overlay rendering

Feature Processing

NumPy & Pandas

Vector manipulation and dataset structuring

Machine Learning

Scikit-learn

Scalers, Random Forest, MLP, metrics computation

Model Serialization

Joblib

Saving and rapid loading of pipeline weights

Visualization

Matplotlib & Seaborn

Confusion matrices and loss curve generation

Web Interface

Streamlit

Interactive dashboard and telemetry visualization

Dataset Source

Kaggle API / Le2i IMVIA

Video acquisition and ground-truth pairing

## Repository Structure

SafeFall-AI/
│
├── app.py                      # Main Streamlit monitoring dashboard
├── pose_utils.py               # 55-D feature engineering and geometric calculation utilities
├── fa1_pipeline.py             # FA-1: Raw dataset parsing, frame extraction, & pose modeling
├── train_classifier.py         # FA-2: Model training, evaluation, comparison, & export
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── .streamlit/
│   └── config.toml             # Server configurations & 150MB upload limits
│
├── fa1_outputs/                # Preprocessing artifacts, sample crops, and validation logs
│
└── fa2_outputs/
    ├── models/
    │   ├── model.joblib        # Trained Random Forest classifier
    │   ├── scaler.joblib       # Fitted StandardScaler
    │   └── model_info.json     # Architecture metadata and training hyperparameters
    └── screenshots/            # Confusion matrices, validation graphs, and UI previews


## Local Installation & Setup

1. Clone the Repository

git clone https://github.com/Jeyaditya/1000406_Jeyaditya_AIY2_Machine_learning_Fall_detector_FA2.git
cd 1000406_Jeyaditya_AIY2_Machine_learning_Fall_detector_FA2


2. Initialize Virtual Environment

# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate


3. Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt


4. (Optional) Run Dataset Pipeline & Retrain Models

To run data preparation from raw Le2i footage:

# Requires Kaggle API key at ~/.kaggle/kaggle.json
python fa1_pipeline.py
python train_classifier.py


5. Launch the Dashboard

streamlit run app.py


The application will be accessible at http://localhost:8501.

Dashboard Walkthrough

┌────────────────────────────────────────────────────────────────────────┐
│ SafeFall AI: Vision-Based Monitoring Dashboard                         │
├───────────────────┬────────────────────────────────────────────────────┤
│ Navigation:       │ Telemetry Display:                                 │
│ • Overview        │ [ Live Frame with YOLO Skeleton Overlay ]          │
│ • Image Analysis  │ Status: FALL DETECTED (Confidence: 94.2%)          │
│ • Video Monitor   │ Grouped Fall Incidents: 1                          │
│ • Analytics       │ Active Class Distribution: [Bar Chart]             │
│ • Model Info      │ Invariant Pose Metrics: Aspect Ratio: 1.84         │
└───────────────────┴────────────────────────────────────────────────────┘


Overview: High-level summary of active monitoring sessions, total frames analyzed, fall alerts triggered, and active model status.

Image Analysis: Direct single-frame diagnostic mode. Upload an image (.jpg, .png), view 17-keypoint overlays, and evaluate 5-class probability distributions.

Video Monitoring: Continuous surveillance simulation supporting .avi, .mp4, and .mov files with real-time pose tracking, dynamic timeline generation, and event cooldown alerts.

Analytics: Session history visualization displaying confidence drift, activity classification distribution, and event timestamps.

Model Information: Architectural documentation exposing confusion matrices, keypoint mappings, training loss curves, and environment profiles.

Ethical Considerations & Responsible Use

Disclaimer: SafeFall AI is an academic prototype and engineering proof-of-concept developed for educational evaluation under FA-2. It is not a certified medical diagnostic device or emergency dispatch system.

Privacy-First Design: Unlike raw facial recognition systems, SafeFall AI discards biometric facial identities once keypoints are extracted. Predictions depend exclusively on skeletal topology.

Human Oversight: Vision-based fall detection models can produce false positives (e.g., rapid lying down, floor exercises) or false negatives (heavy occlusion). This software is intended to assist human caregivers, not replace continuous human vigilance.

Deployment Safeguards: Real-world implementations must secure visual input streams and enforce access controls to safeguard patient privacy in residential care environments.

Project Status

[x] Le2i dataset acquisition and multi-environment annotation parsing

[x] Targeted fall-biased frame sampling pipeline

[x] YOLO11n-Pose integration and 17-keypoint extraction

[x] 55-dimensional scale- and translation-invariant feature engineering

[x] Model exploration: Random Forest vs. Multi-Layer Perceptron (MLP)

[x] Final model serialization (StandardScaler + RandomForest)

[x] Temporal fall-event grouping and alarm cooldown logic

[x] Interactive Streamlit monitoring dashboard

[x] Legacy .avi, .mp4, and .mov media compatibility

[x] Comprehensive QA testing (65 passed checks, 0 failures)

[x] Cloud deployment readiness on Streamlit Community Cloud

Author & Academic Metadata

Author: A Jeyaditya

Registration Number: 1000406

Curriculum: Artificial Intelligence — Machine Learning and Deep Learning (FA-2)

Project Name: SafeFall AI — Vision-Based Elderly Fall Detection and Activity Monitoring# SafeFall AI

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
