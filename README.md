# Early-Stage Dental Caries Classification (MobileNetV2)

This project performs binary image classification of dental X-ray images to detect early-stage caries using transfer learning with MobileNetV2.

## 1. Model Architecture

- **Base model:** MobileNetV2 (`tf.keras.applications.MobileNetV2`, `include_top=False`, `weights="imagenet"`).
- **Total layers:** 159
  - **Trainable layers:** 5
  - **Non-trainable layers:** 154 (frozen MobileNetV2 backbone)
- **Custom classification head added on top of MobileNetV2:**
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(128, activation=`relu`)
  - Dropout(0.5)
  - Dense(1, activation=`sigmoid`)

## 2. Input & Preprocessing

- **Input image size:** 224 x 224 x 3
- **Preprocessing pipeline:**
  - Resize images to 224 x 224
  - Apply `mobilenet_v2.preprocess_input` (model-specific normalization for MobileNetV2)
  - Add batch dimension before inference/training

## 3. Dataset Details

- **Source:** Kaggle dataset `tahuuanh/tooth-decay-datasetraw` (downloaded via `kagglehub`)
- **Classification type:** Binary classification
- **Number of classes:** 2
  - `caries`
  - `without_caries` (training folder) / `no-caries` (test folder)
- **Total images:** Derived directly from dataset folders at runtime (counted using `os.listdir` in `train.py`)
- **Split used by the training script:**
  - Train: dataset folder `Trianing/` (upstream source folder name)
  - Validation: dataset `test/` (used as `validation_data` during fitting)
  - Test: same `test/` generator is also used for final evaluation

## 4. Transfer Learning Strategy

- **Transfer learning used:** Yes
- **Frozen layers:** Entire MobileNetV2 backbone (`base_model.trainable = False`)
- **Fine-tuning:** Not performed in current script (no unfreezing stage)
- **Reason:** Freezing the pretrained backbone reduces training cost and overfitting risk on a relatively small domain dataset while training only a lightweight task-specific head.

## 5. Model Configuration

- **Hidden-layer activations:** ReLU (`Dense(128, activation="relu")`)
- **Output activation:** Sigmoid (`Dense(1, activation="sigmoid")`) for binary output probability
- **Loss function:** Binary cross-entropy
- **Optimizer:** Adam
- **Learning rate:** 0.0001

## 6. Training Details

- **Epochs:** 10
- **Batch size:** 32
- **Training time:** Not logged explicitly in the script
- **Hardware used:** Not explicitly pinned in code; execution depends on environment (CPU/GPU availability in TensorFlow runtime)

## 7. Training Performance

- **Tracked metrics during training:**
  - Training accuracy and loss
  - Validation accuracy and loss
- **Where shown:** Learning curves are generated from `history.history` and plotted after training.
- **Overfitting/underfitting observation:** Should be judged from the divergence/convergence of train vs validation curves per run; no hard-coded conclusion is stored in the repository.

## 8. Evaluation Metrics

- **Precision / Recall / F1-score:** Computed via `classification_report(...)` on test labels vs predictions.
- **Confusion Matrix:** Computed using `confusion_matrix(...)` and visualized with a seaborn heatmap.
- **Additional metric present in implementation:** ROC curve and AUC are also plotted for binary discrimination analysis.

## 9. Data Augmentation

Training augmentation is configured through `ImageDataGenerator` with:
- Rotation (`rotation_range=20`)
- Width shift (`width_shift_range=0.2`)
- Height shift (`height_shift_range=0.2`)
- Shear (`shear_range=0.2`)
- Zoom (`zoom_range=0.2`)
- Horizontal flipping (`horizontal_flip=True`)
- Fill mode (`fill_mode="nearest"`)

## 10. Callbacks Used

- **Early Stopping:** Not used in current implementation
- **Learning Rate Scheduler:** Not used in current implementation
- **Model Checkpoint:** Not used in current implementation

Current training runs with fixed epochs and saves the final model after training.

## 11. Framework & Libraries

- **Primary framework:** TensorFlow / Keras
- **Key libraries:**
  - NumPy
  - OpenCV (`cv2`)
  - scikit-learn (metrics)
  - Matplotlib
  - Seaborn
  - PIL
  - KaggleHub
  - FastAPI (serving/inference API)

## 12. Model Saving & Deployment

- **Model format:** HDF5 (`.h5`)
  - Training script saves model as `29_03_best_model.h5`
  - Deployment expects the trained artifact to be available as `best_model.h5`; after training, rename once with `mv 29_03_best_model.h5 best_model.h5` (or update loading paths consistently)
- **Model loading for inference:** `tf.keras.models.load_model(...)`
- **Deployment approach:**
  - FastAPI backend serves inference endpoint (`POST /api/predict`)
  - Frontend (`/`) uploads X-ray images and displays predicted class + confidence
