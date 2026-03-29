# Early-Stage Dental Caries Detection

A FastAPI + TensorFlow project for early-stage dental caries detection from X-ray images, with a frontend that maps directly to supported backend API operations.

## 1) Project Overview

This repository contains:
- **Backend inference service** built with FastAPI.
- **Model training and prediction scripts** built with TensorFlow/Keras.
- **Frontend UI** in `frontend/index.html` for backend health, model metadata, and inference.

The frontend is directly integrated with backend APIs and only exposes interactions supported by those endpoints.

## 2) Repository Structure

```text
/home/runner/work/early-stage-dental/early-stage-dental
├── app.py                  # FastAPI server and inference endpoints
├── train.py                # Model training pipeline
├── predict.py              # Local prediction utility script
├── requirements.txt        # Python dependencies
├── best_model.h5           # Trained model file (required at runtime)
├── frontend/
│   └── index.html          # Backend-aligned frontend (HTML/CSS/JS)
└── README.md               # Technical documentation
```

## 3) Backend Technical Documentation

## 3.1 Runtime and Dependencies

- Python runtime with dependencies from `requirements.txt`:
  - `fastapi`
  - `uvicorn`
  - `tensorflow`
  - `numpy`
  - `opencv-python`
  - `python-multipart`

Install:

```bash
pip install -r requirements.txt
```

## 3.2 API Service (`app.py`)

### Initialization
- Creates a FastAPI app.
- Enables permissive CORS for development/integration.
- Loads TensorFlow model once at startup:
  - `model = tf.keras.models.load_model("best_model.h5")`

### Inference Preprocessing
`predict_image(path)` does the following:
1. Reads uploaded image using OpenCV.
2. Resizes to `224x224`.
3. Applies `mobilenet_v2.preprocess_input`.
4. Adds batch dimension and runs model prediction.
5. Returns:
   - `{"prediction": "Caries", "confidence": ...}` if probability < 0.5
   - `{"prediction": "No Caries", "confidence": ...}` otherwise

### API Endpoints

- `GET /`
  - Serves `frontend/index.html`.

- `GET /api/health`
  - Health endpoint.
  - Response: `{"status": "ok"}`.

- `GET /api/metrics`
  - Exposes backend/model metadata for UI.
  - Current response:
    ```json
    {
      "model": "mobilenet_v2",
      "status": "loaded"
    }
    ```

- `POST /api/predict`
  - Accepts multipart file upload field: `file`.
  - Runs model inference and returns prediction + confidence.

## 3.3 How to Run Backend

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:
- Frontend: `http://localhost:8000/`
- Health: `http://localhost:8000/api/health`
- Metrics: `http://localhost:8000/api/metrics`

## 4) Frontend Technical Documentation

Frontend file: `frontend/index.html`

## 4.1 Design and Layout

The frontend is intentionally minimal and includes only backend-supported elements:
- Backend status display (`/api/health`).
- Model metadata display (`/api/metrics` model and status).
- Image upload input and prediction action (`/api/predict` with multipart `file`).
- Prediction and confidence display sourced from backend response.

## 4.2 Frontend-Backend Integration

### Startup Integration
On load, frontend calls:
- `GET /api/health` to determine backend availability.
- `GET /api/metrics` to show model architecture and status.

### Inference Flow
1. User selects image (`<input type="file">`).
2. Frontend shows image preview in the X-ray panel.
3. On **Run Prediction**, frontend sends:
    - `POST /api/predict` with multipart `file`.
4. UI updates with backend response:
    - Primary finding (`Caries` / `No Caries`).
    - Confidence percentage and progress bar.

## 4.3 Frontend Responsiveness

- Desktop-first layout optimized for the provided reference.
- Media queries adjust to single-column layout for narrower screens.

## 5) Model Training and Local Prediction Scripts

## 5.1 `train.py`
Contains full training pipeline (data loading, augmentation, model training, checkpoints, and metrics visualization).

Run training:

```bash
python train.py
```

## 5.2 `predict.py`
Single-image prediction utility using the same `best_model.h5` artifact.

Run:

```bash
python predict.py
```

## 6) Validation Notes

This repository currently has no formal test framework configured (`pytest`/`unittest` suite not present).
Minimal validation performed for code safety:

```bash
python -m py_compile app.py predict.py train.py
```

## 7) Known Operational Requirements

- `best_model.h5` must exist in repository root when running `app.py`.
- Backend must be running to execute live inference from frontend.
- Uploaded file must be a valid image compatible with OpenCV.

## 8) Future Improvements (Optional)

- Persist prediction history records in a database instead of static cards.
- Add unit/integration tests for API endpoints.
- Add image cleanup strategy for temporary uploaded files.
- Add CI workflow for linting/testing and release checks.
