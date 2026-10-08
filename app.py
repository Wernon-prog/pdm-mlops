"""FastAPI service with Prometheus metrics and ML drift monitoring."""
import logging
import os
import threading
import time
import uuid
from collections import deque

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from data import FEATURES, make_data
from drift import psi

DEFAULT_THRESHOLD = 0.3
THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("machine-failure-api")


def load_model():
    uri = os.getenv("MODEL_URI")

    if uri:
        import mlflow.sklearn

        return mlflow.sklearn.load_model(uri), uri

    path = os.getenv("MODEL_PATH", "model.joblib")
    return joblib.load(path), path


model, MODEL_SOURCE = load_model()
log.info("Model loaded from %s", MODEL_SOURCE)

# Same deterministic data distribution used during training.
reference = make_data(n_rows=5000, seed=42)
REF_X = reference[FEATURES]

FEATURE_EDGES = {
    feature: np.unique(
        np.quantile(
            REF_X[feature],
            np.linspace(0, 1, 11),
        )
    )
    for feature in FEATURES
}

REF_PRED = model.predict_proba(REF_X)[:, 1]

PRED_EDGES = np.unique(
    np.quantile(
        REF_PRED,
        np.linspace(0, 1, 11),
    )
)

# ------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------

REQUESTS = Counter(
    "pdm_api_requests_total",
    "API requests",
    ["endpoint", "status"],
)

LATENCY = Histogram(
    "pdm_api_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=(
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
    ),
)

PRED_SCORE = Histogram(
    "pdm_prediction_score",
    "Predicted machine-failure probability",
    buckets=(
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
)

FEATURE_PSI = Gauge(
    "pdm_feature_psi",
    "PSI of live feature values versus the training reference",
    ["feature"],
)

PRED_PSI = Gauge(
    "pdm_prediction_psi",
    "PSI of live prediction scores versus the training reference",
)

ROLLING_RECALL = Gauge(
    "pdm_model_rolling_recall",
    "Recall over recent predictions that received ground-truth feedback",
)

LABELS = Counter(
    "pdm_labels_received_total",
    "Ground-truth feedback records received",
)

POSITIVE_LABELS = Counter(
    "pdm_positive_labels_received_total",
    "Positive ground-truth feedback records received",
)

MODEL_INFO = Gauge(
    "pdm_model_info",
    "Model currently being served",
    ["source"],
)

MODEL_INFO.labels(source=MODEL_SOURCE).set(1)

# Short rolling window so the demo clearly moves between phases.
WINDOW = 180
MIN_DRIFT_SAMPLES = 100
MIN_POSITIVE_LABELS = 10

recent = {
    feature: deque(maxlen=WINDOW)
    for feature in FEATURES
}

recent_pred = deque(maxlen=WINDOW)

# prediction_id -> predicted class
pending = {}

# (predicted, actual)
labelled = deque(maxlen=300)

lock = threading.Lock()
n_seen = 0


app = FastAPI(
    title="Machine Failure Prediction API",
    version="1.1.0",
    description=(
        "Predicts machine failure and exposes "
        "service/model monitoring metrics."
    ),
)

app.mount("/metrics", make_asgi_app())


class SensorReading(BaseModel):
    air_temp_k: float = Field(
        ...,
        ge=290,
        le=310,
        description="Air temperature (K)",
        examples=[300.0],
    )

    process_temp_k: float = Field(
        ...,
        ge=300,
        le=320,
        description="Process temperature (K)",
        examples=[310.5],
    )

    rotational_speed_rpm: float = Field(
        ...,
        ge=1000,
        le=2500,
        description="Spindle speed",
        examples=[1550],
    )

    torque_nm: float = Field(
        ...,
        ge=0,
        le=100,
        description="Torque (Nm)",
        examples=[62.0],
    )

    tool_wear_min: float = Field(
        ...,
        ge=0,
        le=300,
        description="Tool wear (minutes)",
        examples=[230],
    )


class Prediction(BaseModel):
    prediction_id: str
    failure_probability: float
    failure_predicted: bool
    recommended_action: str


class Feedback(BaseModel):
    prediction_id: str
    actual_failure: int = Field(..., ge=0, le=1)


def update_drift_metrics():
    for feature in FEATURES:
        if len(recent[feature]) >= MIN_DRIFT_SAMPLES:
            value = psi(
                REF_X[feature].to_numpy(),
                np.asarray(recent[feature]),
                FEATURE_EDGES[feature],
            )

            FEATURE_PSI.labels(
                feature=feature
            ).set(value)

    if len(recent_pred) >= MIN_DRIFT_SAMPLES:
        value = psi(
            REF_PRED,
            np.asarray(recent_pred),
            PRED_EDGES,
        )

        PRED_PSI.set(value)


@app.middleware("http")
async def observe_http(request: Request, call_next):
    """Capture traffic, errors and latency for every API endpoint."""
    started = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response

    finally:
        # Do not let Prometheus scraping itself inflate API traffic.
        if request.url.path.rstrip("/") != "/metrics":
            REQUESTS.labels(
                endpoint=request.url.path,
                status=str(status),
            ).inc()

            LATENCY.labels(
                endpoint=request.url.path
            ).observe(
                time.perf_counter() - started
            )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_source": MODEL_SOURCE,
        "threshold": THRESHOLD,
        "features": FEATURES,
    }


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading):
    global n_seen

    X = pd.DataFrame(
        [reading.model_dump()]
    )[FEATURES]

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    predicted = probability >= THRESHOLD
    prediction_id = str(uuid.uuid4())

    if probability >= 0.7:
        action = "Stop the machine and inspect now"

    elif predicted:
        action = "Schedule maintenance this shift"

    else:
        action = "No action needed"

    with lock:
        for feature in FEATURES:
            recent[feature].append(
                float(X.iloc[0][feature])
            )

        recent_pred.append(probability)

        pending[prediction_id] = int(predicted)

        # Prevent unbounded memory growth if feedback never arrives.
        if len(pending) > 5000:
            pending.pop(next(iter(pending)))

        n_seen += 1

        if n_seen % 25 == 0:
            update_drift_metrics()

    PRED_SCORE.observe(probability)

    log.info(
        "predict p=%.3f failure=%s",
        probability,
        predicted,
    )

    return Prediction(
        prediction_id=prediction_id,
        failure_probability=round(probability, 4),
        failure_predicted=predicted,
        recommended_action=action,
    )


@app.post("/feedback")
def feedback(item: Feedback):
    """Join a real machine outcome back to an earlier prediction."""

    with lock:
        predicted = pending.pop(
            item.prediction_id,
            None,
        )

        if predicted is None:
            raise HTTPException(
                status_code=404,
                detail="unknown prediction_id",
            )

        labelled.append(
            (
                predicted,
                item.actual_failure,
            )
        )

        positives = [
            pred
            for pred, actual in labelled
            if actual == 1
        ]

        if len(positives) >= MIN_POSITIVE_LABELS:
            recall = sum(positives) / len(positives)
            ROLLING_RECALL.set(recall)

    LABELS.inc()

    if item.actual_failure == 1:
        POSITIVE_LABELS.inc()

    return {"ok": True}
