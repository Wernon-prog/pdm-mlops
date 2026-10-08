from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

HEALTHY = {
    "air_temp_k": 300,
    "process_temp_k": 310.5,
    "rotational_speed_rpm": 1550,
    "torque_nm": 38,
    "tool_wear_min": 20,
}


def test_metrics_endpoint_is_exposed():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "pdm_api_requests_total" in response.text
    assert "pdm_feature_psi" in response.text


def test_prediction_can_receive_feedback():
    prediction = client.post(
        "/predict",
        json=HEALTHY,
    )

    assert prediction.status_code == 200

    prediction_id = prediction.json()[
        "prediction_id"
    ]

    feedback = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "actual_failure": 0,
        },
    )

    assert feedback.status_code == 200
    assert feedback.json() == {"ok": True}


def test_unknown_prediction_feedback_returns_404():
    response = client.post(
        "/feedback",
        json={
            "prediction_id": "does-not-exist",
            "actual_failure": 1,
        },
    )

    assert response.status_code == 404
