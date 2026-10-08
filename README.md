# Predictive Maintenance MLOps

Predicts whether a CNC milling machine will fail soon from 5 sensor readings, and serves the model as an API.

| File | Purpose |
|---|---|
| `data.py` | Generates the sensor dataset (modelled on AI4I 2020) |
| `train.py` | Trains candidate models, tracks them in MLflow, registers the best as `@production` |
| `app.py` | FastAPI service: `/predict`, `/health`, `/model-info`, docs at `/docs` |
| `tests/` | Data and API tests (pytest) |
| `Dockerfile` | Container for the API |
| `.github/workflows/ci.yml` | Train, test, build and smoke-test on every push |

```bash
pip install -r requirements.txt
python train.py && python -m pytest -q
uvicorn app:app --reload                          # http://127.0.0.1:8000/docs
docker build -t machine-failure-api . && docker run -p 8000:8000 machine-failure-api
```

## Monitoring

The API exposes Prometheus metrics at `/metrics`.

The monitoring stack contains:

- Prometheus for request rate, error rate, latency and alerts
- feature PSI for input/data drift
- prediction PSI for output-distribution drift
- rolling recall using real outcomes sent to `/feedback`
- Grafana with a provisioned dashboard

Generate the model first:

    python train.py

Start the monitored stack:

    docker compose -f docker-compose.monitoring.yml up --build

Then generate all three monitoring phases:

    python -m monitoring.simulate --phase all --seconds 25 --rps 10

Open:

- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

The simulator runs:

1. baseline traffic
2. shifted sensor distributions to trigger feature drift
3. normal-looking inputs with a changed failure mechanism to demonstrate concept/model drift
