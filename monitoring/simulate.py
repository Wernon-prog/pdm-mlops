"""Generate baseline, data-drift and concept-drift monitoring traffic."""
import argparse
import time

import numpy as np
import requests

from data import make_data

DEFAULT_API = "http://127.0.0.1:8000"


def concept_drift_label(row):
    """A new failure mechanism absent from the training data."""
    return int(
        (
            row["air_temp_k"] < 299.2
            and row["tool_wear_min"] < 120
        )
        or (
            row["rotational_speed_rpm"] > 1800
            and row["torque_nm"] < 30
        )
    )


def run_phase(api, phase, seconds, rps, seed):
    rng = np.random.default_rng(seed)

    rows = make_data(
        n_rows=max(
            int(seconds * rps) + 20,
            120,
        ),
        seed=seed,
    )

    session = requests.Session()
    deadline = time.time() + seconds
    sent = 0

    for _, original in rows.iterrows():
        if time.time() >= deadline:
            break

        reading = original.drop(
            labels=["failure"]
        ).to_dict()

        actual_failure = int(
            original["failure"]
        )

        if phase == "data-drift":
            # Move the input distribution while preserving the
            # basic relationship between air/process temperature.
            reading["air_temp_k"] = min(
                310.0,
                reading["air_temp_k"] + 4.0,
            )

            reading["process_temp_k"] = min(
                320.0,
                reading["process_temp_k"] + 4.0,
            )

        elif phase == "concept-drift":
            # Inputs remain normal, but the real failure mechanism changes.
            actual_failure = concept_drift_label(
                reading
            )

        # About 3% bad requests so the error-rate metric has something to show.
        if rng.random() < 0.03:
            bad = dict(reading)
            bad["torque_nm"] = -5

            session.post(
                f"{api}/predict",
                json=bad,
                timeout=5,
            )

        else:
            prediction = session.post(
                f"{api}/predict",
                json=reading,
                timeout=5,
            )

            prediction.raise_for_status()

            prediction_id = prediction.json()[
                "prediction_id"
            ]

            feedback = session.post(
                f"{api}/feedback",
                json={
                    "prediction_id": prediction_id,
                    "actual_failure": actual_failure,
                },
                timeout=5,
            )

            feedback.raise_for_status()

        sent += 1
        time.sleep(1 / rps)

    print(
        f"{phase}: sent {sent} readings"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api",
        default=DEFAULT_API,
    )

    parser.add_argument(
        "--phase",
        choices=[
            "baseline",
            "data-drift",
            "concept-drift",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--rps",
        type=float,
        default=10,
    )

    args = parser.parse_args()

    phases = (
        [
            "baseline",
            "data-drift",
            "concept-drift",
        ]
        if args.phase == "all"
        else [args.phase]
    )

    for i, phase in enumerate(phases):
        run_phase(
            args.api,
            phase,
            args.seconds,
            args.rps,
            seed=100 + i,
        )

        if phase != phases[-1]:
            time.sleep(5)


if __name__ == "__main__":
    main()
