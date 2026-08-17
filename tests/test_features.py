import pandas as pd
import pytest

from src.features import (
    BASE_FEATURES,
    MODEL_FEATURES,
    add_time_series_features,
    latest_feature_row,
)
from src.training.train_model import split_by_engine


def make_history(engine_id: int, cycles: int) -> pd.DataFrame:
    rows = []
    for cycle in range(1, cycles + 1):
        row = {
            "engine_id": engine_id,
            "cycle": cycle,
            "setting_1": 0.0,
            "setting_2": 0.0,
            "setting_3": 0.0,
        }
        row.update(
            {
                f"sensor_{index}": engine_id * 100 + cycle + index
                for index in range(1, 22)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_features_do_not_mix_engines():
    data = pd.concat(
        [make_history(1, 5), make_history(2, 5)],
        ignore_index=True,
    )
    engineered = add_time_series_features(data)
    engine_two_first = engineered[
        (engineered["engine_id"] == 2) & (engineered["cycle"] == 1)
    ].iloc[0]

    assert engine_two_first["sensor_1_trend"] == 0.0
    assert engine_two_first["sensor_1_roll_mean"] == 202.0


def test_latest_feature_row_requires_five_cycles():
    history = make_history(1, 4)[BASE_FEATURES]
    with pytest.raises(ValueError, match="At least 5"):
        latest_feature_row(history)


def test_latest_feature_row_matches_model_schema():
    history = make_history(1, 5)[BASE_FEATURES]
    features = latest_feature_row(history)
    assert list(features.columns) == MODEL_FEATURES
    assert len(features) == 1


def test_engine_split_has_no_overlap():
    data = pd.concat(
        [make_history(engine, 5) for engine in range(1, 11)],
        ignore_index=True,
    )
    train_df, validation_df = split_by_engine(data)

    assert set(train_df["engine_id"]).isdisjoint(
        set(validation_df["engine_id"])
    )
