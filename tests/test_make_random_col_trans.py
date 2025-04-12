import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer

from pipeline_opt import make_random_col_trans
from pipeline_opt.error_messages import ErrorMessages


def load_test_confs() -> DictConfig:
    with initialize(version_base="1.3", config_path="test_confs/"):
        cfg = compose(config_name="default.yaml")

    return cfg


def test_make_random_col_trans():
    cfg = load_test_confs()

    obj = make_random_col_trans(
        cfg,
        name="test_col_trans",
        components=["scalers", "feature_selection"],
        columns=["no_reason"],
    )

    assert isinstance(obj, ColumnTransformer)


def test_make_random_col_trans_no_cols():
    cfg = load_test_confs()

    with pytest.raises(Exception) as exc_info:
        _ = make_random_col_trans(
            cfg,
            name="test_col_trans",
            components=["scalers", "feature_selection"],
            columns=[],
        )

    assert str(exc_info.value) == ErrorMessages.EMPTY_COLUMN_LIST
