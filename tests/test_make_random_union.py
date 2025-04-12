import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.pipeline import FeatureUnion

from pipeline_opt import make_random_union
from pipeline_opt.error_messages import ErrorMessages


def load_test_confs() -> DictConfig:
    with initialize(version_base="1.3", config_path="test_confs/"):
        cfg = compose(config_name="default.yaml")

    return cfg


def test_make_random_union():
    cfg = load_test_confs()

    obj = make_random_union(
        cfg,
        estimators=[["scalers", "feature_selection"], ["encoders", "scalers"]],
        columns=[["num"], ["cat"]],
        exclusions=[],
    )

    assert isinstance(obj, FeatureUnion)
    # test case where len(columns[i]) == 0
    obj = make_random_union(
        cfg,
        estimators=[["scalers", "feature_selection"], ["models"]],
        columns=[["num"], []],
        exclusions=[],
    )

    assert isinstance(obj, FeatureUnion)


def test_make_random_union_unequal_lengths():
    cfg = load_test_confs()

    # check case where we have more ColumnTransformers than column lists
    with pytest.raises(Exception) as exc_info:
        _ = make_random_union(
            cfg,
            estimators=[["scalers", "feature_selection"], ["encoders", "scalers"]],
            columns=[["num"]],
            exclusions=[],
        )

    assert str(exc_info.value) == ErrorMessages.EQUAL_LENGTH_LISTS

    # check case where we have less ColumnTransformers than column lists
    with pytest.raises(Exception) as exc_info:
        _ = make_random_union(
            cfg,
            estimators=[["scalers", "feature_selection"]],
            columns=[["num"], ["cat"]],
            exclusions=[],
        )

    assert str(exc_info.value) == ErrorMessages.EQUAL_LENGTH_LISTS
