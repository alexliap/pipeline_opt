import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

from pipeline_opt import make_random_voting
from pipeline_opt.error_messages import ErrorMessages


def load_test_confs() -> DictConfig:
    with initialize(version_base="1.3", config_path="test_confs/"):
        cfg = compose(config_name="default.yaml")

    return cfg


def test_make_random_voting():
    cfg = load_test_confs()

    num_of_models = len(cfg["models"].keys())

    for num in range(2, num_of_models):
        obj = make_random_voting(cfg, "models", "voting", num)

        # check that the created object is a BaseEstimator
        assert isinstance(obj, BaseEstimator)
        # assert size of ensemble is less or equal than max_number_of_est
        assert len(obj.estimators) <= num


def test_make_random_voting_size():
    cfg = load_test_confs()

    num_of_models = len(cfg["models"].keys())

    for num in range(num_of_models + 1, num_of_models + 10):
        with pytest.raises(Exception) as exc_info:
            _ = make_random_voting(cfg, "models", "voting", num)

        assert str(exc_info.value) == ErrorMessages.ENSEMBLE_LIMIT


def test_make_random_voting_wrong_object():
    cfg = load_test_confs()

    num_of_models = len(cfg["models"].keys())

    with pytest.raises(Exception) as exc_info:
        _ = make_random_voting(cfg, "models", "log_reg", num_of_models - 2)

    assert str(exc_info.value) == ErrorMessages.ENSEMBLE_OBJECT
