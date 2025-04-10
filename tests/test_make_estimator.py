from hydra import compose, initialize
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

from pipeline_opt import make_estimator


def load_test_confs() -> DictConfig:
    with initialize(version_base="1.3", config_path="test_confs/"):
        cfg = compose(config_name="default.yaml")

    return cfg


def test_make_estimator():
    cfg = load_test_confs()

    for est in cfg["make_estimator"]:
        obj = make_estimator(cfg=cfg, object_key="make_estimator", module_name=est)

        assert isinstance(obj, BaseEstimator)
