from hydra import compose, initialize
from hydra.utils import instantiate
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
        # instantiate estimator
        real_obj = instantiate(cfg["make_estimator"][est])

        print(type(real_obj))

        # use function to do the same thing
        obj = make_estimator(cfg=cfg, object_key="make_estimator", module_name=est)

        # chech if it is a BaseEstimaror object
        assert isinstance(obj, BaseEstimator)
        # chech if it is os the same type
        assert isinstance(obj, type(real_obj))
