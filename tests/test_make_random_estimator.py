from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

from pipeline_opt import make_random_estimator


def load_test_confs() -> DictConfig:
    with initialize(version_base="1.3", config_path="test_confs/"):
        cfg = compose(config_name="default.yaml")

    return cfg


def test_make_random_estimator():
    cfg = load_test_confs()

    # add every available type to a list
    type_list = []
    for est in cfg["make_random_estimator"]:
        type_list.append(type(instantiate(cfg["make_random_estimator"][est])))

    # make a random object
    obj = make_random_estimator(cfg, "make_random_estimator")

    # check that the created object is a BaseEstimator
    assert isinstance(obj, BaseEstimator)
    # and that is of the same type as somehting in the list
    assert type(obj) in type_list


def test_make_random_estimator_exclusions():
    cfg = load_test_confs()

    est_list = [est for est in cfg["make_random_estimator"].keys()]

    for est in est_list:
        type_list = [
            type(instantiate(cfg["make_random_estimator"][rest]))
            for rest in est_list
            if rest != est
        ]

        obj = make_random_estimator(cfg, "make_random_estimator", exclusions=[est])

        assert isinstance(obj, BaseEstimator)
        assert type(obj) in type_list
        assert not isinstance(obj, type(instantiate(cfg["make_random_estimator"][est])))
