import random

from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline, make_union

from .error_messages import ErrorMessages


def make_estimator(
    cfg: DictConfig, object_key: str, module_name: str, **kwargs
) -> BaseEstimator:
    """Create a specific module that exists in the config file with randomized arguments.

    Args:
        cfg (DictConfig): Hydra configuration YAML file.
        object_key (str): Family of modules declared in the configuration file.
        module_name (str): Specific module to be created. The name specified must be the same as in the YAML file. Defaults to [].

    Returns:
        BaseEstimator: The randomly initialized object specified by module_name.
    """
    return instantiate(cfg[object_key][module_name], **kwargs)


def make_random_estimator(
    cfg: DictConfig, object_key: str, exclusions: list[str] = []
) -> BaseEstimator:
    """The same as the make_estimator() function, but it selects a module at random from specified
    family (object_key). The user can also add exclusions by adding their name in a list.

    Args:
        cfg (DictConfig): Hydra configuration YAML file.
        object_key (str): Family of modules declared in the configuration file.
        exclusions (list[str], optional): Modules to exclude from creating. The name specified must be the same as
                                          in the YAML file. Defaults to [].

    Returns:
        BaseEstimator: A randomly initialized object picked randomly from the object_key family.
    """
    object_name = random.choice(list(cfg[object_key].keys()))
    while object_name in exclusions:
        object_name = random.choice(list(cfg[object_key].keys()))

    return make_estimator(cfg, object_key, object_name)


def make_random_pipeline(
    cfg: DictConfig, components: list[str], exclusions: list[str] = []
) -> Pipeline:
    """Create an sklearn Pipeline with random components that are randomly intialized as well. The position of
    the components in the list also declare their position in the Pipeline. Also, only one object will be generated for each
    component.

    Args:
        cfg (DictConfig): Hydra configuration YAML file.
        components (list[str]): The list of components to include in the piepline.
        exclusions (list[str], optional): Modules to exclude from creating. The name specified must be the same as in the YAML file. Defaults to [].

    Returns:
        Pipeline: A Pipeline object with randomly created estimators selected from the specified components.
    """
    pipeline = [
        make_random_estimator(cfg, component, exclusions) for component in components
    ]
    return make_pipeline(*pipeline)


def make_random_col_trans(
    cfg: DictConfig,
    name: str,
    components: list[str],
    columns: list[str | int] | slice,
    exclusions: list[str] = [],
) -> ColumnTransformer:
    """Create a ColumnTransformer for specific columns, using a random Pipeline.

    Args:
        cfg (DictConfig): Hydra configuration YAML file.
        name (str): Given name of the respective pipeline.
        components (list[str]): The list of components to include in the piepline.
        columns (list[str | int]): Columns where the ColumnTransformer is going to be applied.
        exclusions (list[str], optional): Modules to exclude from creating. The name specified must be the same as
                                          in the YAML file. Defaults to [].

    Raises:
        Exception: ErrorMessages.EMPTY_COLUMN_LIST

    Returns:
        ColumnTransformer: A ColumnTransformer object with randomly created Pipeline.
    """
    if isinstance(columns, list) and len(columns) == 0:
        raise Exception(ErrorMessages.EMPTY_COLUMN_LIST)

    return ColumnTransformer(
        transformers=[
            (name, make_random_pipeline(cfg, components, exclusions), columns)
        ],
        n_jobs=-1,
    )


def make_random_union(
    cfg: DictConfig,
    estimators: list[list[str]],
    columns: list[list[str]],
    exclusions: list[list[str]] = [],
):
    if len(estimators) != len(columns):
        raise Exception(ErrorMessages.EQUAL_LENGTH_LISTS)

    num_estimators = len(estimators)

    transformers = []

    for i in range(num_estimators):
        if len(columns[i]) == 0:
            transformers.append(
                make_random_pipeline(
                    cfg,
                    components=estimators[i],
                    exclusions=exclusions[i] if exclusions else [],
                )
            )
        else:
            transformers.append(
                make_random_col_trans(
                    cfg,
                    name=f"estimator_{i}",
                    components=estimators[i],
                    columns=columns[i],
                    exclusions=exclusions[i] if exclusions else [],
                )
            )

    return make_union(*transformers, n_jobs=-1)


def make_random_voting(
    cfg: DictConfig,
    models_key: str,
    voting_config_name: str,
    max_number_of_est: int,
    exclusions: list[str] = [],
):
    exclusions.append(voting_config_name)
    available_estimators = list(cfg[models_key].keys())

    for est in available_estimators:
        if est in exclusions:
            available_estimators.remove(est)

    random.shuffle(available_estimators)

    num_of_estimators = random.choice(list(range(2, max_number_of_est + 1)))

    chosen_estimators = available_estimators[:num_of_estimators]

    instantiated_estimators = []
    for est in chosen_estimators:
        instantiated_estimators.append((est, make_estimator(cfg, models_key, est)))

    voting = make_estimator(cfg, models_key, voting_config_name)

    voting.set_params(estimators=instantiated_estimators)

    return voting
