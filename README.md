# pipeline-opt

<!-- ![deploy on pypi](https://github.com/alexliap/sk_serve/actions/workflows/publish_package.yaml/badge.svg) -->
<!-- ![PyPI Version](https://img.shields.io/pypi/v/simple-serve?label=pypi%20package) -->
<!-- ![Downloads](https://static.pepy.tech/badge/simple-serve) -->
[![cov](https://alexliap.github.io/pipeline_opt/badges/coverage.svg)](https://github.com/alexliap/pipeline_opt/actions)

Utility framework in order to send your hyperparameter optimization loop to the next level. Just define the the you sklearn pipeline steps and let random search determine what is best for your problem.

### Usage

Requirement of the aforementioned claim, is that you have created a list of `yaml` files where the possilble modules' configurations are located. The package utilizes [Hydra](https://hydra.cc/) for this feature.

```bash
configs
|
|- scalers
|--- estimators.yaml
|
|- imputers
|--- estimators.yaml
|
|- models
|--- estimators.yaml
|
|- default.yaml
```

where `default.yaml` is the main configuration file directing to the rest. In each `estimators.yaml` file you should define all the modules you want to experiment with in your porject. An example of one such file as well as the `default.yaml` are shown below.

```yaml
# default.yaml
defaults:
  - _self_
  - scalers: estimators
  - imputers: estimators
  - models: estimators
```

```yaml
# models/estimators.yaml
random_forest:
  _target_: sklearn.ensemble.RandomForestClassifier
  n_estimators:
    _target_: random.randint
    a: 30
    b: 200
  criterion:
    _target_: random.choice
    seq: [gini, entropy, log_loss]
  max_depth:
    _target_: random.randint
    a: 2
    b: 12
  min_samples_split:
    _target_: random.uniform
    a: 0.01
    b: 0.2
  min_samples_leaf:
    _target_: random.uniform
    a: 0.01
    b: 0.05
  max_features:
    _target_: random.choice
    seq: [sqrt, log2]
  random_state: 0

log_reg:
  _target_: sklearn.linear_model.LogisticRegression
  penalty:
    _target_: random.choice
    seq: [l1, l2]
  fit_intercept:
    _target_: random.choice
    seq: [True, False]
  solver: saga
  max_iter:
      _target_: random.randint
      a: 200
      b: 1000

voting:
  _target_: sklearn.ensemble.VotingClassifier
  estimators: null
  voting: "soft"
  n_jobs: -1

```

Finally, in order to create a random pipeline run:

```python
from hydra import compose, initialize
from pipeline_opt import make_random_pipeline

with initialize(version_base="1.3", config_path="configs/"):
    cfg = compose(config_name="default.yaml")

make_random_pipeline(cfg, ["imputers", "scalers", "models"])
```

See the [Examples](https://github.com/alexliap/pipeline_opt/tree/master/examples) section of the repository for more complex cases.

<!-- ### Installation

The package exists on PyPI so you can install it directly to your environment by running the command:

```terminal
pip install
``` -->
