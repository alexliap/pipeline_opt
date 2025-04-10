from hydra import compose, initialize

from pipeline_opt.create_random import make_random_estimator, make_random_pipeline


def main():
    with initialize(version_base="1.3", config_path="random_search/"):
        cfg = compose(config_name="default.yaml")

    est = make_random_pipeline(cfg, ["num_imputers", "scalers", "models"], ["voting"])

    print(est)


main()
