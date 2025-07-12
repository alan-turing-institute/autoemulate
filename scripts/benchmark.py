import itertools

import click
import numpy as np
import pandas as pd
from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.emulators import ALL_EMULATORS
from autoemulate.experimental.simulations.projectile import ProjectileMultioutput
from tqdm import tqdm


def run_benchmark(n_samples, n_iter, n_splits, log_level) -> pd.DataFrame:
    projectile = ProjectileMultioutput()
    x = projectile.sample_inputs(n_samples).float()
    y = projectile.forward_batch(x).float()

    ae = AutoEmulate(
        x,
        y,
        models=ALL_EMULATORS,
        n_iter=n_iter,
        n_splits=n_splits,
        # log_level=log_level,
    )

    return ae.summarise()


@click.command()
@click.option(
    "--n_samples_list",
    type=list[int],
    default=[10, 50, 100, 200, 500],
    help="Number of samples to generate",
)
@click.option(
    "--n_iter_list",
    type=list[int],
    default=[10, 50, 100, 200],
    help="Number of iterations to run",
)
@click.option(
    "--n_splits_list",
    type=list[int],
    default=[2, 4],
    help="Number of splits for cross-validation",
)
@click.option("--log_level", default="info", help="Logging level")
def main(n_samples_list, n_iter_list, n_splits_list, log_level):
    """Run the benchmark for MLP and GaussianProcessExact emulators."""

    dfs = []

    params = list(itertools.product(n_samples_list, n_iter_list, n_splits_list))
    np.random.seed(43)
    params = np.random.permutation(params)
    for n_samples, n_iter, n_splits in tqdm(params):
        print(
            f"Running benchmark with {n_samples} samples, {n_iter} iterations, "
            f"and {n_splits} splits"
        )
        df = run_benchmark(n_samples, n_iter, n_splits, log_level)

        df["n_samples"] = n_samples
        df["n_iter"] = n_iter
        df["n_splits"] = n_splits
        dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.sort_values("r2", ascending=False).to_csv(
            "notebooks/benchmark_results.csv", index=False
        )


if __name__ == "__main__":
    main()
