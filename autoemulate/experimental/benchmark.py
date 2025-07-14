import itertools

import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.emulators import ALL_EMULATORS
from autoemulate.experimental.simulations import SIMULATOR_FROM_STR
from autoemulate.experimental.simulations.base import Simulator


def run_benchmark(
    simulator: Simulator, n_samples: int, n_iter: int, n_splits: int, log_level: str
) -> pd.DataFrame:
    x = simulator.sample_inputs(n_samples).to(torch.float32)
    y = simulator.forward_batch(x).to(torch.float32)
    ae = AutoEmulate(
        x,
        y,
        models=ALL_EMULATORS,
        n_iter=n_iter,
        n_splits=n_splits,
        log_level=log_level,
    )
    return ae.summarise()


@click.command()
@click.option(
    "--simulators",
    type=str,
    multiple=True,
    default=["ProjectileMultioutput"],
    help="Number of samples to generate",
)
@click.option(
    "--n_samples_list",
    type=int,
    multiple=True,
    default=[10, 50, 100, 200, 500],
    help="Number of samples to generate",
)
@click.option(
    "--n_iter_list",
    type=int,
    multiple=True,
    default=[10, 50, 100, 200],
    help="Number of iterations to run",
)
@click.option(
    "--n_splits_list",
    type=int,
    multiple=True,
    default=[2, 4],
    help="Number of splits for cross-validation",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for the permutations over params",
)
@click.option("--log_level", default="progress_bar", help="Logging level")
def main(simulators, n_samples_list, n_iter_list, n_splits_list, seed, log_level):  # noqa: PLR0913
    dfs = []
    for simulator_str in simulators:
        simulator = SIMULATOR_FROM_STR[simulator_str]()
        params = list(itertools.product(n_samples_list, n_iter_list, n_splits_list))
        np.random.seed(seed)
        params = np.random.permutation(params)
        for n_samples, n_iter, n_splits in tqdm(params):
            print(
                f"Running benchmark for {simulator_str} with {n_samples} samples, "
                f"{n_iter} iterations, and {n_splits} splits"
            )
            try:
                df = run_benchmark(simulator, n_samples, n_iter, n_splits, log_level)
                df["simulator"] = simulator_str
                df["n_samples"] = n_samples
                df["n_iter"] = n_iter
                df["n_splits"] = n_splits
                dfs.append(df)
                final_df = pd.concat(dfs, ignore_index=True)
                final_df.sort_values("r2_test", ascending=False).to_csv(
                    "benchmark_results.csv", index=False
                )
            except Exception as e:
                print(f"Error raised while testing :\n{e}")


if __name__ == "__main__":
    main()
