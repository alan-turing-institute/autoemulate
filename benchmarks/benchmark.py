import itertools
from typing import cast

import click
import numpy as np
import pandas as pd
import torch
from autoemulate.experimental.compare import AutoEmulate
from autoemulate.experimental.emulators import ALL_EMULATORS
from autoemulate.experimental.emulators.base import Emulator
from autoemulate.experimental.simulations import SIMULATOR_REGISTRY
from autoemulate.experimental.simulations.base import Simulator
from tqdm import tqdm


def run_benchmark(
    x: torch.Tensor, y: torch.Tensor, n_iter: int, n_splits: int, log_level: str
) -> pd.DataFrame:
    ae = AutoEmulate(
        x,
        y,
        models=cast(list[type[Emulator] | str], ALL_EMULATORS),
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
    default=[20, 50, 100, 200, 500],
    help="Number of samples to generate",
)
@click.option(
    "--n_iter_list",
    type=int,
    multiple=True,
    default=[10, 50, 100],
    help="Number of iterations to run",
)
@click.option(
    "--n_splits_list",
    type=int,
    multiple=True,
    default=[2, 5],
    help="Number of splits for cross-validation",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for the permutations over params",
)
@click.option(
    "--output_file",
    type=str,
    default="benchmark_results.csv",
    help="File name for output",
)
@click.option("--log_level", default="progress_bar", help="Logging level")
def main(  # noqa: PLR0913
    simulators, n_samples_list, n_iter_list, n_splits_list, seed, output_file, log_level
):
    print(f"Running benchmark with simulators: {simulators}")
    print(f"Number of samples: {n_samples_list}")
    print(f"Number of iterations: {n_iter_list}")
    print(f"Number of splits: {n_splits_list}")
    print(f"Seed: {seed}")
    print(f"Output file: {output_file}")
    print(f"Log level: {log_level}")
    print("-" * 50)

    dfs = []
    for simulator_str in simulators:
        # Generate samples
        simulator: Simulator = SIMULATOR_REGISTRY[simulator_str]()
        max_samples = max(n_samples_list)
        x_all = simulator.sample_inputs(max_samples, random_seed=seed).to(torch.float32)
        y_all = simulator.forward_batch(x_all).to(torch.float32)

        params = list(itertools.product(n_samples_list, n_iter_list, n_splits_list))
        np.random.seed(seed)
        params = np.random.permutation(params)
        for n_samples, n_iter, n_splits in tqdm(params):
            print(
                f"Running benchmark for {simulator_str} with {n_samples} samples, "
                f"{n_iter} iterations, and {n_splits} splits"
            )
            try:
                x = x_all[:n_samples]
                y = y_all[:n_samples]
                df = run_benchmark(x, y, n_iter, n_splits, log_level)
                df["simulator"] = simulator_str
                df["n_samples"] = n_samples
                df["n_iter"] = n_iter
                df["n_splits"] = n_splits
                dfs.append(df)
                final_df = pd.concat(dfs, ignore_index=True)
                final_df.sort_values("r2_test", ascending=False).to_csv(
                    output_file, index=False
                )
            except Exception as e:
                print(f"Error raised while testing :\n{e}")


if __name__ == "__main__":
    main()
