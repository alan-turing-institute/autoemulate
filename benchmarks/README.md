# Benchmarks

- [benchmark.py](./benchmark.py): a script with CLI for benchmarking AutoEmulate on a given simulator and set of parameters.
- [run_benchmark.sh](./run_benchmark.sh): runs batches of benchmarks across simulators and parameters, enabling some parallelisation.
- [plot_benchmark.ipynb](./plot_benchmark.ipynb): notebook for plotting results

## Quickstart

Install [pueue](https://github.com/Nukesor/pueue). It is included in [run_benchmark.sh](./run_benchmark.sh) and simplifies running multiple python scripts.

Start the pueue daemon in the background:
```bash
pueued -d
```

Activate the virtual environment (navigate to the repo root first):
```bash
source .venv/bin/activate
```

From the repo root, run:
```bash
./benchmarks/run_benchmark.sh
```

The script will run all benchmarks in the background, using pueue to manage the jobs. It will save the results in `benchmarks/results` as the jobs complete. Note that it takes a long time to complete all the benchmarks. You can check the progress of the jobs using:
```bash
pueue status
```

When finished, shut down the pueue daemon:
```bash
pueue shutdown
```