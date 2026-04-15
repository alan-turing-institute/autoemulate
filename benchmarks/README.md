# Bechmarks

- [benchmark.py](./benchmark.py): a script with CLI for running batches of simulations with AutoEmulate for different numbers of tuningiterations
- [run_benchmark.sh](./run_benchmark.sh): runs batches of simulations enabling some parallelisation
- [plot_benchmark.ipynb](./plot_benchmark.ipynb): notebook for plotting results

## Quickstart
- Install [pueue](https://github.com/Nukesor/pueue): is included in [run_benchmark.sh](./run_benchmark.sh) and simplifies running multiple python scripts
- Start the pueue daemon:
```bash
pueue -d
```
- Activate the virtual environment (navigate to the right directory first):
```bash
source .venv/bin/activate
```
-  Run:
```bash
./run_benchmark.sh
```


