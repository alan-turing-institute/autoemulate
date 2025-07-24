#!/bin/bash
set -e
source .venv/bin/activate

# Run the benchmark script with the specified parameters
date_time=$(date +"%Y-%m-%d_%H%M%S")
outpath="./benchmarks/data/${date_time}/"
mkdir -p "$outpath"
for simulator in Epidemic FlowProblem Projectile ProjectileMultioutput; do
  for n_iter_pair in "10 100" "150 50" "200 20"; do
    for n_splits in 5 2; do  
      n_iter_array=($n_iter_pair)
      n_iter1=${n_iter_array[0]}
      n_iter2=${n_iter_array[1]}
      echo "Running benchmark for simulator: $simulator, n_splits: $n_splits, n_iter: $n_iter1 $n_iter2"
      pueue add "python autoemulate/experimental/benchmark.py --simulators \"$simulator\" --n_splits_list \"$n_splits\" --n_iter_list \"$n_iter1\" --n_iter_list \"$n_iter2\" --log_level info --output_file \"${outpath}/benchmark_results_${simulator}_n_splits_${n_splits}_n_iter_${n_iter1}_${n_iter2}.csv\""
    done
  done
done

# Combine outputs with:
# xsv cat rows benchmarks/data/${date_time}/benchmark_*.csv > benchmark_results.csv
