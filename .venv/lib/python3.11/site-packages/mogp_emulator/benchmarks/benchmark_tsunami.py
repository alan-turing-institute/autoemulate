'''
This benchmark examines the performance of emulators fit in parallel to multiple targets. The
benchmark uses a set of tsunami simulations, where the inputs are 14 values of the seafloor
displacement resulting from an earthquake, and the outputs are tsunami wave heights at
different spatial locations. This benchmark fits 8, 16, 32, and 64 output points using
1, 2, 4, and 8 processes, and records the time required per emulator to perform the fitting.
The actual performance will depend on the specific machine and the number of cores available.
Once the number of processes exceeds the number of cores on the machine, the fitting time will
increase, so the results will depend on the exact setup used. For reference, tests on a quad core
MacBook Pro found that the fitting took roughly 1 second per emulator on a single core, with the time
per emulator dropping by about a factor of 2 when 4 processes were used.
'''

import numpy as np
from mogp_emulator import MultiOutputGP, fit_GP_MAP
from time import time
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

def load_tsunami_data(n_emulators):
    "loads tsunami data from npz archive"

    assert(n_emulators > 0)

    f = np.load('tsunamidata.npz')

    assert('inputs' in f.files)
    assert('targets' in f.files)
    assert(f['inputs'].shape[0] == f['targets'].shape[1])

    return f['inputs'], f['targets'][:n_emulators]

def fit_emulators(n_emulators, processes = None):
    "load data and fit emulators for tsunami data, returning the time required to fit the emulators"

    inputs, targets = load_tsunami_data(n_emulators)

    gp = MultiOutputGP(inputs, targets)

    start_time = time()
    gp = fit_GP_MAP(gp, processes = processes)
    finish_time = time()

    return finish_time - start_time

def make_scaling_plot(n_emulators_list, process_list, execution_times):
    "create scaling plot showing time for fitting tsunami emulators"

    fig = plt.figure(figsize=(4,3))
    for n_emulators, exec_times in zip(n_emulators_list, execution_times):
        plt.plot(process_list, exec_times/float(n_emulators), '-o', label = str(n_emulators)+" emulators")
    plt.legend()
    plt.xlabel('Number of processes')
    plt.ylabel('Execution time per emulator (s)')
    plt.title("Benchmark on tsunami simulation data")
    plt.savefig('tsunami_scaling.png', bbox_inches='tight')

def run_all_models(n_emulators_list, process_list):
    "Run all sets of emulators with varying number of processes"

    execution_times = []

    for n_emulators in n_emulators_list:
        emulator_execution_times = []
        for processes in process_list:
            emulator_execution_times.append(fit_emulators(n_emulators, processes = processes))
        execution_times.append(np.array(emulator_execution_times))

    print("\n")
    print("Num. Emulators    Num. Processors    Execution Time (s)   Execution Time per Emulator (s)")
    for n_emulators, exec_times in zip(n_emulators_list, execution_times):
        for process, exec_time in zip(process_list, exec_times):
            print("{:18}{:19}{:21}{}".format(str(n_emulators), str(process), str(exec_time), str(exec_time/float(n_emulators))))

    if makeplots:
        make_scaling_plot(n_emulators_list, process_list, execution_times)

if __name__ == "__main__":
    n_emulators_list = [8, 16, 32, 64]
    process_list = [1, 2, 4, 8]
    run_all_models(n_emulators_list, process_list)
