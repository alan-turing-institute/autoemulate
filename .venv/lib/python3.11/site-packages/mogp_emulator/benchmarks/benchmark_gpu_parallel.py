'''
This benchmark examines the performance of emulators fitting and predicting in parallel, either
using the GPU implementation of MOGP, or the CPU (Python) one.  
The benchmark uses a set of 30 6-dimensional input points, and the same number of points to predict.
Each input point has a corresponding 'target', which can be 1, 2, 4, 8, or 16 values - this number 
corresponds to the number of emulators, that will be run in parallel. 
'''

import numpy as np
import pandas as pd
import argparse
from mogp_emulator.MultiOutputGP_GPU import MultiOutputGP_GPU
from mogp_emulator import MultiOutputGP, fit_GP_MAP, LibGPGPU
import time
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False

def load_data(n_emulators):
    "loads data from npz archive"

    assert(n_emulators > 0)

    f = np.load('timingtestdata.npz')
    # checks on shape of data
    assert('inputs' in f.files)
    assert('targets' in f.files)
    assert(f['inputs'].shape[0] == f['targets'].shape[1])
    assert('predict_points' in f.files)
    assert(f["predict_points"].shape[1] == f["inputs"].shape[1])

    return f['inputs'], f['targets'][:n_emulators], f['predict_points']


def make_timing_plots(df, output_filename):
    "create plots showing times for fitting emulators and predicting new points"

    df["log_n_emulators"] = np.log2(df["n_emulators"])
    df_GPU = df[df["GPU"]==True]
    df_CPU = df[df["GPU"]==False]

    xvals = list(df_GPU.log_n_emulators)
    gpu_fit = list(df_GPU.fit_time)
    gpu_pred = list(df_GPU.predict_time)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))

    ax1.plot(xvals, gpu_fit, "bo", label="GPU")
    ax1.set_xlabel("log2(num emulators)")
    ax1.set_ylabel("Fitting time (s)")
    if len(df_CPU) > 0:
        cpu_fit = list(df_CPU.fit_time)
        ax1.plot(xvals, cpu_fit, "ro", label="CPU")
    ax1.legend(loc="upper left")

    ax2.plot(xvals, np.log10(gpu_pred), "bo", label="GPU")
    ax2.set_xlabel("log2(num emulators)")
    ax2.set_ylabel("log10(Prediction time (s))")
    if len(df_CPU) > 0:
        cpu_pred = list(df_CPU.predict_time)
        ax2.plot(xvals, np.log10(cpu_pred), "ro", label="CPU")
    ax2.legend(loc="upper left")

    plt.savefig(output_filename, bbox_inches='tight')


def run_single_test(n_emulators, use_gpu, inputs, targets, x_predict):
    """
    Run fitting and prediction, on CPU or GPU, for a specified number of emulators
    """
    print("Running test", use_gpu, n_emulators)
    if use_gpu:
        mgp = MultiOutputGP_GPU(inputs, targets)
    else:
        mgp = MultiOutputGP(inputs, targets)
    fit_start_time = time.perf_counter()
    mgp = fit_GP_MAP(mgp)
    fit_stop_time = time.perf_counter()
    fit_time = fit_stop_time - fit_start_time
    print(f"Fitting time: {fit_time:0.4f} seconds")

    predict_start_time = time.perf_counter()
    result = mgp.predict(x_predict)
    predict_stop_time  = time.perf_counter()
    predict_time = predict_stop_time - predict_start_time
    print(f"Predict time: {predict_time:0.4f} seconds")
    return fit_time, predict_time


def run_all_tests(n_emulators_list, gpu_list, num_repetitions=3):
    "Run all sets of emulators with GPU on (if possible) or off"

    df_dict = {"GPU": [], "n_emulators": [], "fit_time": [], "predict_time": []}

    for n_emulators in n_emulators_list:
        # load the data 
        inputs, targets, x_predict = load_data(n_emulators)
        for gpu in gpu_list:
            for _ in range(num_repetitions):
                try:
                    fit_time, predict_time = run_single_test(
                        n_emulators, 
                        gpu, 
                        inputs, 
                        targets, 
                        x_predict
                    )
                    df_dict["GPU"].append(gpu)
                    df_dict["n_emulators"].append(n_emulators)
                    df_dict["fit_time"].append(fit_time)
                    df_dict["predict_time"].append(predict_time)
                except:
                    print("Problem for ",n_emulators, gpu)
    df = pd.DataFrame(df_dict)
    return df
 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run timing test on CPU and GPU")
    parser.add_argument("--num_reps", type=int, help="how many times to repeat", default=3)
    parser.add_argument("--max_num_emulators", 
                        type=int, 
                        choices=[2,4,8,16,32], 
                        help="Max number of emulators to test", 
                        default=16)
    parser.add_argument("--output_png_filename", help="output image file", default="gpu_timing_plots.png")
    parser.add_argument("--output_csv_filename", help="export data as csv")
    parser.add_argument("--run_cpu", help="Run CPU version for comparison", action="store_true")
    args = parser.parse_args()
                    
    num_repetitions = args.num_reps
    n_em_max = args.max_num_emulators
    n_emulators_list = [pow(2,n) for n in range(int(np.log2(n_em_max))+1)]
    gpu_list = []
    if LibGPGPU.gpu_usable():
        gpu_list.append(True)
    else:
        print("GPU unavailable - will run CPU version only (if requested via --run_cpu)")
    if args.run_cpu:
        gpu_list.append(False)
    df = run_all_tests(n_emulators_list, gpu_list, num_repetitions)
    if args.output_csv_filename:
        df.to_csv(args.output_csv_filename)
    if makeplots and args.output_png_filename:
        make_timing_plots(df, args.output_png_filename)