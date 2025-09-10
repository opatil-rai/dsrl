import numpy as np
import os

# eval_dir = "./eval_videos_one_per_hundred_90successthreshold"
eval_dir = "./eval_videos"

fail_amount = -1000

eval_arr_names = os.listdir(eval_dir)
eval_arrs = [np.load(f"{eval_dir}/{eval_arr_name}/returns.npy") for eval_arr_name in eval_arr_names]
eval_means_unfiltered = [np.mean(eval_arr) for eval_arr in eval_arrs]
eval_means_filtered = [np.mean(eval_arr[np.where(eval_arr != fail_amount)]) for eval_arr in eval_arrs]
eval_success_rates =  [np.where(eval_arr != fail_amount)[0].shape[0] / (eval_arr.shape[0]) for eval_arr in eval_arrs]

for eval_arr_name, eval_mean_unfiltered, eval_mean_filtered, sr in zip(eval_arr_names, eval_means_unfiltered, eval_means_filtered, eval_success_rates):
    print(f"{eval_arr_name}: raw: {eval_mean_unfiltered}: filtered: {eval_mean_filtered}, successrate: {sr}")

breakpoint()

### Graping stuff
import matplotlib.pyplot as plt


def plot_returns(arrays, array_names, X, Y=None):
    plt.figure(figsize=(8, 6))
    color_cycle = plt.cm.tab10.colors  # 10 nice distinct colors
    
    for i, arr in enumerate(arrays):
        arr = np.asarray(arr)
        
        if arr.size == X:
            seeds = np.arange(X)
            returns = arr
            means = arr  # same as raw values
        elif arr.size % X == 0:
            Y = arr.size // X
            seeds = np.repeat(np.arange(X), Y)
            returns = arr
            means = arr.reshape(X, Y).mean(axis=1)
        else:
            raise ValueError(f"Array {i} has unexpected size {arr.size}")
        
        color = color_cycle[i % len(color_cycle)]
        
        # scatter raw returns
        plt.scatter(seeds, returns, color=color, alpha=0.5, label=array_names[i])
        
        # plot mean with same color but different marker
        plt.scatter(np.arange(X), means, 
                    marker="D", s=80, 
                    color=color, edgecolor="black", 
                    label=f"{array_names[i]} mean")
    
    plt.xlabel("Seed")
    plt.ylabel("Return")
    plt.legend()
    plt.title("Returns per Seed")
    plt.xticks(np.arange(X))
    plt.savefig("plot.png")
    plt.show()

plot_returns(eval_arrs, eval_arr_names, X=100)