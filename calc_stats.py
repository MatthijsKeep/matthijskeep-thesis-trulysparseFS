import numpy as np
import os

def calc_stats(file):
    """For a (simple)file, calculate the mean and standard deviation
    
    Args:
        file (str): path to file (csv)
    
    Returns:
        tuple: mean and standard deviation
    """

    # load in the csv
    lines = np.loadtxt(file, delimiter=',')

    # calculate the mean and standard deviation
    # round to 4 decimal places
    # convert to percentage

    mean = round(np.mean(lines)*100, 3)
    std = round(np.std(lines)*100, 3)

    return mean, std


if __name__ == '__main__':
    # for every file in results/set 
    # calculate the mean and standard deviation
    # print the results
    for file in os.listdir('benchmarks/results/set'):
        print("Statistics for the Set-baseline")
        print("==============================")
        print(
            f"The {file} statistics: {calc_stats(str(f'benchmarks/results/set/{file}'))}"
        )
        print("\n")

    for file in os.listdir('benchmarks/results/truly_sparse_base'):
        print("Statistics for the Truly Sparse Base")
        print("=====================================")
        print(f"The {file} statistics: {calc_stats(str(f'benchmarks/results/truly_sparse_base/{file}'))}")
        print("\n")
    
    for file in os.listdir('benchmarks/results/truly_sparse_FS/'):
        print("Statistics for the My method (FS)")
        print("=====================")
        print(f"The {file} statistics: {calc_stats(str(f'benchmarks/results/truly_sparse_FS/{file}'))}")
        print("\n")
