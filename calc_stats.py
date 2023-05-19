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
    lines = np.loadtxt(file, delimiter=",")

    # calculate the mean and standard deviation
    # round to 4 decimal places
    # convert to percentage

    mean = round(np.mean(lines) * 100, 2)
    std = round(np.std(lines) * 100, 2)

    return mean, std


if __name__ == "__main__":
    # for every file in results/set
    # calculate the mean and standard deviation
    # print the results

    print("Statistics for the Set-baseline")
    print("==============================")
    for file in os.listdir("benchmarks/results/set"):
        # format the filename to something more readable for printing
        # remove .csv extension
        print_name = file[:-4]
        stats = calc_stats(str(f"benchmarks/results/set/{file}"))

        print(f"{print_name} statistics: {stats[0]}% +/- {stats[1]}%")
    print("\n")

    print("Statistics for the Truly Sparse Base")
    print("=====================================")
    for file in os.listdir("benchmarks/results/truly_sparse_base"):
        print_name = file[:-4]
        stats = calc_stats(str(f"benchmarks/results/truly_sparse_base/{file}"))
        print(f"{print_name} statistics: {stats[0]}% +/- {stats[1]}%")
    print("\n")

    print("Statistics for the My method (FS)")
    print("=====================")
    for file in os.listdir("benchmarks/results/truly_sparse_FS/"):
        # format the filename to something more readable for printing
        print_name = file[:-4]
        stats = calc_stats(str(f"benchmarks/results/truly_sparse_FS/{file}"))
        print(f"{print_name} statistics: {stats[0]}% +/- {stats[1]}%")
    print("\n")
