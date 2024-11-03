import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_time(file: Path):
    with open(file, "r") as f:
        text = f.read()
        time = re.search(r"Duration: ([\d\.]+)s", text, re.I)
        return float(time.group(1))  # time per step


if __name__ == "__main__":
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    times = []
    logs_dir = Path(__file__).parents[1] / "logs"
    name_map = {"cuda": "CUDA", "torch": "PyTorch"}

    for name in name_map:
        for size in sizes:
            files = logs_dir.glob(f"{name}_{size}*.log")
            for file in files:
                times.append((name_map[name], size, read_time(file)))

    df = pd.DataFrame(times, columns=["Method", "size", "time"])

    df_time = df.groupby(["Method", "size"]).mean().reset_index()
    df_cuda = df_time.query("Method == 'CUDA'").copy().set_index("size")
    df_torch = df_time.query("Method == 'PyTorch'").copy().set_index("size")
    df_time = df_cuda.join(
        df_torch, on="size", lsuffix="_cuda", rsuffix="_torch"
    ).reset_index()
    df_time["speedup"] = df_time["time_torch"] / df_time["time_cuda"]

    sns.set_style("darkgrid")

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
    sns.lineplot(data=df, x="size", y="time", hue="Method", marker="o", ax=ax1)
    ax1.set(xlabel="Hidden dim", ylabel="Run time (s)")

    sns.lineplot(data=df_time, x="size", y="speedup", marker="o", ax=ax2)
    ax2.set(xlabel="Hidden dim", ylabel="Speedup (%)", ylim=(0, None))

    fig.tight_layout()
    plt.savefig("time_graph.png", dpi=300)
