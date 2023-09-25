import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def read_losses(file: Path):
    steps = []
    losses = []
    step_time = []
    with open(file, "r") as f:
        for line in f:
            step = re.search(r"step: (\d+)", line, re.I)
            loss = re.search(r"loss: ([\d\.]+)", line, re.I)
            if step is not None and loss is not None:
                steps.append(int(step.group(1)))
                losses.append(float(loss.group(1)))
    df = pd.DataFrame({"step": steps, "loss": losses})
    return df

if __name__ == "__main__":
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    logs_dir = Path(__file__).parents[1] / "logs"
    name_map = {
        "cuda": "CUDA", 
        "torch": "PyTorch (Compiled)"
    }

    dfs = []
    for name in name_map:
        for size in sizes:
            files = logs_dir.glob(f"{name}_{size}*.log")
            df = pd.concat([read_losses(file) for file in files])
            df["Hidden dim"] = str(size)
            df["Method"] = name_map[name]
            dfs.append(df)

    df = pd.concat(dfs)

    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x="step", 
        y="loss", 
        hue="Hidden dim", 
        style="Method", 
        ax=ax,
    )
    ax.set(xlabel="Step", ylabel="Loss")

    fig.tight_layout()
    # plt.show()
    plt.savefig("loss_graph.png", dpi=300)

