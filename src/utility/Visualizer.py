import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Visualizer:

    def __init__(self, result_directory, do_export = False, do_show = False):

        self.dir = result_directory

        # Path where the CSV log file resides.
        self.log_filepath = os.path.join(self.dir, "log.csv")

        # Path where the folder "figures" is to be created.
        self.figures_dir = os.path.join(self.dir, "figures")

        os.makedirs(self.figures_dir, exist_ok=True)

        self.do_export = do_export
        self.do_show = do_show


    def generate_graphs(self):

        # Load CSV log file.
        df = pd.read_csv(self.log_filepath, sep=",", header=0, encoding="utf-8")

        # Project to reduce compute resources
        df = df[["Name", "Trial", "Round", "Reward", "Regret"]]

        # Sort the dataframe for convenience
        df = df.sort_values(["Name", "Trial", "Round"])

        # Compute the cumulative reward
        df["cum_reward"] = df.groupby(["Name", "Trial"])["Reward"].cumsum()
        df["cum_regret"] = df.groupby(["Name", "Trial"])["Regret"].cumsum()
        df["cum_min_regret"] = df.groupby(["Name", "Trial"])["Regret"].cummin()
        df["cum_avg_regret"] = df["cum_regret"] / df["Round"]

        data = (
            df
            .groupby(["Name", "Round"])[["cum_reward", "cum_regret", "cum_min_regret", "cum_avg_regret"]]
            .agg(
                avg_cum_reward=("cum_reward", "mean"),
                std_cum_reward=("cum_reward", "std"),
                avg_cum_regret=("cum_regret", "mean"),
                std_cum_regret=("cum_regret", "std"),
                avg_cum_min_regret=("cum_min_regret", "mean"),
                std_cum_min_regret=("cum_min_regret", "std"),
                avg_cum_avg_regret=("cum_avg_regret", "mean"),
                std_cum_avg_regret=("cum_avg_regret", "std"),
            )
            .reset_index()
        )

        # Generate The graphs
        self._generate_reward_graph(data)
        self._generate_regret_graphs(data)

    def _generate_reward_graph(self, data):

        #names = data["Name"].to_list()
        names = sorted(set(data["Name"]))

        plt.figure()

        for name in names:

            time = data.loc[data["Name"] == name, "Round"].to_numpy()
            reward = data.loc[data["Name"] == name, "avg_cum_reward"].to_numpy()
            std_reward = data.loc[data["Name"] == name, "std_cum_reward"].to_numpy()

            plt.plot(time, reward, label=f"{name}")
            plt.fill_between(time, reward - std_reward, reward + std_reward, alpha=0.3)

        plt.grid(
            True,             # enable grid
            which="both",     # draw both major and minor grids
            linestyle="--",   # dashed lines
            linewidth=0.5,    # thin
            color="gray",     # light gray
            alpha=0.3         # translucent
        )

        plt.xlabel("Round t")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward across Rounds")
        plt.legend()
        plt.tight_layout()

        self.do_export and plt.savefig(os.path.join(self.figures_dir, "cumulative_reward.png"), dpi=300, bbox_inches="tight", format="png")
        self.do_show and (plt.show())

    def _generate_regret_graphs(self, data):

        names = sorted(set(data["Name"]))

        # Cumulative Regret
        plt.figure()

        for name in names:

            time = data.loc[data["Name"] == name, "Round"].to_numpy()
            cum_regret = data.loc[data["Name"] == name, "avg_cum_regret"].to_numpy()
            std_cum_regret = data.loc[data["Name"] == name, "std_cum_regret"].to_numpy()

            plt.plot(time, cum_regret, label=f"{name}")
            plt.fill_between(time, cum_regret - std_cum_regret, cum_regret + std_cum_regret, alpha=0.3)

        plt.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
            color="gray",
            alpha=0.3
        )

        plt.xlabel("Round t")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret across Rounds")
        plt.legend()
        plt.tight_layout()

        if self.do_export:
            plt.savefig(
                os.path.join(self.figures_dir, "cumulative_regret.png"),
                dpi=300,
                bbox_inches="tight",
                format="png"
            )
        if self.do_show:
            plt.show()

        # Simple Regret
        plt.figure()

        for name in names:

            time = data.loc[data["Name"] == name, "Round"].to_numpy()
            simp_regret = data.loc[data["Name"] == name, "avg_cum_min_regret"].to_numpy()
            std_simp_regret = data.loc[data["Name"] == name, "std_cum_min_regret"].to_numpy()

            plt.plot(time, simp_regret, label=f"{name}")
            plt.fill_between(time, simp_regret - std_simp_regret, simp_regret + std_simp_regret, alpha=0.3)

        plt.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
            color="gray",
            alpha=0.3
        )

        plt.xlabel("Round t")
        plt.ylabel("Simple Regret")
        plt.title("Simple Regret across Rounds")
        plt.legend()
        plt.tight_layout()

        if self.do_export:
            plt.savefig(
                os.path.join(self.figures_dir, "simple_regret.png"),
                dpi=300,
                bbox_inches="tight",
                format="png"
            )
        if self.do_show:
            plt.show()

        # Average Regret
        plt.figure()

        for name in names:

            time = data.loc[data["Name"] == name, "Round"].to_numpy()
            avg_regret = data.loc[data["Name"] == name, "avg_cum_avg_regret"].to_numpy()
            std_avg_regret = data.loc[data["Name"] == name, "std_cum_avg_regret"].to_numpy()

            plt.plot(time, avg_regret, label=f"{name}")
            plt.fill_between(time, avg_regret - std_avg_regret, avg_regret + std_avg_regret, alpha=0.3)

        plt.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.5,
            color="gray",
            alpha=0.3
        )

        plt.xlabel("Round t")
        plt.ylabel("Average Regret")
        plt.title("Average Regret across Rounds")
        plt.legend()
        plt.tight_layout()

        if self.do_export:
            plt.savefig(
                os.path.join(self.figures_dir, "average_regret.png"),
                dpi=300,
                bbox_inches="tight",
                format="png"
            )
        if self.do_show:
            plt.show()

    def generate_heatmap(self):

        # Load CSV log file.
        df = pd.read_csv(self.log_filepath, sep=",", header=0, encoding="utf-8")

        # Project to reduce compute resources
        df = df[["Name", "Trial", "Round", "p", "k", "Regret"]]

        df["cum_regret"] = df.groupby(["Name", "Trial"])["Regret"].cumsum()

        # ["p", "k", "cum_regret"]
        final = (
            df.groupby(["Name","p","k","Trial"])
            .tail(1)                            # last round (T) of each trial
            .groupby(["p","k"])["cum_regret"]
            .mean()                             # across trials
            .reset_index()
        )

        # 2D dataframe (d, T), y-axis: k values, x-axis: p values
        grid = final.pivot(index="k", columns="p", values="cum_regret")

        fig, ax = plt.subplots()

        im = ax.imshow(grid, origin="lower", cmap = "RdYlGn_r") # k on Y, p on X
        fig.colorbar(im, ax = ax, label = "Cumulative Regret")

        p_vals = grid.columns.values
        k_vals = grid.index.values

        ax.set_xticks(np.arange(len(p_vals)))
        ax.set_xticklabels(p_vals)
        ax.set_yticks(np.arange(len(k_vals)))
        ax.set_yticklabels(k_vals)
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax.set_xlabel("Warm-up Rounds p")
        ax.set_ylabel("Selected Features k")
        ax.set_title("Hyper-parameter grid search")

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid.iat[i,j]:.1f}", ha = "center", va = "center", fontsize = 8)
        
        plt.tight_layout()

        if self.do_export:
            plt.savefig(
                os.path.join(self.figures_dir, "cumulative_regret_heatmap.png"),
                dpi=300,
                bbox_inches="tight",
                format="png"
            )
        
        if self.do_show:
            plt.show()