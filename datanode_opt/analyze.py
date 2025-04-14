import pandas as pd
import matplotlib.pyplot as plt


def load_performance_data():
    df_normal = pd.read_csv("perf.csv", index_col="function")
    df_opt = pd.read_csv("perf_opt.csv", index_col="function")

    df_combined = pd.DataFrame(
        {"Normal": df_normal["time"], "Optimized": df_opt["time"]}
    )

    return df_combined


def plot_total_performance(df):
    plt.figure(figsize=(10, 6))
    total_data = df.loc["total"]

    plt.bar(["Normal", "Optimized"], [total_data["Normal"], total_data["Optimized"]])

    plt.yscale("log")
    plt.title("Total Execution Time Comparison")
    plt.ylabel("Wall Time (ms)")
    plt.grid(True, which="both", ls="-", alpha=0.2)

    for i, v in enumerate([total_data["Normal"], total_data["Optimized"]]):
        plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("total_performance.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_function_performance(df):
    df_functions = df.drop("total")

    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()

    for idx, (function_name, data) in enumerate(df_functions.iterrows()):
        ax = axes[idx]

        bars = ax.bar(["Normal", "Optimized"], [data["Normal"], data["Optimized"]])

        ax.set_yscale("log")

        ax.set_title(f"{function_name}")
        ax.set_ylabel("Wall Time (ms)")

        ax.grid(True, which="both", ls="-", alpha=0.2)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:,.0f}",
                ha="center",
                va="bottom",
            )

    if len(df_functions) < len(axes):
        for idx in range(len(df_functions), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()

    plt.savefig("function_performance.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = load_performance_data()
    plot_total_performance(df)
    plot_function_performance(df)

    print(
        "Performance analysis completed. Check total_performance.png and function_performance.png"
    )


if __name__ == "__main__":
    main()
