import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_krr_tuning(csv_path="krr_tuning_results.csv"):
    df = pd.read_csv(csv_path)

    rmse_grid = df.pivot(index="sigma", columns="lambda_reg", values="RMSE")
    r2_grid = df.pivot(index="sigma", columns="lambda_reg", values="R2_test")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(rmse_grid, annot=True, fmt=".4f", cmap="viridis_r", ax=axes[0])
    axes[0].set_title("Kernel Ridge Tuning - RMSE")
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("sigma")

    sns.heatmap(r2_grid, annot=True, fmt=".4f", cmap="viridis", ax=axes[1])
    axes[1].set_title("Kernel Ridge Tuning - R2")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("sigma")

    plt.tight_layout()
    plt.show()


def plot_bayesian_tuning(csv_path="bayesian_tuning_results.csv"):
    df = pd.read_csv(csv_path)

    rmse_grid = df.pivot(index="sigma2", columns="alpha", values="RMSE")
    r2_grid = df.pivot(index="sigma2", columns="alpha", values="R2_test")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(rmse_grid, annot=True, fmt=".4f", cmap="viridis_r", ax=axes[0])
    axes[0].set_title("Bayesian Linear Regression Tuning - RMSE")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("sigma2")

    sns.heatmap(r2_grid, annot=True, fmt=".4f", cmap="viridis", ax=axes[1])
    axes[1].set_title("Bayesian Linear Regression Tuning - R2")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("sigma2")

    plt.tight_layout()
    plt.show()
