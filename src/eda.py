"""Exploratory data analysis: saves plots and summary tables to reports/eda/."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.data import NUMERIC, load_clean, load_raw  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports", "eda")


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    raw = load_raw()
    raw.isna().sum().rename("missing").to_csv(
        os.path.join(OUT, "missing_values.csv"))
    df = load_clean()
    df.describe().T.to_csv(os.path.join(OUT, "summary_stats.csv"))

    df[NUMERIC].hist(bins=20, figsize=(12, 7), color="#4c72b0")
    plt.suptitle("Numeric feature distributions")
    _save("histograms.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    _save("correlation_heatmap.png")

    plt.figure(figsize=(5, 4))
    ax = sns.countplot(x="target", data=df, hue="target", legend=False)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_title("Class balance")
    _save("class_balance.png")

    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(15, 4))
    for col, ax in zip(NUMERIC, axes):
        sns.boxplot(x="target", y=col, data=df, hue="target",
                    legend=False, ax=ax)
    _save("features_by_target.png")
    print(f"EDA saved to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
