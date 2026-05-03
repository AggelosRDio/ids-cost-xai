import pandas as pd
import os
import matplotlib.pyplot as plt

DATA_DIR = "data/nslkdd/processed"

FILES = {
    "Train": "nslkdd_train.csv",
    "Validation": "nslkdd_validation.csv",
    "Test": "nslkdd_test.csv"
}

def load_data():
    datasets = {}
    for split, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            continue

        df = pd.read_csv(path)
        datasets[split] = df

    return datasets


def compute_distribution(df):
    counts = df["macro_label"].value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(2)

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })

    return result


def print_distribution(name, dist, total):
    print(f"\n=== {name} Set ===")
    print(dist)
    print(f"Total samples: {total}")

    # Imbalance ratio
    majority = dist["count"].max()
    print("\nImbalance ratio (majority / class):")
    for cls, row in dist.iterrows():
        ratio = majority / row["count"]
        print(f"{cls:8s}: {ratio:>8.1f}x")


def plot_distribution(distributions):
    plt.figure()

    for name, dist in distributions.items():
        plt.plot(dist.index, dist["percentage"], marker='o', label=name)

    plt.title("Class Distribution Comparison (%)")
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def check_stratification(train_dist, val_dist, test_dist):
    print("\n=== Stratification Check (percentage differences) ===")

    base = train_dist["percentage"]

    for name, dist in [("Validation", val_dist), ("Test", test_dist)]:
        diff = (dist["percentage"] - base).abs()

        print(f"\n{name} vs Train (absolute % difference):")
        print(diff)

        print(f"Max deviation: {diff.max():.2f}%")


def main():
    datasets = load_data()

    distributions = {}

    # Compute & print
    for name, df in datasets.items():
        dist = compute_distribution(df)
        distributions[name] = dist
        print_distribution(name, dist, len(df))

    # Plot comparison
    plot_distribution(distributions)

    # Stratification check
    check_stratification(
        distributions["Train"],
        distributions["Validation"],
        distributions["Test"]
    )


if __name__ == "__main__":
    main()