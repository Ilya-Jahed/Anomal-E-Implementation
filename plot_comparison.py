"""
plot_comparison.py -- Raw Features vs. Anomal-E Embeddings comparison plots.

Full explanation, grid search rationale, and hyperparameter grids are
documented in docs/PLOTCOMPARISON_EXPLANATION.md. This script does NOT
retrain the GNN encoder -- it loads an existing checkpoint from a previous
`python main.py` run.

Run this AFTER a successful `python main.py` run (so a checkpoint exists).
"""

import itertools
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from src.data_pipeline.preprocessor import AnomalEPreprocessor
from src.data_pipeline.graph_builder import AnomalEGraphBuilder
from src.models.dgi_module import AnomalEDGI
from src.models.anomaly_detectors import AnomalEDetector
from src.engine.trainer import AnomalETrainer


ALGORITHMS = ["pca", "iforest", "cblof", "hbos"]
ALGORITHM_DISPLAY_NAMES = {"pca": "PCA", "iforest": "IF", "cblof": "CBLOF", "hbos": "HBOS"}

# See docs/PLOTCOMPARISON_EXPLANATION.md for why these grids match the
# reference Anomal-E notebook's own grid-search cells.
ALGORITHM_PARAM_GRIDS = {
    "pca":     {"n_components": [5, 10, 15, 20, 25, 30]},
    "iforest": {"n_estimators": [20, 50, 100, 150]},
    "cblof":   {"n_clusters":   [2, 3, 5, 7, 9, 10]},
    "hbos":    {"n_bins":       [5, 10, 15, 20, 25, 30]},
}
CONTAMINATION_GRID = [0.001, 0.01, 0.04, 0.05, 0.1, 0.2]


def grid_search_best_f1(model_name, fit_features, eval_features, eval_labels):
    param_grid = ALGORITHM_PARAM_GRIDS[model_name]
    param_name, param_values = next(iter(param_grid.items()))

    best_f1 = -1.0
    best_params = None

    for param_value, contamination in itertools.product(param_values, CONTAMINATION_GRID):
        kwargs = {param_name: param_value}
        detector = AnomalEDetector(model_name=model_name, contamination=contamination, **kwargs)
        detector.fit(fit_features)
        predictions = detector.predict(eval_features)
        f1 = f1_score(eval_labels, predictions, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_params = {param_name: param_value, "contamination": contamination}

    return best_f1, best_params


def run_all_algorithms(fit_features_0pct, fit_features_natural, eval_features, eval_labels):
    results = {"0pct": {}, "natural": {}}
    best_params_log = {"0pct": {}, "natural": {}}

    for algo in ALGORITHMS:
        print(f"  [0% contamination]      grid-searching {algo.upper()}...")
        f1, params = grid_search_best_f1(algo, fit_features_0pct, eval_features, eval_labels)
        results["0pct"][algo] = f1
        best_params_log["0pct"][algo] = params
        print(f"    best: F1={f1*100:.2f}%  params={params}")

        print(f"  [natural contamination] grid-searching {algo.upper()}...")
        f1, params = grid_search_best_f1(algo, fit_features_natural, eval_features, eval_labels)
        results["natural"][algo] = f1
        best_params_log["natural"][algo] = params
        print(f"    best: F1={f1*100:.2f}%  params={params}")

    return results, best_params_log


def plot_comparison(raw_results, embed_results, scenario_key, scenario_title, output_path):
    labels = [ALGORITHM_DISPLAY_NAMES[a] for a in ALGORITHMS]
    raw_scores = [raw_results[scenario_key][a] * 100 for a in ALGORITHMS]
    embed_scores = [embed_results[scenario_key][a] * 100 for a in ALGORITHMS]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_raw = ax.bar(x - width / 2, raw_scores, width, label="Raw Features", color="#ED7D31")
    bars_embed = ax.bar(x + width / 2, embed_scores, width, label="Embeddings", color="#4472C4")

    ax.set_ylabel("Macro F1-Score (%)")
    ax.set_title(f"Anomal-E Macro F1-Score {scenario_title}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()

    for bars in (bars_raw, bars_embed):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[INFO] Saved plot: {output_path}")
    plt.show()


def plot_loss_curve(loss_history, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color="#4472C4", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DGI Loss")
    ax.set_title("Anomal-E DGI Training Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[INFO] Saved plot: {output_path}")
    plt.show()


def main():
    dataset_path = "data/raw/NF-CSE-CIC-IDS2018-v2.csv"
    checkpoint_dir = os.environ.get("ANOMAL_E_CHECKPOINT_DIR", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "anomal_e_dgi.pt")
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] No checkpoint found at '{checkpoint_path}'. "
              f"Run `python main.py` first to train and save a checkpoint.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")

    # Must match the fraction main.py used to produce this checkpoint --
    # see docs/PLOTCOMPARISON_EXPLANATION.md.
    print("=== Rebuilding preprocessed data + graphs (same settings as main.py) ===")
    preprocessor = AnomalEPreprocessor()
    train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=False, fraction=0.1)

    graph_builder = AnomalEGraphBuilder()
    train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
    train_g = train_g.to(device)
    test_g = test_g.to(device)

    ndim_in = train_g.ndata['h'].shape[2]
    edims = train_g.edata['h'].shape[2]
    hidden_dim = 128
    edge_hidden_dim = 256

    dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim).to(device)
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)
    trainer = AnomalETrainer(dgi_model, detector=None, optimizer=optimizer, epochs=4000)

    print(f"[INFO] Loading checkpoint from '{checkpoint_path}'...")
    completed_epochs = trainer.load_checkpoint(checkpoint_path, map_location=device)
    print(f"[INFO] Loaded encoder trained for {completed_epochs} epochs.")

    if trainer.loss_history:
        plot_loss_curve(trainer.loss_history, os.path.join(plots_dir, "training_loss.png"))
    else:
        print("[WARNING] No loss_history found in checkpoint -- skipping loss plot.")

    dgi_model.eval()
    with torch.no_grad():
        print("[INFO] Extracting train edge embeddings from the trained encoder...")
        _, train_embeddings = dgi_model.encoder(
            train_g, train_g.ndata['h'], train_g.edata['h'], corrupt=False
        )
        print("[INFO] Extracting test edge embeddings from the trained encoder...")
        _, test_embeddings = dgi_model.encoder(
            test_g, test_g.ndata['h'], test_g.edata['h'], corrupt=False
        )

    train_raw = train_g.edata['h'].squeeze(1)
    test_raw = test_g.edata['h'].squeeze(1)

    train_labels = train_g.edata['Label'].detach().cpu().numpy()
    test_labels = test_g.edata['Label'].detach().cpu().numpy()

    benign_mask_train = (train_labels == 0)
    train_raw_benign = train_raw[torch.from_numpy(benign_mask_train)]
    train_embeddings_benign = train_embeddings[torch.from_numpy(benign_mask_train)]

    print("\n=== Evaluating: RAW FEATURES ===")
    raw_results, raw_best_params = run_all_algorithms(
        fit_features_0pct=train_raw_benign,
        fit_features_natural=train_raw,
        eval_features=test_raw,
        eval_labels=test_labels,
    )

    print("\n=== Evaluating: EMBEDDINGS ===")
    embed_results, embed_best_params = run_all_algorithms(
        fit_features_0pct=train_embeddings_benign,
        fit_features_natural=train_embeddings,
        eval_features=test_embeddings,
        eval_labels=test_labels,
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Macro F1-Score %, best found via grid search)")
    print("=" * 70)
    for scenario_key, scenario_label in [("0pct", "0% contamination (fit on benign only)"),
                                          ("natural", "Natural contamination (fit on full train set)")]:
        print(f"\n-- {scenario_label} --")
        print(f"{'Algorithm':<10} {'Raw Features':>15} {'Embeddings':>15}")
        for algo in ALGORITHMS:
            raw_f1 = raw_results[scenario_key][algo] * 100
            embed_f1 = embed_results[scenario_key][algo] * 100
            print(f"{ALGORITHM_DISPLAY_NAMES[algo]:<10} {raw_f1:>14.2f}% {embed_f1:>14.2f}%")
        print(f"\n  Best hyperparameters found (Raw Features): {raw_best_params[scenario_key]}")
        print(f"  Best hyperparameters found (Embeddings):   {embed_best_params[scenario_key]}")

    plot_comparison(raw_results, embed_results, "0pct", "(0% contamination)",
                     os.path.join(plots_dir, "comparison_0pct_contamination.png"))
    plot_comparison(raw_results, embed_results, "natural", "(natural contamination)",
                     os.path.join(plots_dir, "comparison_natural_contamination.png"))

    print(f"\n🎉 All plots saved to '{plots_dir}/' and displayed above.")


if __name__ == "__main__":
    main()