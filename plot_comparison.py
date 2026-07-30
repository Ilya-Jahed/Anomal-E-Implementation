"""
plot_comparison.py -- Raw Features vs. Anomal-E Embeddings comparison plots.

Reproduces the style of Fig. 5-8 / Tables 3-6 of the Anomal-E paper: for each
of the four anomaly detectors (PCA, IF, CBLOF, HBOS), compares Macro F1-Score
when the detector is fit/evaluated on:
    - "Raw features": the flow feature vector 'h' built by AnomalEPreprocessor,
      BEFORE it ever goes through the E-GraphSAGE encoder.
    - "Embeddings": the 256-dim edge embeddings produced by the trained
      E-GraphSAGE encoder (loaded from a checkpoint saved by main.py/trainer.py).

Two contamination scenarios are covered, matching the paper's two separate
experiments:
    - 0% contamination: the detector is fit ONLY on benign (Label == 0)
      training rows/embeddings.
    - Natural contamination: the detector is fit on the full training set,
      whatever fraction of attacks it naturally contains after preprocessing
      (no separate "inject exactly 4%" step -- this project's train split
      already contains whatever real attack ratio survived downsampling).

This script does NOT retrain the GNN encoder -- it loads the already-trained
model from the checkpoint saved during a previous `python main.py` run, then
only re-runs the cheap classical detectors (PCA/HBOS/CBLOF/IForest), which is
what makes it fast to iterate on the plots without waiting through DGI
training again.

Run this AFTER a successful `python main.py` run (so a checkpoint exists).
"""

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


def fit_and_score(model_name, contamination, fit_features, eval_features, eval_labels):
    """
    Fits one AnomalEDetector on `fit_features` (unsupervised -- no labels
    used here, matching the paper's protocol) and evaluates it on
    `eval_features` against `eval_labels`, returning the Macro F1-score.

    Args:
        model_name: one of 'pca', 'iforest', 'cblof', 'hbos'.
        contamination: contamination parameter passed to AnomalEDetector.
        fit_features: (n_train_samples, feat_dim) array/tensor to fit on.
        eval_features: (n_test_samples, feat_dim) array/tensor to evaluate on.
        eval_labels: (n_test_samples,) ground-truth 0/1 labels for eval_features.

    Returns:
        float: Macro F1-score of the fitted detector's predictions on
            eval_features vs. eval_labels.
    """
    detector = AnomalEDetector(model_name=model_name, contamination=contamination)
    detector.fit(fit_features)
    predictions = detector.predict(eval_features)
    return f1_score(eval_labels, predictions, average="macro")


def run_all_algorithms(fit_features_0pct, fit_features_natural, eval_features, eval_labels,
                        contamination_natural):
    """
    Runs all four algorithms for both contamination scenarios on one feature
    space (either raw features or embeddings -- the caller decides which by
    what it passes in for the *_features arguments).

    Returns:
        dict: {"0pct": {"pca": f1, "iforest": f1, ...},
               "natural": {"pca": f1, "iforest": f1, ...}}
    """
    results = {"0pct": {}, "natural": {}}
    for algo in ALGORITHMS:
        print(f"  [0% contamination]      fitting {algo.upper()}...")
        results["0pct"][algo] = fit_and_score(
            algo, contamination=0.001, fit_features=fit_features_0pct,
            eval_features=eval_features, eval_labels=eval_labels,
        )
        print(f"  [natural contamination] fitting {algo.upper()}...")
        results["natural"][algo] = fit_and_score(
            algo, contamination=contamination_natural, fit_features=fit_features_natural,
            eval_features=eval_features, eval_labels=eval_labels,
        )
    return results


def plot_comparison(raw_results, embed_results, scenario_key, scenario_title, output_path):
    """
    Draws one bar chart (matching Fig. 5-8 of the paper): one group of bars
    per algorithm, with "Raw Features" and "Embeddings" side by side within
    each group, Macro F1-Score (%) on the y-axis.

    Args:
        raw_results: dict from run_all_algorithms() for raw features.
        embed_results: dict from run_all_algorithms() for embeddings.
        scenario_key: "0pct" or "natural" -- which sub-dict to plot.
        scenario_title: human-readable title suffix, e.g. "(0% contamination)".
        output_path: file path (.png) to save the figure to.
    """
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
    """
    Plots the DGI training loss curve (loss_history from AnomalETrainer),
    one point per epoch.
    """
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

    # --- Rebuild the exact same data + graph the checkpoint was trained on ---
    # IMPORTANT: this MUST use the same fraction/sanity_check settings as the
    # main.py run that produced the checkpoint, otherwise the graph structure
    # (and therefore the encoder's expected input dimensions) will not match
    # what was saved. Keep this in sync with main.py's
    # preprocessor.process_pipeline(...) call.
    print("=== Rebuilding preprocessed data + graphs (same settings as main.py) ===")
    preprocessor = AnomalEPreprocessor()
    train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=False, fraction=0.05)

    graph_builder = AnomalEGraphBuilder()
    train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
    train_g = train_g.to(device)
    test_g = test_g.to(device)

    # --- Load the trained encoder from checkpoint ---
    ndim_in = train_g.ndata['h'].shape[2]
    edims = train_g.edata['h'].shape[2]
    hidden_dim = 128
    edge_hidden_dim = 256

    dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim).to(device)
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)  # required by load_checkpoint's API
    trainer = AnomalETrainer(dgi_model, detector=None, optimizer=optimizer, epochs=50)

    print(f"[INFO] Loading checkpoint from '{checkpoint_path}'...")
    completed_epochs = trainer.load_checkpoint(checkpoint_path, map_location=device)
    print(f"[INFO] Loaded encoder trained for {completed_epochs} epochs.")

    # --- Plot the loss curve from this checkpoint's training history ---
    if trainer.loss_history:
        plot_loss_curve(trainer.loss_history, os.path.join(plots_dir, "training_loss.png"))
    else:
        print("[WARNING] No loss_history found in checkpoint -- skipping loss plot.")

    # --- Extract GNN edge embeddings (train + test), without retraining ---
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

    # --- Raw features: the 'h' vector BEFORE the GNN, straight from the graph ---
    # Shape (E, 1, edims) -> squeeze the middle dimension to (E, edims), matching
    # what AnomalEDetector expects (same shape convention as the embeddings).
    train_raw = train_g.edata['h'].squeeze(1)
    test_raw = test_g.edata['h'].squeeze(1)

    # --- Labels (edge order is guaranteed aligned with features -- see
    # trainer.py's evaluate() docstring for why labels are always read
    # straight from edata rather than a separately-supplied array) ---
    train_labels = train_g.edata['Label'].detach().cpu().numpy()
    test_labels = test_g.edata['Label'].detach().cpu().numpy()

    natural_contamination = float(np.clip(train_labels.mean(), 0.001, 0.5))
    print(f"[INFO] Natural attack contamination in training data: {natural_contamination * 100:.2f}%")

    # --- 0%-contamination fit sets: keep ONLY benign (Label == 0) training rows ---
    benign_mask_train = (train_labels == 0)
    train_raw_benign = train_raw[torch.from_numpy(benign_mask_train)]
    train_embeddings_benign = train_embeddings[torch.from_numpy(benign_mask_train)]

    # --- Run all 4 algorithms x 2 contamination scenarios, for both feature spaces ---
    print("\n=== Evaluating: RAW FEATURES ===")
    raw_results = run_all_algorithms(
        fit_features_0pct=train_raw_benign,
        fit_features_natural=train_raw,
        eval_features=test_raw,
        eval_labels=test_labels,
        contamination_natural=natural_contamination,
    )

    print("\n=== Evaluating: EMBEDDINGS ===")
    embed_results = run_all_algorithms(
        fit_features_0pct=train_embeddings_benign,
        fit_features_natural=train_embeddings,
        eval_features=test_embeddings,
        eval_labels=test_labels,
        contamination_natural=natural_contamination,
    )

    # --- Print a results table (Tables 3/5-style) to the console ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Macro F1-Score %)")
    print("=" * 70)
    for scenario_key, scenario_label in [("0pct", "0% contamination"), ("natural", "Natural contamination")]:
        print(f"\n-- {scenario_label} --")
        print(f"{'Algorithm':<10} {'Raw Features':>15} {'Embeddings':>15}")
        for algo in ALGORITHMS:
            raw_f1 = raw_results[scenario_key][algo] * 100
            embed_f1 = embed_results[scenario_key][algo] * 100
            print(f"{ALGORITHM_DISPLAY_NAMES[algo]:<10} {raw_f1:>14.2f}% {embed_f1:>14.2f}%")

    # --- Plot both scenarios ---
    plot_comparison(raw_results, embed_results, "0pct", "(0% contamination)",
                     os.path.join(plots_dir, "comparison_0pct_contamination.png"))
    plot_comparison(raw_results, embed_results, "natural",
                     f"(natural contamination, ~{natural_contamination * 100:.1f}%)",
                     os.path.join(plots_dir, "comparison_natural_contamination.png"))

    print(f"\n🎉 All plots saved to '{plots_dir}/' and displayed above.")


if __name__ == "__main__":
    main()