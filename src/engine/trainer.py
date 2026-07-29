"""
trainer.py — Execution Engine ("Conductor") for the Anomal-E architecture.

Orchestrates the two things that happen after the graph is built:

    1. train()    — self-supervised DGI training of the E-GraphSAGE encoder.
                     Implements Algorithm 2, lines 2-9 of the Anomal-E paper.
                     No attack/benign labels are used anywhere in this step.

    2. evaluate()  — extracts edge embeddings from the trained encoder, fits
                     a classical PyOD-based anomaly detector on them, and
                     scores the result against ground-truth labels. Labels
                     are used ONLY here, purely for evaluation (Tables 3-8
                     of the paper), never to influence training.
"""

import time
from typing import Dict, List, Optional

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class AnomalETrainer:
    """
    Conductor class tying together the DGI model (encoder + discriminator)
    and a classical anomaly detector.

    Attributes:
        dgi_model: an AnomalEDGI instance (self-supervised encoder + discriminator).
        detector: an AnomalEDetector instance (PCA / HBOS / CBLOF / IForest via PyOD).
        optimizer: a torch optimizer already bound to dgi_model.parameters().
        epochs: number of full training iterations over the graph.
        log_every: print a training log line every N epochs (plus epoch 1).
        loss_history: per-epoch loss values from the most recent train() call.
    """

    def __init__(
        self,
        dgi_model: torch.nn.Module,
        detector,
        optimizer: torch.optim.Optimizer,
        epochs: int = 50,
        log_every: int = 5,
    ):
        if epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs}")
        if log_every <= 0:
            raise ValueError(f"log_every must be a positive integer, got {log_every}")

        self.dgi_model = dgi_model
        self.detector = detector
        self.optimizer = optimizer
        self.epochs = epochs
        self.log_every = log_every

        # Populated by train(); kept around for later inspection / plotting.
        self.loss_history: List[float] = []

    def train(self, g, n_features: torch.Tensor, e_features: torch.Tensor) -> List[float]:
        """
        Runs the self-supervised DGI training loop (Algorithm 2, lines 2-9).

        The loss comes entirely from `dgi_model`'s internal real-vs-corrupted
        discrimination signal -- no labels are read or needed here.

        Args:
            g: training DGL graph (e.g. train_g).
            n_features: node feature tensor, shape (N, 1, ndim_in).
            e_features: edge feature tensor, shape (E, 1, edims).

        Returns:
            List of per-epoch loss values (also stored in self.loss_history).
        """
        print(f"\n--- Starting DGI Training for {self.epochs} Epochs ---")
        self.dgi_model.train()
        self.loss_history = []

        training_start = time.time()

        for epoch in range(self.epochs):
            t0 = time.time()

            self.optimizer.zero_grad()
            loss = self.dgi_model(g, n_features, e_features)
            loss.backward()
            self.optimizer.step()

            t1 = time.time()
            loss_value = loss.item()
            self.loss_history.append(loss_value)

            if (epoch + 1) % self.log_every == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:03d}/{self.epochs} | "
                    f"Loss: {loss_value:.4f} | Time: {t1 - t0:.4f}s"
                )

        total_time = time.time() - training_start
        best_epoch = self.loss_history.index(min(self.loss_history)) + 1
        print(
            f"--- DGI Training Completed | Total Time: {total_time:.2f}s | "
            f"Final Loss: {self.loss_history[-1]:.4f} | "
            f"Best Loss: {min(self.loss_history):.4f} (Epoch {best_epoch}) ---"
        )

        return self.loss_history

    def evaluate(
        self,
        g,
        n_features: torch.Tensor,
        e_features: torch.Tensor,
        label_key: str = "Label",
    ) -> Dict[str, Optional[float]]:
        """
        Extracts edge embeddings from the trained encoder, fits/uses the
        anomaly detector, and computes metrics against ground-truth labels.

        Ground-truth labels are used ONLY here -- never inside train(). This
        mirrors the paper's protocol: encoder training and detector.fit()
        are both fully unsupervised; labels only enter to *score* the final
        predictions.

        Label source: `graph_builder.py` builds `g` as a DIRECTED graph via
        NetworkX's `to_directed()`, which reorders edges by node adjacency
        (grouped by source node) rather than preserving the original
        dataframe row order -- and also emits two directed edges per original
        flow. Because of this, labels are ALWAYS read from `g.edata[label_key]`
        rather than accepted as a separately-supplied array: `Label` is
        carried as an edge attribute through the entire
        nx -> to_directed -> dgl conversion, so `g.edata[label_key]` is
        guaranteed to be in the exact same order as the edges the encoder
        just embedded. A label array built any other way (e.g. pulled
        straight from a dataframe) cannot be safely lined up against
        `edge_embeddings` by position -- even if it happens to have the same
        length, its ORDER is not guaranteed to match, and a same-length but
        wrongly-ordered array would silently produce wrong metrics with no
        error raised. Removing that possibility entirely (rather than
        offering a length-matched override) is the whole point of this
        design: there is exactly one way to supply labels here, and it is
        the one that is structurally guaranteed correct.

        Args:
            g: evaluation DGL graph (e.g. test_g).
            n_features: node feature tensor, shape (N, 1, ndim_in).
            e_features: edge feature tensor, shape (E, 1, edims).
            label_key: key into `g.edata` holding ground-truth 0/1 labels.

        Returns:
            Dict with keys "auc", "f1", "precision", "recall". "auc" is
            None if it could not be computed (e.g. only one class present
            in the labels -- this can genuinely happen on small
            sanity-check subsets).
        """
        print("\n--- Starting Evaluation Phase ---")
        self.dgi_model.eval()

        with torch.no_grad():
            print("1. Extracting edge embeddings from the trained Encoder...")
            _, edge_embeddings = self.dgi_model.encoder(
                g, n_features, e_features, corrupt=False
            )

        # Labels come exclusively from the graph itself -- guaranteed to be
        # in the same edge order as edge_embeddings, regardless of how
        # to_directed() reordered or duplicated edges relative to the
        # source dataframe. No externally-supplied array is ever accepted
        # here, which removes the possibility of a same-length-but-
        # wrong-order array silently corrupting the evaluation.
        labels_to_use = g.edata[label_key]
        if isinstance(labels_to_use, torch.Tensor):
            labels_to_use = labels_to_use.detach().cpu().numpy()

        # getattr guards against detector implementations that don't expose
        # model_name as a public attribute; falls back to the class name.
        detector_name = getattr(self.detector, "model_name", type(self.detector).__name__)
        print(f"2. Fitting {str(detector_name).upper()} model on embeddings...")
        self.detector.fit(edge_embeddings)

        print("3. Predicting anomalies and calculating scores...")
        predictions = self.detector.predict(edge_embeddings)
        scores = self.detector.get_anomaly_scores(edge_embeddings)

        # ROC AUC needs both classes present in true_labels; guarded on its
        # own so one degenerate split doesn't take down the other metrics.
        try:
            auc = roc_auc_score(labels_to_use, scores)
        except ValueError as exc:
            print(f"[WARNING] Could not compute ROC AUC: {exc}")
            auc = None

        f1 = f1_score(labels_to_use, predictions)
        precision = precision_score(labels_to_use, predictions)
        recall = recall_score(labels_to_use, predictions)

        print("\n======================================")
        print("📊 FINAL MODEL PERFORMANCE METRICS")
        print("======================================")
        print(f"ROC AUC Score : {auc:.4f}" if auc is not None else "ROC AUC Score : N/A")
        print(f"F1 Score      : {f1:.4f}")
        print(f"Precision     : {precision:.4f}")
        print(f"Recall        : {recall:.4f}")
        print("======================================\n")

        return {"auc": auc, "f1": f1, "precision": precision, "recall": recall}

    def save_checkpoint(self, path: str) -> None:
        """
        Saves model + optimizer state and loss history to `path`.
        Purely additive: lets training be resumed or the trained encoder be
        reused later without rerunning the full DGI loop from scratch.
        """
        torch.save(
            {
                "dgi_model_state": self.dgi_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
            },
            path,
        )
        print(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str, map_location=None) -> None:
        """Restores model + optimizer state and loss history from `path`."""
        checkpoint = torch.load(path, map_location=map_location)
        self.dgi_model.load_state_dict(checkpoint["dgi_model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.loss_history = checkpoint.get("loss_history", [])
        print(f"Checkpoint loaded from: {path}")