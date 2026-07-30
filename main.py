import os
import torch
from src.data_pipeline.preprocessor import AnomalEPreprocessor
from src.data_pipeline.graph_builder import AnomalEGraphBuilder
from src.models.dgi_module import AnomalEDGI
from src.models.anomaly_detectors import AnomalEDetector
from src.engine.trainer import AnomalETrainer

def main():
    dataset_path = "data/raw/NF-CSE-CIC-IDS2018-v2.csv"    
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        return

    # --- Device setup ---------------------------------------------------
    # Picks CUDA automatically if a GPU is available (e.g. Colab GPU
    # runtime), otherwise falls back to CPU so the script still runs
    # locally without changes.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=== Starting Phase 1: Data Pipeline ===")
    preprocessor = AnomalEPreprocessor()
    # Full run: sanity_check=False processes the whole dataset via memory-safe
    # chunked reading (see preprocessor.py's load_and_clean_data), keeping
    # ~5% of rows after stratified downsampling (fraction=0.05). This was
    # lowered from 0.1 after a CUDA OutOfMemoryError during training: with
    # fraction=0.1 the graph had ~5.29M edges, and the full-batch E-GraphSAGE
    # forward pass (which processes the entire graph at once, no
    # mini-batching) tried to allocate more GPU memory than was available on
    # a 14.56GB GPU. fraction=0.05 roughly halves the edge count and should
    # fit comfortably. Raise this back up if you have a larger GPU (e.g.
    # A100 40GB) available, or lower it further (e.g. 0.02-0.03) if you still
    # hit CUDA OOM.
    train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=False, fraction=0.05)

    # NOTE: we no longer extract a separate test_labels array from test_df here.
    # trainer.evaluate() now reads ground-truth labels directly from
    # test_g.edata['Label'] instead -- this is the array that is guaranteed to
    # be in the same edge order as the embeddings the encoder produces (see
    # trainer.py's evaluate() docstring / section 4a of its README for why a
    # dataframe-order array like test_df['Label'].values can NOT be trusted
    # to line up with the graph's edges after graph_builder's
    # MultiGraph -> to_directed() conversion).

    # --- Checkpoint setup -------------------------------------------------
    # trainer.py's train() now saves a checkpoint every CHECKPOINT_EVERY
    # epochs (and always on the final epoch), so a dropped Colab session
    # loses at most CHECKPOINT_EVERY - 1 epochs of progress instead of the
    # entire run. If a checkpoint already exists when this script starts, it
    # is loaded and training resumes from the epoch it left off at, rather
    # than restarting from epoch 1 or skipping training entirely.
    #
    # Defaults to a local "checkpoints/" folder inside the repo. Override
    # with the ANOMAL_E_CHECKPOINT_DIR environment variable to point this at
    # a persistent location instead (e.g. a mounted Google Drive path), so
    # the checkpoint survives even if the Colab runtime itself is torn down:
    #     ANOMAL_E_CHECKPOINT_DIR="/content/drive/MyDrive/anomal_e_checkpoints" python main.py
    checkpoint_dir = os.environ.get("ANOMAL_E_CHECKPOINT_DIR", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "anomal_e_dgi.pt")
    checkpoint_every = 10  # save a mid-training checkpoint every N epochs
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[INFO] Checkpoint path: {checkpoint_path} (saved every {checkpoint_every} epochs)")

    graph_builder = AnomalEGraphBuilder()
    train_g, test_g = graph_builder.generate_graphs(train_df, test_df)

    # --- Move graphs + features to the training device -------------------
    # DGL graphs carry their own node/edge feature storage, so `g.to(device)`
    # moves both the graph structure and everything in ndata/edata at once.
    # We still index ndata['h']/edata['h'] afterwards, on the already-moved
    # graphs, so those feature tensors end up on `device` too.
    train_g = train_g.to(device)
    test_g = test_g.to(device)

    print("\n=== Starting Final Phase: Execution Engine ===")
    ndim_in = train_g.ndata['h'].shape[2]
    edims = train_g.edata['h'].shape[2]
    hidden_dim = 128
    edge_hidden_dim = 256
    
    # 1. GNN Model
    print("Initializing DGI Model...")
    dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim).to(device)
    
    # 2. Optimizer
    # Built AFTER moving the model to `device`, so it references the
    # parameters that actually live on the GPU (building it before .to(device)
    # would optimize stale CPU copies).
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)
    
    # 3. Anomaly Detector (HBOS with 10% contamination)
    # Runs on CPU regardless of `device` -- PyOD/scikit-learn detectors don't
    # use CUDA. AnomalEDetector._prepare_data() already handles moving the
    # encoder's GPU embeddings back to CPU/NumPy before fitting/scoring.
    print("Initializing HBOS Detector...")
    detector = AnomalEDetector(model_name='hbos', contamination=0.10)
    
    # 4. Trainer
    trainer = AnomalETrainer(dgi_model, detector, optimizer, epochs=50)
    
    # 5. Execute Pipeline
    # If a checkpoint already exists (from a previous run that was cut off
    # partway through training, finished training entirely, or is being
    # re-used to skip straight to evaluation), load it first. load_checkpoint
    # restores model + optimizer state and returns how many epochs were
    # already completed, which is passed straight into train() as
    # start_epoch so training resumes exactly where it left off instead of
    # restarting from epoch 1 or silently skipping remaining epochs.
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] Found existing checkpoint at '{checkpoint_path}' -- resuming from it.")
        start_epoch = trainer.load_checkpoint(checkpoint_path, map_location=device)

    trainer.train(
        train_g,
        train_g.ndata['h'],
        train_g.edata['h'],
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        start_epoch=start_epoch,
    )

    # No labels array passed here -- evaluate() reads test_g.edata['Label']
    # internally, which is guaranteed aligned with the edge embeddings it
    # just computed from test_g.edata['h'].
    trainer.evaluate(test_g, test_g.ndata['h'], test_g.edata['h'])
    
    print("\n🎉 === ANOMAL-E PIPELINE SUCCESSFULLY COMPLETED === 🎉")

if __name__ == "__main__":
    main()