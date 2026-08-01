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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Using device: {device} ===")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("=== Starting Phase 1: Data Pipeline ===")
    preprocessor = AnomalEPreprocessor()
    # fraction=0.1 matches the reference notebook -- see docs/MAIN_EXPLANATION.md
    train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=False, fraction=0.1)

    checkpoint_dir = os.environ.get("ANOMAL_E_CHECKPOINT_DIR", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "anomal_e_dgi.pt")
    checkpoint_every = 100
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[INFO] Checkpoint path: {checkpoint_path} (saved every {checkpoint_every} epochs)")

    graph_builder = AnomalEGraphBuilder()
    train_g, test_g = graph_builder.generate_graphs(train_df, test_df)

    train_g = train_g.to(device)
    test_g = test_g.to(device)

    print("\n=== Starting Final Phase: Execution Engine ===")
    ndim_in = train_g.ndata['h'].shape[2]
    edims = train_g.edata['h'].shape[2]
    hidden_dim = 128
    edge_hidden_dim = 256
    
    print("Initializing DGI Model...")
    dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim).to(device)
    
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)
    
    print("Initializing HBOS Detector...")
    detector = AnomalEDetector(model_name='hbos', contamination=0.10)
    
    # epochs=4000 matches the reference notebook -- see docs/MAIN_EXPLANATION.md
    trainer = AnomalETrainer(dgi_model, detector, optimizer, epochs=4000)
    
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

    trainer.evaluate(test_g, test_g.ndata['h'], test_g.edata['h'])
    
    print("\n🎉 === ANOMAL-E PIPELINE SUCCESSFULLY COMPLETED === 🎉")

if __name__ == "__main__":
    main()