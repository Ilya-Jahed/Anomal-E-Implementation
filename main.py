import os
import torch
from src.data_pipeline.preprocessor import AnomalEPreprocessor
from src.data_pipeline.graph_builder import AnomalEGraphBuilder
from src.models.dgi_module import AnomalEDGI
from src.models.e_graphsage import AnomalESAGEEncoder
# Import the new Anomaly Detector wrapper
from src.models.anomaly_detectors import AnomalEDetector

def main():
    dataset_path = "data/raw/NF-CSE-CIC-IDS2018-v2.csv"    
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        return

    print("=== Starting Phase 1: Data Pipeline ===")
    preprocessor = AnomalEPreprocessor()
    train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=True)
    
    graph_builder = AnomalEGraphBuilder()
    train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
    print("=== Phase 1 Execution Successful ===\n")
    
    # ==========================================
    # Phase 2: Encoder & DGI Sanity Check
    # ==========================================
    print("=== Starting Phase 2: DGI & Encoder Sanity Check ===")
    ndim_in = train_g.ndata['h'].shape[2]
    edims = train_g.edata['h'].shape[2]
    hidden_dim = 128
    edge_hidden_dim = 256
    
    # 1. Test DGI Loss
    dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim)
    loss = dgi_model(train_g, train_g.ndata['h'], train_g.edata['h'])
    print(f"Calculated DGI Loss: {loss.item():.4f}")
    
    # 2. Get real embeddings from Encoder for Phase 3
    print("Extracting edge embeddings from Encoder...")
    encoder = AnomalESAGEEncoder(ndim_in, edims, hidden_dim, edge_hidden_dim)
    _, out_e = encoder(train_g, train_g.ndata['h'], train_g.edata['h'], corrupt=False)
    print("=== Phase 2 Execution Successful ===\n")

    # ==========================================
    # NEW: Phase 3: Anomaly Detector Sanity Check
    # ==========================================
    print("=== Starting Phase 3: Anomaly Detector Sanity Check ===")
    print("Initializing HBOS Anomaly Detector...")
    
    # Create the detector (e.g., HBOS) with a 10% expected anomaly rate
    detector = AnomalEDetector(model_name='hbos', contamination=0.10)
    
    # Train the detector on the extracted edge embeddings (out_e)
    print(f"Fitting model on embedding tensor of shape {out_e.shape}...")
    detector.fit(out_e)
    
    # Predict normal (0) vs anomalous (1) labels
    predictions = detector.predict(out_e)
    
    # Calculate some basic stats to ensure it worked
    num_anomalies = sum(predictions)
    total_samples = len(predictions)
    print(f"Prediction complete! Found {num_anomalies} anomalies out of {total_samples} edges.")
    print("=== Phase 3 Execution Successful! ===")

if __name__ == "__main__":
    main()