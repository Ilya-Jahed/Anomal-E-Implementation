import os
from src.data_pipeline.preprocessor import AnomalEPreprocessor
from src.data_pipeline.graph_builder import AnomalEGraphBuilder
# Import the newly implemented E-GraphSAGE encoder
from src.models.e_graphsage import AnomalESAGEEncoder

def main():
    # Define the dataset path.
    # We explicitly use the original .csv file here instead of .parquet.
    dataset_path = "data/raw/NF-CSE-CIC-IDS2018-v2.csv"    
    
    # Check if the file exists before running
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        print("Please place the original 'NF-CSE-CIC-IDS2018-v2.csv' inside the 'data/raw' folder.")
        return

    print("=== Starting Phase 1: Data Pipeline ===")
    
    # 1. Initialize Preprocessor and run pipeline in Sanity Check mode
    preprocessor = AnomalEPreprocessor()
    train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=True)
    
    # 2. Initialize Graph Builder and generate DGL graphs.
    graph_builder = AnomalEGraphBuilder()
    train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
    
    print("\n=== Phase 1 Execution Successful ===")
    
    # --- INSPECTING THE GENERATED GRAPHS ---
    print("\n[INSPECTION] Tensor Dimensions (Training Graph):")
    # Expected shape: (Number of nodes, 1, Number of features)
    print(f"Node Features (ndata['h']) shape: {train_g.ndata['h'].shape}")
    
    # Expected shape: (Number of edges, 1, Number of features)
    print(f"Edge Features (edata['h']) shape: {train_g.edata['h'].shape}")
    print("-" * 50)

    # ==========================================
    # NEW: Phase 2 (Encoder) Sanity Check
    # ==========================================
    print("\n=== Starting Phase 2: Encoder Sanity Check ===")
    
    # Automatically extract dimensions from the generated DGL graph
    ndim_in = train_g.ndata['h'].shape[2]
    edims = train_g.edata['h'].shape[2]
    hidden_dim = 128
    edge_hidden_dim = 256
    
    print("Initializing AnomalESAGEEncoder...")
    encoder = AnomalESAGEEncoder(
        ndim_in=ndim_in, 
        edims=edims, 
        hidden_dim=hidden_dim, 
        edge_hidden_dim=edge_hidden_dim
    )
    
    print("Running forward pass (Real Graph - corrupt=False)...")
    out_n, out_e = encoder(train_g, train_g.ndata['h'], train_g.edata['h'], corrupt=False)
    print(f"Output Node Embeddings shape: {out_n.shape} -> (Expected: N, 128)")
    print(f"Output Edge Embeddings shape: {out_e.shape} -> (Expected: E, 256)")
    
    print("\nRunning forward pass (Corrupted Graph - corrupt=True)...")
    fake_n, fake_e = encoder(train_g, train_g.ndata['h'], train_g.edata['h'], corrupt=True)
    print(f"Corrupted Edge Embeddings shape: {fake_e.shape} -> (Expected: E, 256)")
    
    print("\n=== Phase 2 Encoder Check Successful! ===")

if __name__ == "__main__":
    main()