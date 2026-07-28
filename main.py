import os
from src.data_pipeline.preprocessor import AnomalEPreprocessor
from src.data_pipeline.graph_builder import AnomalEGraphBuilder
# Import the newly implemented AnomalEDGI module
from src.models.dgi_module import AnomalEDGI

def main():
    # Define the dataset path.
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
    print(f"Node Features (ndata['h']) shape: {train_g.ndata['h'].shape}")
    print(f"Edge Features (edata['h']) shape: {train_g.edata['h'].shape}")
    print("-" * 50)

    # ==========================================
    # Phase 2: DGI Module Sanity Check
    # ==========================================
    print("\n=== Starting Phase 2: DGI Module Sanity Check ===")
    
    # Automatically extract dimensions from the generated DGL graph
    ndim_in = train_g.ndata['h'].shape[2]
    edims = train_g.edata['h'].shape[2]
    hidden_dim = 128
    edge_hidden_dim = 256
    
    print("Initializing AnomalEDGI model...")
    dgi_model = AnomalEDGI(
        ndim_in=ndim_in, 
        edims=edims, 
        ndim_out=hidden_dim, 
        edge_out_dim=edge_hidden_dim
    )
    
    print("Running DGI forward pass to compute loss...")
    # Pass the graph and features to the DGI module
    loss = dgi_model(train_g, train_g.ndata['h'], train_g.edata['h'])
    
    print(f"Calculated Loss Value: {loss.item():.4f}")
    print(f"Loss Tensor shape: {loss.shape} -> (Expected: torch.Size([]))")
    
    print("\n=== Phase 2 DGI Sanity Check Successful! ===")

if __name__ == "__main__":
    main()