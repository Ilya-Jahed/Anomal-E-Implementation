import os
from src.data_pipeline.preprocessor import AnomalEPreprocessor
from src.data_pipeline.graph_builder import AnomalEGraphBuilder

def main():
    # Define the dataset path.
    # We explicitly use the original .csv file here instead of .parquet.
    # Although .parquet is faster and lighter for multi-million-row datasets,
    # converted Parquet versions (e.g., from Kaggle) frequently drop crucial 
    # IP address columns (IPV4_SRC_ADDR, IPV4_DST_ADDR) which are strictly 
    # required for GNN node construction in this pipeline.
    dataset_path = "data/raw/NF-CSE-CIC-IDS2018-v2.csv"    
    
    # Check if the file exists before running
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        print("Please place the original 'NF-CSE-CIC-IDS2018-v2.csv' inside the 'data/raw' folder.")
        return

    print("=== Starting Phase 1: Data Pipeline ===")
    
    # 1. Initialize Preprocessor and run pipeline in Sanity Check mode
    # (Setting sanity_check=True ensures it only reads 50,000 rows for a
    # quick end-to-end smoke test -- confirms the whole pipeline runs
    # without errors before committing to a full multi-million-row run).
    preprocessor = AnomalEPreprocessor()
    train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=True)
    
    # 2. Initialize Graph Builder and generate DGL graphs.
    # train_df and test_df already contain the normalised edge feature
    # column 'h' produced by apply_normalization -- graph_builder turns
    # each dataframe into an independent DGL graph (see
    # AnomalEGraphBuilder.generate_graphs).
    graph_builder = AnomalEGraphBuilder()
    train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
    
    print("\n=== Phase 1 Execution Successful ===")
    
    # --- INSPECTING THE GENERATED GRAPHS ---
    # Printing a DGL graph object shows a summary: number of nodes, number
    # of edges, and the names/shapes of every registered node/edge feature
    # -- a quick way to confirm the graph was built as expected.
    print("\n[INSPECTION] Training Graph Overview:")
    print(train_g)
    
    print("\n[INSPECTION] Testing Graph Overview:")
    print(test_g)
    
    print("\n[INSPECTION] Tensor Dimensions (Training Graph):")
    # Expected shape: (Number of nodes, 1, Number of features)
    # The middle '1' is the extra dimension added in graph_builder's
    # reshape step (Step 7.4), required by the E-GraphSAGE layers.
    print(f"Node Features (ndata['h']) shape: {train_g.ndata['h'].shape}")
    
    # Expected shape: (Number of edges, 1, Number of features)
    print(f"Edge Features (edata['h']) shape: {train_g.edata['h'].shape}")
    print("-" * 50)

if __name__ == "__main__":
    main()