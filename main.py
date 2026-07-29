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

    print("=== Starting Phase 1: Data Pipeline ===")
    preprocessor = AnomalEPreprocessor()
    # Using sanity_check=True for fast execution
    train_df, test_df = preprocessor.process_pipeline(dataset_path, sanity_check=True)

    # NOTE: we no longer extract a separate test_labels array from test_df here.
    # trainer.evaluate() now reads ground-truth labels directly from
    # test_g.edata['Label'] instead -- this is the array that is guaranteed to
    # be in the same edge order as the embeddings the encoder produces (see
    # trainer.py's evaluate() docstring / section 4a of its README for why a
    # dataframe-order array like test_df['Label'].values can NOT be trusted
    # to line up with the graph's edges after graph_builder's
    # MultiGraph -> to_directed() conversion).

    graph_builder = AnomalEGraphBuilder()
    train_g, test_g = graph_builder.generate_graphs(train_df, test_df)
    
    print("\n=== Starting Final Phase: Execution Engine ===")
    ndim_in = train_g.ndata['h'].shape[2]
    edims = train_g.edata['h'].shape[2]
    hidden_dim = 128
    edge_hidden_dim = 256
    
    # 1. GNN Model
    print("Initializing DGI Model...")
    dgi_model = AnomalEDGI(ndim_in, edims, hidden_dim, edge_hidden_dim)
    
    # 2. Optimizer
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)
    
    # 3. Anomaly Detector (HBOS with 10% contamination)
    print("Initializing HBOS Detector...")
    detector = AnomalEDetector(model_name='hbos', contamination=0.10)
    
    # 4. Trainer
    trainer = AnomalETrainer(dgi_model, detector, optimizer, epochs=50)
    
    # 5. Execute Pipeline
    trainer.train(train_g, train_g.ndata['h'], train_g.edata['h'])
    # No labels array passed here -- evaluate() reads test_g.edata['Label']
    # internally, which is guaranteed aligned with the edge embeddings it
    # just computed from test_g.edata['h'].
    trainer.evaluate(test_g, test_g.ndata['h'], test_g.edata['h'])
    
    print("\n🎉 === ANOMAL-E PIPELINE SUCCESSFULLY COMPLETED === 🎉")

if __name__ == "__main__":
    main()