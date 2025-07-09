import os
import zipfile
import urllib.request
import shutil

from dicee.executer import Execute
from dicee.config import Namespace
from dicee.knowledge_graph_embeddings import KGE

from dicee.eval_static_funcs import evaluate_literal_prediction
"""
Literal Prediction using Interactive Knowledge Graph Embedding (KGE) model.

Setup Steps:
1. Install required package (`dicee`).
2. Download & extract the 'Literal_KGs' dataset if not present.
"""

# ---- INSTALL DICE framework if not installed ----
# pip install dicee

# ---- DOWNLOAD & EXTRACT DATASET IF NOT PRESENT ----

literal_kg_dir = "Literal_KGs/Family"
# Check if the dataset directory exists, if not create it
if not os.path.exists(literal_kg_dir):
    url = "https://files.dice-research.org/datasets/dice-embeddings/Literal_KGs.zip"
    output_path = "Literal_KGs.zip"
    # Download the zip file
    urllib.request.urlretrieve(url, output_path)
    # Unzip the file into the specified directory
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(literal_kg_dir)
    # Remove the zip file after extraction
    os.remove(output_path)

# -------------------------------
#     INTERACTIVE KGE TRAINING
# -------------------------------

# --- Training KGE Model ---
args = Namespace()
args.model = 'Keci'
args.optim = 'Adam'
args.scoring_technique = "KvsAll"
args.dataset_dir = "Literal_KGs/Family"
args.backend = "pandas"
args.num_epochs = 256
args.batch_size = 512
args.lr = 0.1
args.embedding_dim = 32
args.trainer = "TorchCPUTrainer"

print("Training Knowledge Graph Embedding model...")
result = Execute(args).start()

# --- Training a Literal Embedding Model ---
"""
The Literal Embedding model is a simple, non-linear, 2-layer NN used to predict
numerical attributes leveraging pretrained KGE.
Configurable parameters include:
- train_file_path: path to literal train triples
- num_epochs: training epochs
- lit_lr: learning rate
- lit_normalization_type: normalization ('z-norm', 'min-max', or None)
- batch_size: training batch size
- loader_backend: 'pandas' or 'rdflib'
- freeze_entity_embeddings: freeze entity embeddings during training
- gate_residual: use gate residual connection
- device: 'cpu' or 'cuda'
"""

# Load trained KGE model
pre_trained_kge = KGE(path=result['path_experiment_folder'])

# Path for literal training and eval file
lit_train_file_path = "Literal_KGs/Family/literals/train.txt"
lit_eval_file_path = "Literal_KGs/Family/literals/test.txt"

assert os.path.exists(lit_train_file_path), "Literal train file not found!"
assert os.path.exists(lit_eval_file_path), "Literal eval file not found!"

print("Training literal embedding model...")
pre_trained_kge.train_literals(
    train_file_path=lit_train_file_path,
    lit_lr=0.001,
    lit_normalization_type="z-norm",
    loader_backend="pandas",
    num_epochs=200,
    batch_size=50,
    device="cpu"
)

# ---------------------------
#     LITERAL PREDICTION
# ---------------------------



# Load and validate the test data
test_df_unfiltered = pre_trained_kge.literal_dataset.load_and_validate_literal_data(
    file_path=lit_eval_file_path, loader_backend="pandas"
)

# Filter test data to use only valid entity/attribute pairs
test_df = test_df_unfiltered[
    test_df_unfiltered["head"].isin(pre_trained_kge.entity_to_idx.keys()) &
    test_df_unfiltered["attribute"].isin(pre_trained_kge.data_property_to_idx.keys())
]

entities = test_df["head"].tolist()
attributes = test_df["attribute"].tolist()

assert entities, "No valid entities in test set — check entity_to_idx."
assert attributes, "No valid attributes in test set — check data_property_to_idx."

# --- Single Prediction ---
print("\nSingle prediction example:")
single_pred = pre_trained_kge.predict_literals(
    entity=entities[0], attribute=attributes[0]
)[0]
print(f"Predicted {attributes[0]} for entity {entities[0]} : {single_pred:.2f}")

# --- Batch Prediction ---
print("\nBatch literal predictions (first 10):")
test_df["predictions"] = pre_trained_kge.predict_literals(
    entity=entities, attribute=attributes
)
print(test_df[["head", "attribute", "value", "predictions"]].head(10))

# ---------------------------
#     LITERAL EVALUATION
# ---------------------------

print("\nEvaluating literal prediction (MAE, RMSE):")
lit_prediction_errors = evaluate_literal_prediction(
    kge_model=pre_trained_kge,
    eval_file_path=lit_eval_file_path,
    eval_literals=True,
    store_lit_preds=False,
    loader_backend="pandas",
    return_attr_error_metrics=True
)

# ---------------------------
#     OPTIONAL CLEANUP
# ---------------------------
"""
To remove the dataset and experiments after running, set `cleanup` to `True`.
This will delete the 'Literal_KGs' directory and the experiment folder created during training.
"""
cleanup = True
if cleanup:
    print("\nCleaning up dataset and experiment artifacts...")
    shutil.rmtree("Literal_KGs", ignore_errors=True)
    shutil.rmtree(result['path_experiment_folder'], ignore_errors=True)