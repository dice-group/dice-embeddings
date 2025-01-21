import os
import json
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

from dicee.models import Keci
from dicee.knowledge_graph import KG
from dicee.static_funcs import read_or_load_kg
from dicee.config import Namespace
from dicee.dataset_classes import KvsAll
from dicee.evaluator import Evaluator


# Configuration setup
args = Namespace()
args.scoring_technique = "KvsAll"
args.dataset_dir = "KGs/FB15k-237"
# args.dataset_dir = "KGs/FamilyT"
args.eval_model = "train_test_eval"
args.apply_reciprical_or_noise = True
args.full_storage_path = "Experiments_lit/train_data"
args.neg_ratio = 0
args.label_smoothing_rate = 0.0
args.batch_size = 1024
args.normalization = None
args.num_epochs = 250
args.embedding_dim = 128
args.lr = 0.05
lit_dataset_dir = "KGs/FB15K-237/train.txt"
lit_test_dir = "KGs/FB15K-237/test.txt"
optimize_with_literal = True


# lit_dataset_dir = "KGs/FamilyL/literals.txt"
# lit_test_dir = "KGs/FamilyL/literal_test.txt"

os.makedirs(args.full_storage_path, exist_ok=True)
# Device setup (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LiteralData:
    def __init__(self, file_path: str, ent_idx):
        self.file_path = file_path
        self.ent_idx = {value: idx for idx, value in ent_idx["entity"].items()}
        df = pd.read_csv(
            self.file_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        top_10_items = df["relation"].value_counts().nlargest(10).index

        # Step 2: Filter the DataFrame to include only rows with these top 10 items
        df = df[df["relation"].isin(top_10_items)]
        self.train_rels = top_10_items
        df = df[df["head"].isin(self.ent_idx)]
        df["tail"] = df["tail"].astype(float)

        self.unique_relations = df["relation"].unique()
        self.num_data_properties = len(self.unique_relations)
        self.data_property_to_idx = {
            relation: idx for idx, relation in enumerate(self.unique_relations)
        }

        df["head_idx"] = df["head"].map(self.ent_idx)
        df["rel_idx"] = df["relation"].map(self.data_property_to_idx)

        # Calculate normalization parameters for each relation group
        self.normalization_params = {}
        for relation in self.unique_relations:
            group_data = df.loc[df["relation"] == relation, "tail"]
            mean = group_data.mean()
            std = group_data.std()
            self.normalization_params[relation] = {"mean": mean, "std": std}

        # Normalize the tail values using the stored parameters
        df["normalized_tail"] = (
            df["tail"] - df.groupby("relation")["tail"].transform("mean")
        ) / df.groupby("relation")["tail"].transform("std")
        self.train_df = df


class LiteralEmbeddings(torch.nn.Module):
    def __init__(
        self,
        num_of_data_properties: int = None,
        dropout: float = 0.3,
        embedding_dims: int = None,
    ):
        super().__init__()
        self.embeddings_dim = embedding_dims
        self.data_property_embeddings = torch.nn.Embedding(
            num_embeddings=num_of_data_properties, embedding_dim=self.embeddings_dim
        )
        self.fc1 = torch.nn.Linear(
            in_features=self.embeddings_dim * 2,
            out_features=self.embeddings_dim * 2,
            bias=True,
        )
        self.fc2 = torch.nn.Linear(
            in_features=self.embeddings_dim * 2, out_features=1, bias=True
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, relation_idx, train_ent_embeddings=True):
        head_entity_embeddings = x
        if not train_ent_embeddings:
            head_entity_embeddings = x.detach()
        relation_embeddings = self.data_property_embeddings(relation_idx)
        tuple_embeddings = torch.cat(
            (head_entity_embeddings, relation_embeddings), dim=1
        )
        out1 = F.relu(self.fc1(tuple_embeddings))
        out1 = self.dropout(out1)
        out2 = self.fc2(out1 + tuple_embeddings)
        return out2.flatten()


# Model initialization
entity_dataset = read_or_load_kg(args, KG)
args.num_entities = entity_dataset.num_entities
args.num_relations = entity_dataset.num_relations

train_dataset = KvsAll(
    train_set_idx=entity_dataset.train_set,
    entity_idxs=entity_dataset.entity_to_idx,
    relation_idxs=entity_dataset.relation_to_idx,
    form="EntityPrediction",
)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

model = Keci(
    args={
        "num_entities": entity_dataset.num_entities,
        "num_relations": entity_dataset.num_relations,
        "embedding_dim": args.embedding_dim,
        "p": 0,
        "q": 1,
        "optim": "Adam",
    }
).to(device)
model_evaluator = Evaluator(args=args)

# Literal model initialization
if optimize_with_literal:

    literal_dataset = LiteralData(
        file_path=lit_dataset_dir,
        ent_idx=entity_dataset.entity_to_idx,
    )
    Literal_model = LiteralEmbeddings(
        num_of_data_properties=literal_dataset.num_data_properties,
        embedding_dims=args.embedding_dim,
    ).to(device)

    lit_y = torch.FloatTensor(literal_dataset.train_df["normalized_tail"].tolist()).to(
        device
    )
    lit_entities = torch.LongTensor(literal_dataset.train_df["head_idx"].values).to(
        device
    )
    lit_properties = torch.LongTensor(literal_dataset.train_df["rel_idx"].values).to(
        device
    )

    lr_keci = args.lr  # Learning rate for Keci model
    lr_literal = 0.001  # Learning rate for LiteralEmbeddings model
    optimizer = optim.Adam(
        [
            {"params": model.parameters(), "lr": lr_keci},  # Keci model parameters
            {
                "params": Literal_model.parameters(),
                "lr": lr_literal,
            },  # LiteralEmbeddings parameters
        ]
    )
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Initialize dictionary
loss_log = {
    "ent_loss": [],
    "lit_loss": [],
}
evaluator = Evaluator(args=args)
# assert isinstance(train_dataset, torch.utils.data.Dataset)
# tensor_X, tensor_y = zip(*(train_dataset[idx] for idx in range(len(train_dataset))))
# train_X = torch.stack(tensor_X, dim=0).to(device)
# train_y = torch.stack(tensor_y, dim=0).to(device)
# Training loop
for epoch in (tqdm_bar := tqdm(range(args.num_epochs))):
    ent_loss = 0
    model.train()
    # ent_loss = model.training_step(batch=(train_X, train_y))
    for batch in train_dataloader:
        train_X, train_y = batch
        train_X, train_y = train_X.to(device), train_y.to(device)

        ent_loss_batch = model.training_step(batch=(train_X, train_y))
        ent_loss += ent_loss_batch

    avg_epoch_loss = ent_loss / len(train_dataloader)
    loss_log["ent_loss"].append(avg_epoch_loss.item())

    if optimize_with_literal:
        ent_ebds = model.entity_embeddings(lit_entities)
        yhat = Literal_model.forward(ent_ebds, lit_properties)
        lit_loss = F.l1_loss(yhat, lit_y)
        loss_log["lit_loss"].append(lit_loss.item())
        total_loss = ent_loss + lit_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tqdm_bar.set_postfix_str(
            f" Avg_loss_ent={avg_epoch_loss:.5f} , loss_lit={lit_loss:.5f} "
        )

    else:
        ent_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tqdm_bar.set_postfix_str(f" loss_epoch={ent_loss:.5f}")

    # Evaluating the model
    if (epoch > 50 and epoch % 50 == 0) or epoch == args.num_epochs:
        print("Model Evaluations at epoch {}".format(epoch))

        model.to("cpu")
        model_evaluator.eval(
            dataset=entity_dataset,
            trained_model=model,
            form_of_labelling="EntityPrediction",
        )
        model.to(device)


print("Model Evaluations after final training epoch")
# evaluator = Evaluator(args=args)
model.to("cpu")
evaluator.eval(
    dataset=entity_dataset, trained_model=model, form_of_labelling="EntityPrediction"
)
model.to(device)

# Testing loop (denormalizing results and calculating metrics)
test_df = pd.read_csv(
    lit_dataset_dir, sep="\t", header=None, names=["head", "relation", "tail"]
)
test_df = test_df[test_df["relation"].isin(literal_dataset.train_rels)]
test_df["head_idx"] = test_df["head"].map(literal_dataset.ent_idx)
test_df["rel_idx"] = test_df["relation"].map(literal_dataset.data_property_to_idx)

lit_entities_test = torch.LongTensor(test_df["head_idx"].values).to(device)
lit_properties_test = torch.LongTensor(test_df["rel_idx"].values).to(device)

model.eval()
Literal_model.eval()

with torch.no_grad():
    ent_ebds_test = model.entity_embeddings(lit_entities_test)
    pred = Literal_model.forward(ent_ebds_test, lit_properties_test)

test_df["preds"] = pred.cpu().numpy()


# Denormalize predictions
def denormalize(row):
    type_stats = literal_dataset.normalization_params[row["relation"]]
    return (row["preds"] * type_stats["std"]) + type_stats["mean"]


test_df["denormalized_preds"] = test_df.apply(denormalize, axis=1)


# Compute MAE and RMSE for each relation
def compute_errors(group):
    actuals = group["tail"]
    predictions = group["denormalized_preds"]
    mae = mean_absolute_error(actuals, predictions)
    rmse = root_mean_squared_error(actuals, predictions)
    return pd.Series({"MAE": mae, "RMSE": rmse})


error_metrics = test_df.groupby("relation").apply(compute_errors).reset_index()
pd.options.display.float_format = "{:.6f}".format  # 6 decimal places
print("Literal Prediction Results on Test Set")
print(error_metrics)

############# storing the experiment results


exp_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
exp_path_name = f"Experiments_lit/{exp_date_time}"
os.makedirs(exp_path_name, exist_ok=True)
print(f"The experiment results are stored at {exp_path_name}")

lit_results_file_path = os.path.join(exp_path_name, "lit_results.json")
with open(lit_results_file_path, "w") as f:
    json.dump(error_metrics.to_dict(orient="records"), f, indent=4)

df_loss_log = pd.DataFrame(loss_log)
loss_log_file_path = os.path.join(exp_path_name, "loss_log.tsv")
df_loss_log.to_csv(loss_log_file_path, sep="\t", index=False)

lp_results_file_path = os.path.join(exp_path_name, "lp_results.json")
with open(lp_results_file_path, "w") as f:
    json.dump(evaluator.report, f, indent=4)

exp_details = {
    "model_name": model.name,
    "entity_lr": lr_keci,
    "lit_lr": lr_literal,
    "embedding_dims": args.embedding_dim,
    "dataset_dir": args.dataset_dir,
    "batch_size": args.batch_size,
    "epochs": args.num_epochs,
}
exp_details_file_path = os.path.join(exp_path_name, "exp_details.json")
with open(exp_details_file_path, "w") as f:
    json.dump(exp_details, f, indent=4)
