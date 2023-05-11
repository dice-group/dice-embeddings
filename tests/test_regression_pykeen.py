from dicee.executer import Execute,get_default_arguments
import sys
import pytest
import wandb

def template(model_name):
    args = get_default_arguments([])
    args.model = model_name
    args.scoring_technique = "NegSample"  # default value of args.eval is 'val_test'
    # args.path_dataset_folder = "KGs/Nations"
    args.path_dataset_folder = "KGs/UMLS"
    # args.path_dataset_folder = "KGs/FB15k-237"
    args.num_epochs = 3
    args.batch_size = 128
    args.lr = 0.01
    args.embedding_dim = 64
    args.input_dropout_rate = 0.0
    args.hidden_dropout_rate = 0.0
    args.feature_map_dropout_rate = 0.0
    args.sample_triples_ratio = None
    args.read_only_few = None
    args.sample_triples_ratio = None
    args.trainer = "PL"
    # args.trainer = "torchCPUTrainer"
    args.neg_ratio = 25
    args.pykeen_model_kwargs = dict(
        embedding_dim=args.embedding_dim, loss="BCEWithLogitsLoss"
    )
    # args.pykeen_model_kwargs = dict(
    #     embedding_dim=args.embedding_dim, loss="BCEWithLogitsLoss",
    #     tokenizers=["RelationTokenizer"],
    #     num_tokens=[12],

    # )
    args.interaction_kwargs = None
    args.use_SLCWALitModule = False
    args.num_core = 1
    args.save_embeddings_as_csv = True
    args.eval_model = 'train_val_test'
    args.optim = 'Adam'
    args.accelerator = 'gpu'
    args.devices = 1
    args.normalization = None
    return args


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "model_name",
    [
        "Pykeen_DistMult",
        "Pykeen_TuckER",
        "Pykeen_UM",
        "Pykeen_TransR",
        "Pykeen_TransH",
        "Pykeen_TransF",
        "Pykeen_TransE",
        "Pykeen_TransD",
        "Pykeen_TorusE",
        "Pykeen_SimplE",
        "Pykeen_SE",
        "Pykeen_RESCAL",
        "Pykeen_RotatE",
        "Pykeen_QuatE",
        "Pykeen_PairRE",
        "Pykeen_ProjE",
        "Pykeen_NTN",
        "Pykeen_NodePiece",
        "Pykeen_MuRE",
        "Pykeen_KG2E",
        "Pykeen_InductiveNodePiece",
        "Pykeen_InductiveNodePieceGNN",
        "Pykeen_HolE",
        "Pykeen_FixedModel",
        "Pykeen_ERMLPE",
        "Pykeen_DistMA",
        "Pykeen_CrossE",
        "Pykeen_CooccurrenceFilteredModel",
        "Pykeen_ConvKB",  # this one is really slow
        "Pykeen_ConvE",
        "Pykeen_ComplExLiteral",
        "Pykeen_ComplEx",
        "Pykeen_CompGCN",
        "Pykeen_CP",
        "Pykeen_BoxE",
        "Pykeen_AutoSF",
        "Pykeen_DistMultLiteral",
    ],
)
def test_model(model_name):
    args = template(model_name)
    # config = {
    #     "epoch":args.num_epochs,"lr":args.lr,"embedding_dim":args.embedding_dim
    # }
    # dataset = args.path_dataset_folder.split('/')[1]
    # wandb.setup(wandb.Settings(program="test_pykeen_model.py", program_relpath="test_pykeen_model.py"))
    # wandb.init(project="dice_demo",config=config,name=f'{args.model}-{dataset}')
    Execute(args).start()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "model_name",
    [
        "Pykeen_TripleREInteraction",
        "Pykeen_TransformerInteraction",
        "Pykeen_MultiLinearTuckerInteraction",
        "Pykeen_LineaREInteraction",
    ],
)
def test_pykeenInteraction(model_name):
    args = template(model_name)
    if model_name == "Pykeen_LineaREInteraction":
        args.interaction_kwargs = {"p": 1}  # the L_p norm, usually 1 or 2
    if model_name == "Pykeen_TransformerInteraction": # relativ slow
        args.embedding_dim = 512 # embedding_dim can only be 512
        args.pykeen_model_kwargs = dict(
            embedding_dim=args.embedding_dim, loss="bcewithlogits",
        )
    Execute(args).start()


class TestPykeen:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_specific_model(self):
        args = get_default_arguments([])
        args.model = "Pykeen_LineaREInteraction"
        args.scoring_technique = "NegSample"
        args.path_dataset_folder = "KGs/Nations"
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 64
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.torch_trainer = "PL"
        args.neg_ratio = 1
        args.use_SLCWALitModule = False
        args.pykeen_model_kwargs = dict(
            embedding_dim=args.embedding_dim,
            loss="bcewithlogits",
            # entity_representations=[None],
            # tokenizers=["AnchorTokenizer", "RelationTokenizer"],
            # num_tokens=[3, 12],
        )

        Execute(args).start()

