from dicee.executer import Execute, get_default_arguments
from dicee.config import Args
import pytest


class TestCallback:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_conex_torch_cpu_trainer(self):
        args = Args()  # get_default_arguments([])
        args.model = 'AConEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_aconex_pl_trainer(self):
        args = Args()  # get_default_arguments([])
        args.model = 'AConEx'
        args.num_epochs = 1
        args.scoring_technique = 'KvsAll'
        args.path_dataset_folder = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.trainer = 'PL'
        Execute(args).start()

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
    def test_torchDDP_trainer(self, model_name):
      import torch 
      args = get_default_arguments([])
      # args.model = 'DistMult'
      # args.model = 'Pykeen_DistMult'
      # args.model = 'Pykeen_KG2E'
      # args.model = 'AConEx'
      args.model = model_name
      args.scoring_technique = "NegSample"
      args.path_dataset_folder = "KGs/KINSHIP"
      args.num_epochs = 20
      args.batch_size = 20
      args.lr = 0.01
      args.embedding_dim = 64
      args.trainer = 'torchDDP'
      # args.trainer = "torchCPUTrainer"
      args.num_core = 1  # need to be bigger than 0
      args.eval_model = "train_val_test"
      args.normalization = None
      args.devices = "auto"
      args.accelerator = "auto"
      args.optim = "Adam"
      args.pykeen_model_kwargs = dict(
          embedding_dim=args.embedding_dim, loss="BCEWithLogitsLoss"
      )
      args.use_ddp_batch_finder = True
      torch.cuda.empty_cache()
      Execute(args).start()