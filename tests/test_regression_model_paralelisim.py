import pytest
from dicee.static_funcs import write_csv_from_model_parallel
from dicee.executer import Execute
from dicee.config import Namespace
import os
import torch
class TestRegressionTensorParallel:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        if torch.cuda.is_available():
            os.system(f'rm -rf Keci_UMLS')
            args = Namespace()
            args.model = 'Keci'
            args.trainer = "TP"
            args.scoring_technique = "KvsAll"  # 1vsAll, or AllvsAll, or NegSample
            args.dataset_dir = "KGs/UMLS"
            args.path_to_store_single_run = "Keci_UMLS"
            args.save_embeddings_as_csv=True
            args.optim="Adopt"
            args.num_epochs = 100
            args.embedding_dim = 32
            args.batch_size = 32
            args.lr=0.1
            reports = Execute(args).start()
            assert reports["Train"]["MRR"] >= 0.60
            assert reports["Test"]["MRR"] >= 0.58
            assert os.path.exists("Keci_UMLS/Keci_entity_embeddings.csv")
            assert os.path.exists("Keci_UMLS/Keci_relation_embeddings.csv")
            os.system(f'rm -rf Keci_UMLS')
