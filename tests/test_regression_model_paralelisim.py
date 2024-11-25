import pytest
from dicee.static_funcs import write_csv_from_model_parallel
from dicee.executer import Execute
from dicee.config import Namespace
import os
import torch
class TestRegressionModelParallel:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        # @TODO:
        """
        if torch.cuda.is_available():
            args = Namespace()
            args.model = 'Keci'
            args.trainer = "MP"
            args.scoring_technique = "KvsAll"  # 1vsAll, or AllvsAll, or NegSample
            args.dataset_dir = "KGs/UMLS"
            args.path_to_store_single_run = "Keci_UMLS"
            args.num_epochs = 100
            args.embedding_dim = 32
            args.batch_size = 1024
            reports = Execute(args).start()
            assert reports["Train"]["MRR"] >= 0.990
            assert reports["Test"]["MRR"] >= 0.810
            write_csv_from_model_parallel(path="Keci_UMLS")
            assert os.path.exists("Keci_UMLS/entity_embeddings.csv")
            assert os.path.exists("Keci_UMLS/relation_embeddings.csv")

            os.system(f'rm -rf Keci_UMLS')

        """
