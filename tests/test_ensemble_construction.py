from dicee.executer import Execute
from dicee.knowledge_graph_embeddings import KGE
import pytest
import os
from dicee.config import Namespace

class TestEnsembleConstruction:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'QMult'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.input_dropout_rate = 0.0
        args.hidden_dropout_rate = 0.0
        args.feature_map_dropout_rate = 0.0
        args.eval_model = 'train_val_test'
        args.read_only_few = None
        args.sample_triples_ratio = None
        args.num_folds_for_cv = None
        args.save_model_at_every_epoch = 3
        args.normalization = 'LayerNorm'
        args.backend = 'pandas'
        args.trainer = 'torchCPUTrainer'
        result = Execute(args).start()
        assert 0.71 >= result['Train']['H@1'] >= 0.03
        assert 0.71 >= result['Val']['H@1'] >= 0.03
        assert 0.71 >= result['Test']['H@1'] >= 0.03
        assert os.path.isdir(result['path_experiment_folder'])

        # (1) Load single model
        KGE(path=result['path_experiment_folder'])
        KGE(path=result['path_experiment_folder'], construct_ensemble=True)
