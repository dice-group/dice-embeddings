import pytest
from dicee.static_funcs import from_pretrained_model_write_embeddings_into_csv
from dicee.executer import Execute
from dicee.config import Namespace

class TestSavingEmbeddings:
    def test_saving_embeddings(self):
        # (1) Train a KGE model
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.path_single_kg = "KGs/Family/family-benchmark_rich_background.owl"
        args.backend = "rdflib"
        args.num_epochs = 0
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 512
        result = Execute(args).start()
        from_pretrained_model_write_embeddings_into_csv(result["path_experiment_folder"])

    def test_model_parallel_saving_embeddings(self):
        # (1) Train a KGE model
        if torch.cuda.is_available():
            from dicee.config import Namespace
            from dicee.executer import Execute
            import os
            args = Namespace()
            args.model = 'Keci'
            args.p = 0
            args.q = 1
            args.optim = 'Adam'
            args.scoring_technique = "KvsAll"
            args.path_single_kg = "KGs/Family/family-benchmark_rich_background.owl"
            args.backend = "rdflib"
            args.trainer = "TP"
            args.num_epochs = 1
            args.batch_size = 1024
            args.lr = 0.1
            args.embedding_dim = 32
            args.save_embeddings_as_csv = True
            result = Execute(args).start()
            assert os.path.exists(result["path_experiment_folder"] + "/Keci_entity_embeddings.csv")
            assert os.path.exists(result["path_experiment_folder"] + "/Keci_relation_embeddings.csv")
