import os
import tempfile
from dicee.executer import Execute
from dicee.config import Namespace

class TestExperimentReuse:
    def test_experiment_dir_reuse(self):
        # Create a temporary directory for the experiment
        with tempfile.TemporaryDirectory() as tmpdir:
            # --- First run: create new experiment directory ---
            args1 = Namespace()
            args1.model = 'Keci'
            args1.p = 0
            args1.q = 1
            args1.optim = 'Adam'
            args1.scoring_technique = "KvsAll"
            args1.path_single_kg = "KGs/Family/family-benchmark_rich_background.owl"
            args1.backend = "rdflib"
            args1.num_epochs = 0
            args1.batch_size = 1024
            args1.lr = 0.1
            args1.embedding_dim = 32
            args1.path_to_store_single_run = tmpdir
            args1.reuse_existing_run_dir = False
            args1.random_seed = 42
            args1.save_embeddings_as_csv = False
            args1.eval_model = None
            args1.storage_path = tmpdir

            result1 = Execute(args1).start()
            config_path = os.path.join(tmpdir, "configuration.json")
            assert os.path.exists(config_path)
            report_path1 = result1['path_experiment_folder']
            assert report_path1 == tmpdir

            # --- Second run: reuse the same directory ---
            args2 = Namespace()
            args2.model = 'Keci'
            args2.p = 0
            args2.q = 1
            args2.optim = 'Adam'
            args2.scoring_technique = "KvsAll"
            args2.path_single_kg = "KGs/Family/family-benchmark_rich_background.owl"
            args2.backend = "rdflib"
            args2.num_epochs = 0
            args2.batch_size = 1024
            args2.lr = 0.1
            args2.embedding_dim = 32
            args2.path_to_store_single_run = tmpdir
            args2.reuse_existing_run_dir = True
            args2.random_seed = 42
            args2.save_embeddings_as_csv = False
            args2.eval_model = None
            args2.storage_path = tmpdir

            result2 = Execute(args2).start()
            # The configuration file should still exist and be updated
            assert os.path.exists(config_path)
            report_path2 = result2['path_experiment_folder']
            # The report path should be the same as the reused directory
            assert report_path2 == tmpdir