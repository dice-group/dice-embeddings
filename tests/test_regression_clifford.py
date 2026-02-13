from dicee.executer import Execute
import pytest
from dicee.config import Namespace

class TestRegressionClifford:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_k_vs_all(self):
        args = Namespace()
        args.model = 'Keci'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.p = 0
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 32
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        keci_result = Execute(args).start()

        args = Namespace()
        args.model = 'DeCaL'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.p = 0
        args.q = 1
        args.r = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 32
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        decal_result = Execute(args).start()

        # assert decal_result["Train"]["MRR"] > keci_result["Train"]["MRR"]
        # assert decal_result["Test"]["MRR"] > keci_result["Test"]["MRR"]

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_neg_sample_vs_fixed_neg_sample(self):
        """Test that NegSample and FixedNegSample produce similar results."""
        args = Namespace()
        args.model = 'Keci'
        args.scoring_technique = 'NegSample'
        args.trainer = 'PL'
        args.optim = 'Adam'
        args.p = 0
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.neg_ratio = 10
        args.eval_model = 'train_val_test'
        neg_sample_result = Execute(args).start()

        args = Namespace()
        args.model = 'Keci'
        args.scoring_technique = 'FixedNegSample'
        args.trainer = 'PL'
        args.optim = 'Adam'
        args.p = 0
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.neg_ratio = 10
        args.eval_model = 'train_val_test'
        fixed_neg_sample_result = Execute(args).start()

        # Both techniques should achieve reasonable performance
        assert neg_sample_result["Test"]["MRR"] > 0.10, \
            f"NegSample MRR {neg_sample_result['Test']['MRR']} should be > 0.10"
        assert fixed_neg_sample_result["Test"]["MRR"] > 0.10, \
            f"FixedNegSample MRR {fixed_neg_sample_result['Test']['MRR']} should be > 0.10"

        # Results should be within a reasonable margin (not more than 2x difference)
        mrr_ratio = neg_sample_result["Test"]["MRR"] / fixed_neg_sample_result["Test"]["MRR"]
        assert 0.5 < mrr_ratio < 2.0, \
            f"MRR ratio {mrr_ratio} between NegSample and FixedNegSample is too large"

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_keci_transformer_k_vs_all(self):
        """Test KeciTransformer with KvsAll scoring technique."""
        args = Namespace()
        args.model = 'KeciTransformer'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.p = 0
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        result = Execute(args).start()

        # KeciTransformer should achieve reasonable performance
        assert result["Test"]["MRR"] > 0.05, \
            f"KeciTransformer MRR {result['Test']['MRR']} should be > 0.05"

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_keci_transformer_1_vs_all(self):
        """Test KeciTransformer with 1vsAll scoring technique."""
        args = Namespace()
        args.model = 'KeciTransformer'
        args.scoring_technique = '1vsAll'
        args.optim = 'Adam'
        args.p = 0
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        result = Execute(args).start()

        # KeciTransformer should achieve reasonable performance
        assert result["Test"]["MRR"] > 0.05, \
            f"KeciTransformer MRR {result['Test']['MRR']} should be > 0.05"

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_keci_transformer_with_p_and_q(self):
        """Test KeciTransformer with both p and q > 0."""
        args = Namespace()
        args.model = 'KeciTransformer'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Adam'
        args.p = 1
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 10
        args.batch_size = 1024
        args.lr = 0.01
        args.embedding_dim = 33  # Must be divisible by (p + q + 1) = 3
        args.eval_model = 'train_val_test'
        result = Execute(args).start()

        # KeciTransformer should run without errors
        assert result is not None
        assert "Test" in result