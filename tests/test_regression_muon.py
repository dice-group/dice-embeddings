from dicee.executer import Execute
import pytest
from dicee.config import Namespace


class TestRegressionMuon:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_keci_p0_q0_muon(self):
        args = Namespace()
        args.model = 'Keci'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Muon'
        args.p = 0
        args.q = 0
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 32
        args.batch_size = 1024
        args.lr = 0.02
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        result = Execute(args).start()
        assert result["Train"]["MRR"] > 0.0, \
            f"Keci (p=0,q=0) Muon Train MRR should be > 0, got {result['Train']['MRR']}"
        assert result["Test"]["MRR"] > 0.0, \
            f"Keci (p=0,q=0) Muon Test MRR should be > 0, got {result['Test']['MRR']}"

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_decal_p0_q0_r0_muon(self):
        args = Namespace()
        args.model = 'DeCaL'
        args.scoring_technique = 'KvsAll'
        args.optim = 'Muon'
        args.p = 0
        args.q = 0
        args.r = 0
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 32
        args.batch_size = 1024
        args.lr = 0.02
        args.embedding_dim = 32
        args.eval_model = 'train_val_test'
        result = Execute(args).start()
        assert result["Train"]["MRR"] > 0.0, \
            f"DeCaL (p=0,q=0,r=0) Muon Train MRR should be > 0, got {result['Train']['MRR']}"
        assert result["Test"]["MRR"] > 0.0, \
            f"DeCaL (p=0,q=0,r=0) Muon Test MRR should be > 0, got {result['Test']['MRR']}"

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_keci_vs_decal_muon(self):
        """Compare Keci (p=0,q=0) vs DeCaL (p=0,q=0,r=0) with Muon — both on UMLS."""
        base_args = dict(
            scoring_technique='KvsAll',
            optim='Muon',
            p=0,
            q=0,
            dataset_dir='KGs/UMLS',
            num_epochs=32,
            batch_size=1024,
            lr=0.02,
            embedding_dim=32,
            eval_model='train_val_test',
        )

        keci_args = Namespace()
        keci_args.__dict__.update(base_args)
        keci_args.model = 'Keci'
        keci_result = Execute(keci_args).start()

        decal_args = Namespace()
        decal_args.__dict__.update(base_args)
        decal_args.model = 'DeCaL'
        decal_args.r = 0
        decal_result = Execute(decal_args).start()

        keci_test_mrr = keci_result["Test"]["MRR"]
        decal_test_mrr = decal_result["Test"]["MRR"]
        print(f"\nKeci  (p=0,q=0)      Test MRR: {keci_test_mrr:.4f}")
        print(f"DeCaL (p=0,q=0,r=0)  Test MRR: {decal_test_mrr:.4f}")
        winner = "DeCaL" if decal_test_mrr > keci_test_mrr else "Keci"
        print(f"Winner: {winner}")

        assert keci_test_mrr > 0.0, f"Keci Test MRR should be > 0, got {keci_test_mrr}"
        assert decal_test_mrr > 0.0, f"DeCaL Test MRR should be > 0, got {decal_test_mrr}"
        assert keci_test_mrr > decal_test_mrr, (
            f"Keci (p=0,q=0) expected to outperform DeCaL (p=0,q=0,r=0) with Muon, "
            f"but got Keci={keci_test_mrr:.4f} <= DeCaL={decal_test_mrr:.4f}"
        )
