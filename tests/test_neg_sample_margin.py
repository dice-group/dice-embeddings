import pytest
import torch

from dicee.config import Namespace
from dicee.executer import Execute
from dicee.models.base_model import BaseKGELightning


class DummyMarginModel(BaseKGELightning):
    """Minimal stand-in exposing loss_function/self.loss for unit-level checks."""

    def __init__(self, margin: float):
        super().__init__()
        self.args = {"scoring_technique": "NegSampleMargin", "margin": margin}
        self.loss = torch.nn.MarginRankingLoss(margin=margin)

    def loss_function(self, yhat_batch, y_batch):
        if self.args.get("scoring_technique") == "NegSampleMargin":
            pos_mask = y_batch >= 0.5
            pos_scores = yhat_batch[pos_mask]
            neg_scores = yhat_batch[~pos_mask]
            neg_ratio = neg_scores.numel() // pos_scores.numel()
            pos_scores = pos_scores.repeat(neg_ratio)
            target = torch.ones_like(pos_scores)
            return self.loss(pos_scores, neg_scores, target)
        return self.loss(yhat_batch, y_batch)


class TestNegSampleMarginLossFunction:
    def test_neg_ratio_1_zero_loss_when_margin_satisfied(self):
        model = DummyMarginModel(margin=1.0)
        # 2 positives followed by 1 tiled negative block (neg_ratio=1)
        yhat = torch.tensor([5.0, 5.0, 1.0, 1.0])
        y = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = model.loss_function(yhat, y)
        assert loss.item() == pytest.approx(0.0)

    def test_neg_ratio_1_positive_loss_when_margin_violated(self):
        model = DummyMarginModel(margin=1.0)
        yhat = torch.tensor([1.0, 1.0, 1.0, 1.0])
        y = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = model.loss_function(yhat, y)
        # pos - neg == 0 < margin(1.0) => loss == margin
        assert loss.item() == pytest.approx(1.0)

    def test_neg_ratio_3_alignment(self):
        model = DummyMarginModel(margin=1.0)
        # 2 positives, then 3 tiled negative blocks of size 2 each
        yhat = torch.tensor([5.0, 5.0,  1.0, 1.0,  1.0, 1.0,  1.0, 1.0])
        y = torch.tensor([1.0, 1.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0])
        loss = model.loss_function(yhat, y)
        assert loss.item() == pytest.approx(0.0)


class TestNegSampleMarginTraining:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_keci_neg_sample_margin(self):
        args = Namespace()
        args.model = 'Keci'
        args.scoring_technique = 'NegSampleMargin'
        args.optim = 'Adam'
        args.p = 0
        args.q = 1
        args.dataset_dir = 'KGs/UMLS'
        args.num_epochs = 20
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 32
        args.neg_ratio = 5
        args.margin = 1.0
        args.eval_model = 'train_val_test'
        result = Execute(args).start()

        assert result["Test"]["MRR"] > 0.05, \
            f"NegSampleMargin MRR {result['Test']['MRR']} should be > 0.05"
