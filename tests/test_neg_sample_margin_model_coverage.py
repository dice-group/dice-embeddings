"""Coverage sweep: every dicee model (native and PyKEEN-wrapped), except Pyke
and Shallom, must be trainable with scoring_technique='NegSampleMargin'.

Pyke and Shallom are excluded on purpose:
- Shallom is a RelationPrediction-only model; its forward_triples delegates
  through forward_k_vs_all and indexes the full (batch, batch) relation-score
  matrix rather than gathering per-row scores, so NegSample-family techniques
  do not produce a correctly shaped batch for it.
- Pyke is excluded per project convention (not covered by this sweep).
"""
import pytest

from dicee.config import Namespace
from dicee.executer import Execute
from dicee.static_funcs import MODEL_REGISTRY

EXCLUDED_MODELS = {"Shallom", "Pyke"}

NATIVE_MODELS = sorted(set(MODEL_REGISTRY) - EXCLUDED_MODELS - {"BytE"})

PYKEEN_MODELS = [
    "Pykeen_DistMult", "Pykeen_ComplEx", "Pykeen_HolE", "Pykeen_CP",
    "Pykeen_ProjE", "Pykeen_TuckER", "Pykeen_TransR", "Pykeen_TransH",
    "Pykeen_TransD", "Pykeen_TransE", "Pykeen_QuatE", "Pykeen_MuRE",
    "Pykeen_BoxE", "Pykeen_RotatE", "Pykeen_TransF",
]

# Extra per-model args required by a model's own constructor, independent of
# scoring technique (e.g. Clifford-algebra dimension params, LFMult's
# polynomial degree).
MODEL_EXTRA_ARGS = {
    "DeCaL": {"r": 0},
    "LFMult": {"degree": 1},
}


def _base_args(model_name, **overrides):
    args = Namespace()
    args.model = model_name
    args.dataset_dir = "KGs/UMLS"
    args.optim = "Adam"
    args.num_epochs = 2
    args.batch_size = 1024
    args.lr = 0.01
    args.embedding_dim = 32
    args.input_dropout_rate = 0.0
    args.hidden_dropout_rate = 0.0
    args.feature_map_dropout_rate = 0.0
    args.scoring_technique = "NegSampleMargin"
    args.neg_ratio = 1
    args.margin = 1.0
    args.eval_model = "train"
    args.read_only_few = None
    args.sample_triples_ratio = None
    args.num_folds_for_cv = None
    args.trainer = "torchCPUTrainer"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestNegSampleMarginNativeModelCoverage:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("model_name", NATIVE_MODELS)
    def test_trains(self, model_name):
        args = _base_args(model_name, **MODEL_EXTRA_ARGS.get(model_name, {}))
        result = Execute(args).start()
        assert result is not None

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_byte_trains(self):
        """BytE uses its own next-token-prediction pipeline regardless of
        scoring_technique (byte_pair_encoding routes it through
        MultiClassClassificationDataset), so this only checks it still trains
        end to end alongside every other model in the registry.

        eval_model is left off (None) on purpose: BytE has no fixed, discrete
        entity vocabulary (entities are open subword sequences), so the
        entity-rank evaluator used by NegSample/NegSampleMargin/FixedNegSample
        (which needs self.num_entities) cannot run for it. This is a
        pre-existing limitation shared with plain NegSample, not something
        introduced by NegSampleMargin.
        """
        args = _base_args(
            "BytE",
            embedding_dim=16,
            byte_pair_encoding=True,
            block_size=8,
            eval_model=None,
        )
        result = Execute(args).start()
        assert result is not None


class TestNegSampleMarginPykeenModelCoverage:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize("model_name", PYKEEN_MODELS)
    def test_trains(self, model_name):
        args = Namespace()
        args.dataset_dir = "KGs/UMLS"
        args.trainer = "torchCPUTrainer"
        args.model = model_name
        args.num_epochs = 2
        args.batch_size = 256
        args.lr = 0.1
        args.num_workers = 1
        args.num_core = 1
        args.scoring_technique = "NegSampleMargin"
        args.neg_ratio = 1
        args.margin = 1.0
        args.sample_triples_ratio = None
        args.read_only_few = None
        args.num_folds_for_cv = None
        args.eval_model = "train"
        result = Execute(args).start()
        assert result is not None
