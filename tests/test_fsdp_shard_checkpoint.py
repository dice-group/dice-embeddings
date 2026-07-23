"""Unit tests for the FSDP sharded checkpoint save/resume mechanism.

Covers (non-distributed, single-process — no torchrun / GPU required):
- _LocalSparseAdam.state_dict / load_state_dict round trip
- FSDPShardedEntityModel.save_local_shard_checkpoint / load_local_shard_checkpoint
  round trip: weights and optimizer state survive a save+load cycle unchanged
- load_local_shard_checkpoint rejects a checkpoint saved with a different
  world_size (shard-layout mismatch), instead of silently loading wrong rows
"""
import os

import pytest
import torch

from dicee.models.fsdp_models import _LocalSparseAdam, create_fsdp_sharded_model_class
from dicee.models.real import DistMult


def _minimal_args(num_entities: int = 37, num_relations: int = 5, embedding_dim: int = 8) -> dict:
    return dict(
        model="DistMult",
        embedding_dim=embedding_dim,
        num_entities=num_entities,
        num_relations=num_relations,
        learning_rate=0.01,
        optim="Adam",
        scoring_technique="NegSample",
        input_dropout_rate=0.0,
        hidden_dropout_rate=0.0,
        normalization=None,
        init_param=None,
        byte_pair_encoding=False,
    )


def _make_sharded_model(**overrides) -> DistMult:
    args = _minimal_args(**overrides)
    sharded_cls = create_fsdp_sharded_model_class(DistMult)
    model = sharded_cls(args)
    model.setup_fsdp_training(device=torch.device("cpu"), lr=0.01)
    return model


class TestLocalSparseAdamStateDict:
    def test_round_trip_preserves_moments_and_step_count(self):
        adam = _LocalSparseAdam(local_rows=10, embedding_dim=4, device=torch.device("cpu"))
        weight = torch.nn.Parameter(torch.randn(10, 4))
        grad = torch.randn(10, 4).to_sparse(1)
        adam.step(weight, grad)
        adam.step(weight, grad)

        state = adam.state_dict()
        assert state["step_count"] == 2

        restored = _LocalSparseAdam(local_rows=10, embedding_dim=4, device=torch.device("cpu"))
        restored.load_state_dict(state)

        assert restored.step_count == adam.step_count
        torch.testing.assert_close(restored.exp_avg, adam.exp_avg)
        torch.testing.assert_close(restored.exp_avg_sq, adam.exp_avg_sq)


class TestShardCheckpointRoundTrip:
    def test_save_then_load_restores_weight_and_optimizer_state(self, tmp_path):
        model = _make_sharded_model()
        adapter = model._fsdp_adapter

        # Advance the local optimizer so exp_avg/exp_avg_sq/step_count are non-trivial.
        fake_grad = torch.randn_like(adapter.weight).to_sparse(1)
        adapter._local_adam.step(adapter.weight, fake_grad)
        adapter._local_adam.step(adapter.weight, fake_grad)

        saved_weight = adapter.weight.data.clone()
        saved_exp_avg = adapter._local_adam.exp_avg.clone()
        saved_exp_avg_sq = adapter._local_adam.exp_avg_sq.clone()
        saved_step_count = adapter._local_adam.step_count

        ckpt_path = os.path.join(tmp_path, "entity_shard_rank0_of_1.pt")
        model.save_local_shard_checkpoint(ckpt_path)
        assert os.path.isfile(ckpt_path)

        # Perturb the live model so we can tell load actually overwrote it.
        adapter.weight.data.zero_()
        adapter._local_adam.exp_avg.zero_()
        adapter._local_adam.exp_avg_sq.zero_()
        adapter._local_adam.step_count = 0

        model.load_local_shard_checkpoint(ckpt_path)

        torch.testing.assert_close(adapter.weight.data, saved_weight)
        torch.testing.assert_close(adapter._local_adam.exp_avg, saved_exp_avg)
        torch.testing.assert_close(adapter._local_adam.exp_avg_sq, saved_exp_avg_sq)
        assert adapter._local_adam.step_count == saved_step_count

    def test_load_rejects_world_size_mismatch(self, tmp_path):
        saved_by = _make_sharded_model(num_entities=37)
        ckpt_path = os.path.join(tmp_path, "entity_shard_rank0_of_1.pt")
        saved_by.save_local_shard_checkpoint(ckpt_path)

        # Tamper with the checkpoint to simulate it having been saved under a
        # different world_size — same num_entities, different shard boundaries.
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["world_size"] = 4
        ckpt["start_row"] = 19
        ckpt["local_rows"] = 9
        torch.save(ckpt, ckpt_path)

        resuming = _make_sharded_model(num_entities=37)
        with pytest.raises(RuntimeError, match="world_size"):
            resuming.load_local_shard_checkpoint(ckpt_path)

    def test_save_before_setup_fsdp_training_raises(self, tmp_path):
        args = _minimal_args()
        sharded_cls = create_fsdp_sharded_model_class(DistMult)
        model = sharded_cls(args)  # setup_fsdp_training() not called yet
        with pytest.raises(RuntimeError, match="setup_fsdp_training"):
            model.save_local_shard_checkpoint(os.path.join(tmp_path, "shard.pt"))
