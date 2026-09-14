from types import SimpleNamespace

import pytest
import torch

from dicee.trainer.torch_trainer import TorchTrainer


@pytest.mark.parametrize('accumulation_steps', [0, 1, 2, 3, 8])
def test_accumulation_matches_grouped_batches(accumulation_steps):
    trainer = TorchTrainer(SimpleNamespace(
        random_seed=0, gradient_accumulation_steps=accumulation_steps,
    ), [])
    targets = torch.tensor([1., -2., 3., 0., 2.])
    trainer.train_dataloaders = targets
    weight = torch.nn.Parameter(torch.tensor(0.0))
    reference = torch.nn.Parameter(weight.detach().clone())
    trainer.optimizer = torch.optim.SGD([weight], lr=0.1)
    optimizer = torch.optim.SGD([reference], lr=0.1)
    trainer.training_step = lambda batch: (weight - batch[1]).square()
    steps = max(1, accumulation_steps)

    for _ in range(2):
        for i, target in enumerate(targets):
            expected_loss = (weight.detach() - target).square().item()
            loss = trainer._run_batch(i, None, target)
            assert loss == pytest.approx(expected_loss)
            if (i + 1) % steps == 0 or i + 1 == len(targets):
                start = i - i % steps
                optimizer.zero_grad()
                (reference - targets[start:i + 1]).square().mean().backward()
                optimizer.step()
            torch.testing.assert_close(weight, reference)
