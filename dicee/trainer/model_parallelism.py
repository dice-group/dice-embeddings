import torch
from ..abstracts import AbstractTrainer
from ..static_funcs_training import make_iterable_verbose

class MP(AbstractTrainer):

    def __init__(self, args, callbacks):
        super().__init__(args, callbacks)

    def get_ensemble(self):
        return self.models
    def fit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        models, = args
        for i in models:
            self.on_fit_start(self, i)

        for epoch in (tqdm_bar := make_iterable_verbose(range(self.attributes.num_epochs),
                                                        verbose=True, position=0, leave=True)):
            epoch_loss = 0
            num_of_batches = len(kwargs['train_dataloaders'])
            for i, z in enumerate(kwargs['train_dataloaders']):
                source, targets = self.extract_input_outputs(z)
                yhat = 0
                # Perform forward for each model
                for kge_model in models:
                    source = tuple(_.to(kge_model.device) for _ in source) if isinstance(source, tuple) else source.to(kge_model.device)
                    yhat+= kge_model(source).to("cpu")
                # Normalize
                yhat /=len(models)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat, targets)

                loss.backward()
                for opt in models.optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                batch_loss = loss.item()
                epoch_loss += batch_loss
                if hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                    else:
                        tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")

        for kge_model in models:
            self.on_fit_end(self, kge_model)


    def extract_input_outputs(self, z: list):
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        if len(z) == 2:
            x_batch, y_batch = z
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            # x_batch, y_batch = x_batch.pin_memory().to(self.local_rank, non_blocking=True), y_batch.pin_memory().to(self.local_rank, non_blocking=True)
            return x_batch, y_batch
        elif len(z) == 3:
            x_batch, y_idx_batch, y_batch, = z
            # x_batch, y_batch,y_idx_batch = x_batch.pin_memory().to(self.local_rank, non_blocking=True), y_batch.pin_memory().to(self.local_rank, non_blocking=True),y_idx_batch.pin_memory().to(self.local_rank, non_blocking=True)
            return (x_batch, y_idx_batch), y_batch
        else:
            raise ValueError('Unexpected batch shape..')

