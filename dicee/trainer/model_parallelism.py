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
        model, = args
        # (1) Run the fit the start callback.
        self.on_fit_start(self, model)
        # (2) Setup DDP.
        torch.distributed.init_process_group(backend="nccl")


        from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import loss_parallel
        from torch.distributed._tensor import Shard
        import os

        # create a device mesh based on the given world_size.
        device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(int(os.environ["WORLD_SIZE"]),))

        _rank = device_mesh.get_rank()

        # create model and move it to GPU.  Init_device_mesh has already assigned gpu ids...
        model = model.to("cuda")

        # Custom parallelization plan for the model
        model = parallelize_module(
            module=model,
            device_mesh=device_mesh,
            parallelize_plan={
                "entity_embeddings": ColwiseParallel(input_layouts=Shard(0)),
                "relation_embeddings": ColwiseParallel(output_layouts=Shard(0))})
        # optimizer = model.selected_optimizer



        for epoch in (tqdm_bar := make_iterable_verbose(range(self.attributes.num_epochs),
                                                        verbose=True, position=0, leave=True)):
            epoch_loss = 0
            num_of_batches = len(kwargs['train_dataloaders'])
            for i, z in enumerate(kwargs['train_dataloaders']):
                source, targets = self.extract_input_outputs(z)
                dist_input = distribute_tensor(source, device_mesh, placements=[Shard(1)])

                yhat = model(dist_input)
                continue
                loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                batch_loss = loss.item()
                epoch_loss += batch_loss
                if hasattr(tqdm_bar, 'set_description_str'):
                    tqdm_bar.set_description_str(f"Epoch:{epoch + 1}")
                    if i > 0:
                        tqdm_bar.set_postfix_str(f"batch={i} | {num_of_batches}, loss_step={batch_loss:.5f}, loss_epoch={epoch_loss / i:.5f}")
                    else:
                        tqdm_bar.set_postfix_str(f"loss_step={batch_loss:.5f}, loss_epoch={batch_loss:.5f}")


        # (3) Start NodeTrainer.
        # NodeTrainer(self, model, train_dataset_loader, self.callbacks, self.attributes.num_epochs).train()
        torch.distributed.destroy_process_group()
        self.on_fit_end(self, model)

    def odlfit(self, *args, **kwargs):
        """ Train model        """
        assert len(args) == 1
        model, = args

        from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        from torch.distributed.device_mesh import init_device_mesh
        # Define the module.
        tp_mesh = init_device_mesh("cuda", (1,))
        model = parallelize_module(model, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})

        self.on_fit_start(self, model)


        exit(1)

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

