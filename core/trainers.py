import torch


class CustomTrainer:
    """ Custom Trainer"""

    def __init__(self, args):
        self.attributes = vars(args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getattr__(self, attr):
        return self.attributes[attr]

    def fit(self, *args, **kwargs):
        assert len(args) == 1
        model, = args
        print(model)
        print(kwargs)
        data_loader = kwargs['train_dataloaders']
        optimizer = model.configure_optimizers()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            # TODO we also need ModelParallel
            # We can also store embedding matrixes in CPU
            # and do the computaiton in GPU
            # https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
            # https://fairscale.readthedocs.io/en/latest/tutorials/oss.html ?
            model = nn.DataParallel(model)
            self.device = torch.device("cuda:0")
            model.to(self.device)

        # model = model.to(torch.float)
        for epoch in range(self.attributes['max_epochs']):
            epoch_loss = 0
            i = 1
            for i, z in enumerate(data_loader):
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                # Forward pass
                batch_loss = model.training_step(batch=[i.to(self.device) for i in z], batch_idx=i)

                epoch_loss += batch_loss.item()
                # Backward pass
                batch_loss.backward()
                # Adjust learning weights
                optimizer.step()
            print(f"{epoch} epoch: Average batch loss:{epoch_loss / i:.3f}")

    @staticmethod
    def save_checkpoint(path):
        print('no checkpoint saving')
