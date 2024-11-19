import torch
import copy

class EnsembleKGE:
    def __init__(self, seed_model):
        self.models = []
        self.optimizers = []
        self.loss_history = []
        for i in range(torch.cuda.device_count()):
            i_model=copy.deepcopy(seed_model)
            i_model.to(torch.device(f"cuda:{i}"))
            i_model = torch.compile(i_model)
            self.optimizers.append(i_model.configure_optimizers())
            self.models.append(i_model)
    def named_children(self):
        return self.models[0].named_children()
    @property
    def example_input_array(self):
        return self.models[0].example_input_array
    @property
    def _trainer(self):
        return self.models[0]._trainer

    def parameters(self):
        return self.models[0].parameters()
    def modules(self):
        return self.models[0].modules()

    def __iter__(self):
        return (i for i in self.models)

    def __len__(self):
        return len(self.models)

    def __call__(self,x_batch):
        for opt in self.optimizers:
            opt.zero_grad()
        yhat=None
        for gpu_id, model in enumerate(self.models):
            # Move batch into the GPU where the i.th model resides
            if isinstance(x_batch, tuple):
                x_batch=(x_batch[0].to(f"cuda:{gpu_id}"),x_batch[1].to(f"cuda:{gpu_id}"))
            else:
                x_batch=x_batch.to(f"cuda:{gpu_id}")
            if yhat is None:
                yhat=model(x_batch)
            else:
                yhat+=model(x_batch).to("cuda:0")
        return yhat/len(self.models)
    
    def step(self):
        for opt in self.optimizers:
            opt.step()
    
    """
    def __getattr__(self, name):
        # Create a function that will call the same attribute/method on each model
        def method(*args, **kwargs):
            results = []
            for model in self.models:
                attr = getattr(model, name)
                if callable(attr):
                    # If it's a method, call it with provided arguments
                    results.append(attr(*args, **kwargs))
                else:
                    # If it's an attribute, just get its value
                    results.append(attr)
            return results
        return method
    """
    def __str__(self):
        return f"EnsembleKGE of {len(self.models)} {self.models[0]}"
