import torch
import copy

class EnsembleKGE:
    def __init__(self, model):
        self.models = []
        self.optimizers=[]

        for i in range(torch.cuda.device_count()):
            i_model=copy.deepcopy(model)
            i_model.to(torch.device(f"cuda:{i}"))
            i_model = torch.compile(i_model)
            self.optimizers.append(i_model.configure_optimizers())
            self.models.append(i_model)

    def __iter__(self):
        return (i for i in self.models)

    def __len__(self):
        return len(self.models)

    def __call__(self, *args, **kwargs):
        # Forward
        results = None
        for model in self.models:
            if results is None:
                results=model(*args, **kwargs)
            else:
                results += model(*args, **kwargs)
        return results/len(self.models)

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

    def __str__(self):
        return f"EnsembleKGE of {len(self.models)} {self.models[0]}"
