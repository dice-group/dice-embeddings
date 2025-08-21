import torch
import copy
from typing import List
class EnsembleKGE:
    def __init__(self, models : list=None, seed_model=None, pretrained_models:List=None):

        if models is not None:
            self.models = []
            self.optimizers = []
            self.loss_history = []
            for i in range(len(models)):
                i_model = models[i]
                # TODO: Why we cant send the compile model to cpu ?
                #i_model = torch.compile(i_model)
                i_model.to(torch.device(f"cuda:{i}"))
                self.optimizers.append(i_model.configure_optimizers())
                self.models.append(i_model)
        else:
            assert pretrained_models is not None
            self.models = pretrained_models
            self.optimizers = []
            self.loss_history = []

            for i in range(torch.cuda.device_count()):
                self.models[i].to(torch.device(f"cuda:{i}"))
                self.optimizers.append(self.models[i].configure_optimizers())
            # Maybe use the original model's name ?
        self.name=self.models[0].name
        self.train_mode=True
    def named_children(self):
        return self.models[0].named_children()
    @property
    def example_input_array(self):
        return self.models[0].example_input_array
    @property
    def _trainer(self):
        return self.models[0]._trainer

    def parameters(self):
        return [ x  for i in self.models for x in i.parameters()]
        # return self.models[0].parameters()
    def modules(self):
        return [x for i in self.models for x in i.modules()]

    def __iter__(self):
        return (i for i in self.models)

    def __len__(self):
        return len(self.models)

    def eval(self):
        for model in self.models:
            model.eval()
        self.train_mode=False
    def to(self,device):
        for i in range(len(self.models)):
            if device == "cpu":
                self.models[i].cpu()
            else:
                raise NotImplementedError


    def mem_of_model(self):
        mem_of_ensemble={'EstimatedSizeMB': 0, 'NumParam': 0}
        for i in self.models:
            for k,v in i.mem_of_model().items():
                mem_of_ensemble[k] += v
        return mem_of_ensemble
    def __call__(self,x_batch):

        if self.train_mode is False:
            yhat=0
            for gpu_id, model in enumerate(self.models):
                yhat += model(x_batch)
            return yhat / len(self.models)
        else:
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

    def get_embeddings(self):
        entity_embeddings=[]
        relation_embeddings=[]
        # () Iterate
        for trained_model in self.models:
            entity_emb, relation_ebm = trained_model.get_embeddings()
            entity_embeddings.append(entity_emb)
            if relation_ebm is not None:
                relation_embeddings.append(relation_ebm)
        # () Concat the embedding vectors horizontally.
        entity_embeddings=torch.cat(entity_embeddings,dim=1)
        if relation_embeddings:
            relation_embeddings=torch.cat(relation_embeddings,dim=1)
        else:
            relation_embeddings=None

        return entity_embeddings, relation_embeddings

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
