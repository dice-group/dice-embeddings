import torch
import lightning as pl
import os, psutil, numpy as np
from dicee.abstracts import AbstractTrainer
from typing import List


class MemoryOptimizedTrainer(AbstractTrainer):

    def __init__(self, args, callbacks=None):
        super().__init__(args, callbacks)
        self.args = args
        self.model = None
        self.train_dataloaders = None
        self.form_of_labelling = None
        self.dataset = None
        self.evaluator = None

        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed_all(self.args.random_seed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.process = psutil.Process(os.getpid())

        # CPU cache settings
        self.threshold = self.args.threshold
        self.entity_capacity = getattr(self.args, "max_entities", self.threshold)
        self.entity_embed_cache = {}    # {ent_id: embedding}
        self.entity_id_to_slot = {}
        self.slot_to_entity_id = {}
        self.next_free_slot = 0

        print(f'# of CPUs:{os.cpu_count()} | # of GPUs:{torch.cuda.device_count()} | '
              f'# of CPUs for dataloader:{self.args.num_core}')
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))


    # Training
    def fit(self, *args, train_dataloaders, **kwargs) -> None:
        assert len(args) == 1
        self.model, = args
        self.train_dataloaders = train_dataloaders
        self.model.to(self.device)

        self._wrap_training_step()

        trainer = pl.Trainer(
            accelerator=kwargs.get("accelerator", "auto"),
            strategy=kwargs.get("strategy", "auto"),
            num_nodes=kwargs.get("num_nodes", 1),
            precision=kwargs.get("precision", None),
            logger=kwargs.get("logger", None),
            fast_dev_run=kwargs.get("fast_dev_run", False),
            max_epochs=self.args.num_epochs,
            min_epochs=self.args.num_epochs,
            max_steps=kwargs.get("max_step", -1),
            min_steps=kwargs.get("min_steps", None),
            detect_anomaly=False,
        )

        trainer.fit(self.model, train_dataloaders)

    def _wrap_training_step(self):
        orig_train_step = self.model.training_step
        def wrapped_training_step(batch, batch_idx):
            s_ids = torch.stack([item[0] for item in batch[0]])
            o_ids = torch.stack([item[1] for item in batch[0]])

            entity_ids = torch.unique(torch.cat([s_ids, o_ids])).tolist()

            #print("no. of unique entity_ids in batch: ", len(entity_ids))

            #  Load embeddings from CPU cache into GPU slots
            self._hydrate_model_from_cache(entity_ids)

            loss = orig_train_step(batch, batch_idx)

            #  Save embeddings back to CPU cache
            self._cache_update_from_model(entity_ids)

            return loss

        self.model.training_step = wrapped_training_step


    # Cache operations
    def _hydrate_model_from_cache(self, entity_ids: List[int]):
        """Load embeddings from CPU cache into GPU model."""
        for eid in entity_ids:
            slot = self.entity_id_to_slot.get(eid, None)
            if slot is None:
                # assign a new slot
                if self.next_free_slot >= self.entity_capacity:
                    evict_slot = np.random.choice(list(self.slot_to_entity_id.keys()))
                    evict_eid = self.slot_to_entity_id[evict_slot]
                    #print("entity capacity full, evict_eid, slot: ", evict_eid, evict_slot)
                    # save evicted entity to CPU cache
                    self.entity_embed_cache[evict_eid] = self.model.entity_embeddings.weight[evict_slot].detach().cpu().clone()
                    del self.entity_id_to_slot[evict_eid]
                    del self.slot_to_entity_id[evict_slot]
                    slot = evict_slot
                else:
                    slot = self.next_free_slot
                    self.next_free_slot += 1

                self.entity_id_to_slot[eid] = slot
                self.slot_to_entity_id[slot] = eid

                # initialize embedding from cache if exists, else random init
                if eid in self.entity_embed_cache:
                    self.model.entity_embeddings.weight[slot].data.copy_(
                        self.entity_embed_cache[eid].to(self.model.entity_embeddings.weight.device)
                    )
                else:
                    torch.nn.init.normal_(self.model.entity_embeddings.weight[slot], std=0.01)

    def _cache_update_from_model(self, entity_ids: List[int]):
        """Save embeddings from GPU model into CPU cache."""
        for eid in entity_ids:
            slot = self.entity_id_to_slot.get(eid, None)
            if slot is not None:
                self.entity_embed_cache[eid] = self.model.entity_embeddings.weight[slot].detach().cpu().clone()
                #print("embedding saved to cpu entity_embed_cache")
