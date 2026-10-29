"""Rig-only wrapper: force dicee's PL trainer into 2-node DDP, then run the normal dicee entry point."""
import runpy

import lightning
import lightning.pytorch as pl

_Trainer = pl.Trainer


class DDPTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        kwargs.update(num_nodes=2, strategy="ddp")
        super().__init__(*args, **kwargs)
        print(f"RIG strategy={type(self.strategy).__name__} world_size={self.world_size} global_rank={self.global_rank}", flush=True)


pl.Trainer = lightning.Trainer = DDPTrainer
runpy.run_path("/dicee/dicee/scripts/run.py", run_name="__main__")
