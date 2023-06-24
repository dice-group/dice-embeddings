from pykeen.contrib.lightning import LCWALitModule
import torch
from .pykeen_Module import *
from pykeen.triples.triples_factory import CoreTriplesFactory
import wandb

class MyLCWALitModule(LCWALitModule,Pykeen_Module):

    def __init__(self, *, model_name: str,args, **kwargs):
        Pykeen_Module.__init__(self,model_name , kwargs['optimizer'])
        super().__init__(**kwargs)
        self.loss_history = []
        self.args=args
        self.train_dataloaders = self.train_dataloader()


    def training_epoch_end(self, training_step_outputs) -> None:
        # batch_losses = [i["loss"].item() for i in training_step_outputs]
        # avg = sum(batch_losses) / len(batch_losses)
        
        # log_dict = {
        #     'train_loss':avg,
        #     "epoch":self.current_epoch+1,
            
        # }
        # print(avg)
        # wandb.log(log_dict)
        
        # self.loss_history.append(avg)
        # self.log('val_loss',avg,on_epoch=True)
        # print(training_step_outputs)
        # log_dict = {
        #     'train_loss':training_step_outputs,
        #     "epoch":self.current_epoch+1,
            
        # }
        # wandb.log(log_dict)
        pass

    def _dataloader(
        self, triples_factory: CoreTriplesFactory, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        
        
        return torch.utils.data.DataLoader(dataset=triples_factory.create_lcwa_instances(), batch_size=self.args['batch_size'], shuffle=True,
                              num_workers=self.args['num_core'], persistent_workers=True)



  