from pykeen.datasets.umls import UMLS_TRAIN_PATH, UMLS_TEST_PATH
from pykeen.datasets import UMLS,FB15k237,YAGO310
from pykeen.models import ERModel,DistMult,Model,ComplEx
from pykeen.sampling import BasicNegativeSampler
from torch.optim import Adam
import time
from pykeen.pipeline import pipeline
import torch
from pykeen.training import LCWATrainingLoop,SLCWATrainingLoop

def experiment_fb15k_distmult():
  start_time = time.time()
  
  dataset = FB15k237()
  training_triples_factory = dataset.training
  model = DistMult(triples_factory=training_triples_factory,
            embedding_dim=64,

            
       )
  
  model.to(torch.device('cuda'))
  
  optimizer = Adam(params=model.get_grad_params())

  
  # training_loop = LCWATrainingLoop(
  #   model=model,
  #   triples_factory=training_triples_factory,
  #   optimizer=optimizer,
  #   optimizer_kwargs=dict(lr=0.001849662035249092),

        
  # )
  training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    optimizer_kwargs=dict(lr=0.1),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos=1,
  ),   
  )


  _ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=512,
    
  )

  report_time = time.time() - start_time
  return report_time



def experiment_fb15k_complex():
  start_time = time.time()
  
  dataset = FB15k237()
  training_triples_factory = dataset.training
  model = ComplEx(triples_factory=training_triples_factory,
            embedding_dim=32,

            
       )
  
  model.to(torch.device('cuda'))
  
  optimizer = Adam(params=model.get_grad_params())

  # from pykeen.training import LCWATrainingLoop
  # training_loop = LCWATrainingLoop(
  #   model=model,
  #   triples_factory=training_triples_factory,
  #   optimizer=optimizer,
  #   optimizer_kwargs=dict(lr=0.007525067744232913),

        
  # )
  training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    optimizer_kwargs=dict(lr=0.1),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos=1,
  ),   
  )


  _ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=512,
    
  )

  report_time = time.time() - start_time
  return report_time


def experiment_yago_distmult():
  start_time = time.time()
  
  dataset = YAGO310()
  training_triples_factory = dataset.training
  model = DistMult(triples_factory=training_triples_factory,
            embedding_dim=64,

            
       )
  
  model.to(torch.device('cuda'))
  
  optimizer = Adam(params=model.get_grad_params())

  # from pykeen.training import LCWATrainingLoop
  # training_loop = LCWATrainingLoop(
  #   model=model,
  #   triples_factory=training_triples_factory,
  #   optimizer=optimizer,
  #   optimizer_kwargs=dict(lr=0.00113355532419969),

        
  # )
  training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    optimizer_kwargs=dict(lr=0.1),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos=1,
  ),   
  )



  _ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=512,
    
  )

  report_time = time.time() - start_time
  return report_time


def experiment_yago_complex():
  start_time = time.time()
  
  dataset = YAGO310()
  training_triples_factory = dataset.training
  model = ComplEx(triples_factory=training_triples_factory,
            embedding_dim=32,

            
       )
  
  model.to(torch.device('cuda'))
  
  optimizer = Adam(params=model.get_grad_params())

  # from pykeen.training import LCWATrainingLoop
  # training_loop = LCWATrainingLoop(
  #   model=model,
  #   triples_factory=training_triples_factory,
  #   optimizer=optimizer,
  #   optimizer_kwargs=dict(lr=0.001723135381847608),

        
  # )

  training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    optimizer_kwargs=dict(lr=0.1),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos=1,
  ),   
  )


  _ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=512,
    
  )

  report_time = time.time() - start_time
  return report_time


def save_report(runtime,filename):
  res = dict()
  res['report_time'] = runtime

  with open(filename + '.json', 'w') as fp:
      import json
      json.dump(res, fp)
      print('dictionary saved successfully to file')


# runtime = experiment_fb15k_distmult()
# save_report(runtime,'fb15k_distmult_slcwa1')

# runtime = experiment_fb15k_complex()
# save_report(runtime,'fb15k_complex_slcwa1')

runtime = experiment_yago_distmult()
# save_report(runtime,'yago_distmult_slcwa1')

# runtime = experiment_yago_complex()
# save_report(runtime,'yago_complex_slcwa1')