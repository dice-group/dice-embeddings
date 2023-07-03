from pykeen.datasets import UMLS,Kinships
from pykeen.models import ERModel,DistMult,Model,ComplEx
from pykeen.sampling import BasicNegativeSampler
from torch.optim import Adam
import time
import torch
from pykeen.training import LCWATrainingLoop,SLCWATrainingLoop
from pykeen.evaluation import LCWAEvaluationLoop,SampledRankBasedEvaluator,RankBasedEvaluator


def experiment_umls_distmult():
  start_time = time.time()
  
  dataset = UMLS()
  training_triples_factory = dataset.training
  model = DistMult(triples_factory=training_triples_factory,
            embedding_dim=128,

            
       )
  
  model.to(torch.device('cpu'))
  # model.to(torch.device('cuda'))
  training_triples_factory =training_triples_factory
  optimizer = Adam(params=model.get_grad_params())


  training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    optimizer_kwargs=dict(lr=0.01),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos=32,
  ),
  )
  
  # training_loop = LCWATrainingLoop(
  #   model=model,
  #   triples_factory=training_triples_factory,
  #   optimizer=optimizer,
  #   optimizer_kwargs=dict(lr=0.01),

  # )

        
  

  _ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=512,
    
  )

  report_time = time.time() - start_time
  
    # Pick an evaluation loop (NEW)
  
  evaluator = RankBasedEvaluator()

# Get triples to test
  mapped_triples = dataset.testing.mapped_triples

  # Evaluate
  results = evaluator.evaluate(
      model=model,
      mapped_triples=mapped_triples,
      batch_size=512,
      additional_filter_triples=[
          dataset.training.mapped_triples,
          dataset.validation.mapped_triples,
      ],
  )
  
  

  
  return report_time,results


def experiment_umls_complex():
  start_time = time.time()
  
  dataset = UMLS()
  training_triples_factory = dataset.training
  model = ComplEx(triples_factory=training_triples_factory,
            embedding_dim=32,

            
       )
  # model.to(torch.device('cuda'))
  model.to(torch.device('cpu'))
  training_triples_factory =training_triples_factory
  optimizer = Adam(params=model.get_grad_params())


  training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    optimizer_kwargs=dict(lr=0.1),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos=32,
  ),   
  )
  
  # training_loop = LCWATrainingLoop(
  #   model=model,
  #   triples_factory=training_triples_factory,
  #   optimizer=optimizer,
  #   optimizer_kwargs=dict(lr=0.1),

  # )


  _ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=512,
    
  )

  report_time = time.time() - start_time
  
  
  evaluator = RankBasedEvaluator()

# Get triples to test
  mapped_triples = dataset.testing.mapped_triples

  # Evaluate
  results = evaluator.evaluate(
      model=model,
      mapped_triples=mapped_triples,
      batch_size=512,
      additional_filter_triples=[
          dataset.training.mapped_triples,
          dataset.validation.mapped_triples,
      ],
  )
  
  return report_time,results


def experiment_kinship_distmult():
  start_time = time.time()
  
  dataset = Kinships()
  training_triples_factory = dataset.training
  model = DistMult(triples_factory=training_triples_factory,
            embedding_dim=64,

            
       )
  # model.to(torch.device('cuda'))
  model.to(torch.device('cpu'))
  training_triples_factory =training_triples_factory
  optimizer = Adam(params=model.get_grad_params())


  training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    optimizer_kwargs=dict(lr=0.1),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos=32,
  ),   
  )
  
  # training_loop = LCWATrainingLoop(
  #   model=model,
  #   triples_factory=training_triples_factory,
  #   optimizer=optimizer,
  #   optimizer_kwargs=dict(lr=0.1),

  # )


  _ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=512,
    
  )

  report_time = time.time() - start_time
  
  
  evaluator = RankBasedEvaluator()

# Get triples to test
  mapped_triples = dataset.testing.mapped_triples

  # Evaluate
  results = evaluator.evaluate(
      model=model,
      mapped_triples=mapped_triples,
      batch_size=512,
      additional_filter_triples=[
          dataset.training.mapped_triples,
          dataset.validation.mapped_triples,
      ],
  )
  
  
  return report_time,results

def experiment_kinship_complex():
  start_time = time.time()
  
  dataset = Kinships()
  training_triples_factory = dataset.training
  model = ComplEx(triples_factory=training_triples_factory,
            embedding_dim=32,

            
       )
  
  # model.to(torch.device('cuda'))
  model.to(torch.device('cpu'))
  training_triples_factory =training_triples_factory
  optimizer = Adam(params=model.get_grad_params())


  training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    optimizer_kwargs=dict(lr=0.1),
    negative_sampler=BasicNegativeSampler,
    negative_sampler_kwargs=dict(
      num_negs_per_pos=32,
  ),   
  )
  
  # training_loop = LCWATrainingLoop(
  #   model=model,
  #   triples_factory=training_triples_factory,
  #   optimizer=optimizer,
  #   optimizer_kwargs=dict(lr=0.1),

  # )


  _ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=100,
    batch_size=512,
    
  )

  report_time = time.time() - start_time
  
  
  evaluator = RankBasedEvaluator()

# Get triples to test
  mapped_triples = dataset.testing.mapped_triples

  # Evaluate
  results = evaluator.evaluate(
      model=model,
      mapped_triples=mapped_triples,
      batch_size=512,
      additional_filter_triples=[
          dataset.training.mapped_triples,
          dataset.validation.mapped_triples,
      ],
  )
  
  return report_time,results



def save_evaluation(results,path):
    # Pickle the object and save it to a file
  file_path = path

  # Save the dictionary as text in a file
  with open(file_path, 'w') as file:
      for key, value in results.data.items():
          file.write(f"{key}: {value}\n")


def save_report(runtime,filename):
  res = dict()
  res['report_time'] = runtime

  with open(filename + '.json', 'w') as fp:
      import json
      json.dump(res, fp)
      print('dictionary saved successfully to file')



for i in range(5):
  
  runtime,results = experiment_umls_distmult()
  save_report(runtime,f'slcwa32_cpu_umls_distmult_{i}')
  # save_evaluation(results,f'gpu_umls_distmult_eval{i}')


  runtime,results  = experiment_umls_complex()
  save_report(runtime,f'slcwa32_cpu_umls_complex_{i}')
  # save_evaluation(results,f'gpu_umls_complex_eval{i}')


  runtime,results  = experiment_kinship_distmult()
  save_report(runtime,f'slcwa32_cpu_kinship_distmult_{i}')
  # save_evaluation(results,f'gpu_kinship_distmult_eval{i}')


  runtime,results  = experiment_kinship_complex()
  save_report(runtime,f'slcwa32_cpu_kinship_complex_{i}')
  # save_evaluation(results,f'gpu_kinship_complex_eval{i}')


