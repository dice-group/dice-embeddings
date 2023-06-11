from pykeen.pipeline import pipeline
from pykeen.sampling import BasicNegativeSampler

pipeline_result = pipeline(
      model="Distmult",
      dataset="FB15k237",
      model_kwargs=dict(
          embedding_dim=64,
      ),

      training_loop="LCWA",
      
      training_kwargs=dict(
          batch_size=256,
      ),
      # negative_sampler=BasicNegativeSampler,
      # negative_sampler_kwargs=dict(
      #     num_negs_per_pos=32,
      # ),
      optimizer="adam",
      optimizer_kwargs=dict(lr=0.001849662035249092),
      epochs=100,
      evaluator="RankBasedEvaluator",
      # result_tracker='wandb',
      # result_tracker_kwargs=dict(
      #     project='pykeen_project',
      # ),
  )
pipeline_result.save_to_directory(f"/upb/users/r/renzhong/profiles/unix/cs/Dicee/dice-embeddings/benchmarks/pykeen_bigdataset/GPU/distmult_fb15k")


pipeline_result = pipeline(
      model="complex",
      dataset="FB15k237",
      model_kwargs=dict(
          embedding_dim=256,
      ),

      training_loop="LCWA",
      
      training_kwargs=dict(
          batch_size=256,
      ),
      # negative_sampler=BasicNegativeSampler,
      # negative_sampler_kwargs=dict(
      #     num_negs_per_pos=32,
      # ),
      optimizer="adam",
      optimizer_kwargs=dict(lr=0.007525067744232913),
      epochs=100,
      evaluator="RankBasedEvaluator",
      # result_tracker='wandb',
      # result_tracker_kwargs=dict(
      #     project='pykeen_project',
      # ),
  )


pipeline_result.save_to_directory(f"/upb/users/r/renzhong/profiles/unix/cs/Dicee/dice-embeddings/benchmarks/pykeen_bigdataset/GPU/complex_fb15k")


pipeline_result = pipeline(
      model="distmult",
      dataset="YAGO310",
      model_kwargs=dict(
          embedding_dim=256,
      ),

      training_loop="LCWA",
      
      training_kwargs=dict(
          batch_size=4096,
      ),
      # negative_sampler=BasicNegativeSampler,
      # negative_sampler_kwargs=dict(
      #     num_negs_per_pos=42,
      # ),
      optimizer="adam",
      optimizer_kwargs=dict(lr=0.00113355532419969),
      epochs=100,
      evaluator="RankBasedEvaluator",
      # result_tracker='wandb',
      # result_tracker_kwargs=dict(
      #     project='pykeen_project',
      # ),
  )

pipeline_result.save_to_directory(f"/upb/users/r/renzhong/profiles/unix/cs/Dicee/dice-embeddings/benchmarks/pykeen_bigdataset/GPU/distmult_yago310")


pipeline_result = pipeline(
      model="complex",
      dataset="YAGO310",
      model_kwargs=dict(
          embedding_dim=256,
      ),

      training_loop="LCWA",
      
      training_kwargs=dict(
          batch_size=8192,
      ),
      # negative_sampler=BasicNegativeSampler,
      # negative_sampler_kwargs=dict(
      #     num_negs_per_pos=32,
      # ),
      optimizer="adam",
      optimizer_kwargs=dict(lr=0.001723135381847608),
      epochs=100,
      evaluator="RankBasedEvaluator",
      # result_tracker='wandb',
      # result_tracker_kwargs=dict(
      #     project='pykeen_project',
      # ),
  )
pipeline_result.save_to_directory(f"/upb/users/r/renzhong/profiles/unix/cs/Dicee/dice-embeddings/benchmarks/pykeen_bigdataset/GPU/complex_yago310")