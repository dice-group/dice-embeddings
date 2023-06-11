from pykeen.pipeline import pipeline
from pykeen.sampling import BasicNegativeSampler
from pykeen.datasets.umls import UMLS_TRAIN_PATH, UMLS_TEST_PATH
from pykeen.datasets.yago import YAGO310
from pathlib import Path


def run_experiment(index):
    
    pipeline_result = pipeline(
        model="Distmult",
        device="gpu",
        model_kwargs=dict(
            embedding_dim=128,
        ),
        training=UMLS_TRAIN_PATH,
        testing=UMLS_TEST_PATH,
        training_loop="sLCWA",
        
        training_kwargs=dict(
            batch_size=512,
        ),
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=16,
        ),
        optimizer="adam",
        optimizer_kwargs=dict(lr=0.01),
        epochs=100,
        evaluator="RankBasedEvaluator",
        # result_tracker='wandb',
        # result_tracker_kwargs=dict(
        #     project='pykeen_project',
        # ),
    )

    pipeline_result.save_to_directory(
        f"pykeen_benchmarks\pykeen_distmultumls\{index}"
    )

    pipeline_result = pipeline(
        model="ComplEx",
        device="gpu",
        model_kwargs=dict(
            embedding_dim=128,
        ),
        training=UMLS_TRAIN_PATH,
        testing=UMLS_TEST_PATH,
        training_loop="sLCWA",
        
        training_kwargs=dict(
            batch_size=512,
        ),
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=16,
        ),
        optimizer="adam",
        optimizer_kwargs=dict(lr=0.1),
        epochs=100,
        evaluator="RankBasedEvaluator",
        # result_tracker='wandb',
        # result_tracker_kwargs=dict(
        #     project='pykeen_project',
        # ),
    )
    pipeline_result.save_to_directory(
        f"pykeen_benchmarks\pykeen_ComplEx_umls\{index}"
    )

    from pykeen.datasets.kinships import KINSHIPS_TRAIN_PATH, KINSHIPS_TEST_PATH

    pipeline_result = pipeline(
        model="Distmult",
        device="gpu",
        model_kwargs=dict(
            embedding_dim=128,
        ),
        training=KINSHIPS_TRAIN_PATH,
        testing=KINSHIPS_TEST_PATH,
        training_loop="sLCWA",
        
        training_kwargs=dict(
            batch_size=256,
        ),
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=16,
        ),
        optimizer="adam",
        optimizer_kwargs=dict(lr=0.1),
        epochs=100,
        evaluator="RankBasedEvaluator",
        # result_tracker='wandb',
        # result_tracker_kwargs=dict(
        #     project='pykeen_project',
        # ),
    )
    pipeline_result.save_to_directory(
        f"pykeen_benchmarks\pykeen_Distmult_kinships\{index}"
    )

    pipeline_result = pipeline(
        model="ComplEx",
        device="gpu",
        model_kwargs=dict(
            embedding_dim=64,
        ),
        training=KINSHIPS_TRAIN_PATH,
        testing=KINSHIPS_TEST_PATH,
        training_loop="sLCWA",
        
        training_kwargs=dict(
            batch_size=128,
        ),
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=16,
        ),
        optimizer="adam",
        optimizer_kwargs=dict(lr=0.01),
        epochs=100,
        evaluator="RankBasedEvaluator",
        # result_tracker='wandb',
        # result_tracker_kwargs=dict(
        #     project='pykeen_project',
        # ),
    )
    pipeline_result.save_to_directory(
        f"pykeen_benchmarks\pykeen_ComplEx_kinships\{index}"
    )


for i in range(5):
    run_experiment(i + 1)



