import pykeen.evaluation.rank_based_evaluator
from pykeen.datasets import UMLS, Kinships
from pykeen.models import ERModel, DistMult, Model, ComplEx
from pykeen.sampling import BasicNegativeSampler
from torch.optim import Adam
import time
import torch
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from pykeen.evaluation import LCWAEvaluationLoop, SampledRankBasedEvaluator, RankBasedEvaluator

# @ TODO: Write args to train and eval pykeen models.
def experiment_umls_distmult():
    start_time = time.time()
    dataset = UMLS()
    print(dataset)
    print('Train:', dataset.training.num_triples)
    print('Val:', dataset.validation.num_triples)
    print('Test:', dataset.testing.num_triples)
    training_triples_factory = dataset.training
    model = DistMult(triples_factory=training_triples_factory, embedding_dim=128)
    training_triples_factory = training_triples_factory
    optimizer = Adam(params=model.get_grad_params())
    # SLCWA => Negative Sampling
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training_triples_factory,
        optimizer=optimizer,
        optimizer_kwargs=dict(lr=0.01),
        negative_sampler=BasicNegativeSampler,
        negative_sampler_kwargs=dict(num_negs_per_pos=1, ), )
    training_loop.train(triples_factory=training_triples_factory, num_epochs=100, batch_size=1024)

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
        additional_filter_triples=[dataset.training.mapped_triples, dataset.validation.mapped_triples], )
    return report_time, results


def experiment_umls_complex():
    start_time = time.time()

    dataset = UMLS()
    training_triples_factory = dataset.training
    model = ComplEx(triples_factory=training_triples_factory,
                    embedding_dim=1,

                    )
    model.to(torch.device('cuda'))
    # model.to(torch.device('cpu'))
    training_triples_factory = training_triples_factory
    optimizer = Adam(params=model.get_grad_params())

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

    return report_time, results


def experiment_kinship_distmult():
    start_time = time.time()

    dataset = Kinships()
    training_triples_factory = dataset.training
    model = DistMult(triples_factory=training_triples_factory,
                     embedding_dim=64,

                     )
    model.to(torch.device('cuda'))
    # model.to(torch.device('cpu'))
    training_triples_factory = training_triples_factory
    optimizer = Adam(params=model.get_grad_params())

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

    return report_time, results


def experiment_kinship_complex():
    start_time = time.time()

    dataset = Kinships()
    training_triples_factory = dataset.training
    model = ComplEx(triples_factory=training_triples_factory,
                    embedding_dim=32,

                    )

    model.to(torch.device('cuda'))
    # model.to(torch.device('cpu'))
    training_triples_factory = training_triples_factory
    optimizer = Adam(params=model.get_grad_params())

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

    return report_time, results


def save_evaluation(results, path):
    # Pickle the object and save it to a file
    file_path = path

    # Save the dictionary as text in a file
    with open(file_path, 'w') as file:
        for key, value in results.data.items():
            file.write(f"{key}: {value}\n")


def save_report(runtime, filename):
    res = dict()
    res['report_time'] = runtime

    with open(filename + '.json', 'w') as fp:
        import json
        json.dump(res, fp)
        print('dictionary saved successfully to file')


runtime, results = experiment_umls_distmult()
# @OTOD:Report Link Prediction Results..
for key,val in results.data.items():
    if key[0] in ['hits_at_1','hits_at_3','hits_at_10']:
        hit_name, head_or_tail, optimistic_pessimistic_realistic=key
        print(key,val)
