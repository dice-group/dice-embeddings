import pykeen.evaluation.rank_based_evaluator
from pykeen.datasets import UMLS, Kinships
from pykeen.models import ERModel, DistMult, Model, ComplEx, QuatE
from pykeen.sampling import BasicNegativeSampler
from torch.optim import Adam
import time
import torch
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from pykeen.evaluation import LCWAEvaluationLoop, SampledRankBasedEvaluator, RankBasedEvaluator

from dicee.executer import Execute
import pytorch_lightning as pl
from dicee.config import ParseDict
import argparse


def get_default_arguments():
    """ Extends pytorch_lightning Trainer's arguments with ours """
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    # Default Trainer param https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#methods
    parser.add_argument("--dataset", type=str, default='UMLS')
    parser.add_argument("--model", type=str,
                        default='ComplEx', help='DistMult, ComplEx')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Number of dimensions for an embedding vector. ')
    parser.add_argument("--num_epochs", type=int, default=100, help='Number of epochs for training. ')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini batch size')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--scoring_technique', default='KvsAll', help="KvsAll, NegSample")
    parser.add_argument('--neg_ratio', type=int, default=1,
                        help='The number of negative triples generated per positive triple.')
    return parser.parse_args()


def train_eval(args):
    start_time = time.time()
    if args.dataset == 'UMLS':
        dataset = UMLS()
    else:
        raise NotImplementedError()

    print(dataset)
    print('Train:', dataset.training.num_triples)
    print('Val:', dataset.validation.num_triples)
    print('Test:', dataset.testing.num_triples)
    training_triples_factory = dataset.training
    if args.model == 'DistMult':
        model = DistMult(triples_factory=training_triples_factory, embedding_dim=args.embedding_dim)
    elif args.model == 'ComplEx':
        model = ComplEx(triples_factory=training_triples_factory, embedding_dim=args.embedding_dim)
    else:
        raise NotImplementedError()

    optimizer = Adam(params=model.get_grad_params())
    if args.scoring_technique == 'KvsAll':
        # LCWA => KvsAll.
        training_loop = LCWATrainingLoop(model=model, triples_factory=training_triples_factory,
                                         optimizer=optimizer, optimizer_kwargs=dict(lr=args.lr))
    elif args.scoring_technique == 'NegSample':
        assert args.neg_ratio > 0
        # SLCWA => Negative Sampling
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=training_triples_factory,
            optimizer=optimizer,
            optimizer_kwargs=dict(lr=args.lr),
            negative_sampler=BasicNegativeSampler,
            negative_sampler_kwargs=dict(num_negs_per_pos=args.neg_ratio, ))
    else:
        raise KeyError('Invalid Argument')

    training_loop.train(triples_factory=training_triples_factory, num_epochs=args.num_epochs,
                        batch_size=args.batch_size)
    report_time = time.time() - start_time
    print('Total Training Time:', report_time)
    test(model, dataset, args)


def test(model, dataset, args):
    evaluator = RankBasedEvaluator()
    # Get triples to test
    mapped_triples = dataset.testing.mapped_triples
    # Evaluate
    results = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        batch_size=args.batch_size,
        additional_filter_triples=[dataset.training.mapped_triples, dataset.validation.mapped_triples], )

    for key, val in results.data.items():
        if key[0] in ['hits_at_1', 'hits_at_3', 'hits_at_10']:
            hit_name, head_or_tail_or_both, optimistic_pessimistic_realistic = key
            if optimistic_pessimistic_realistic == 'pessimistic' and head_or_tail_or_both == 'both':
                print(key, val)
if __name__ == '__main__':
    pretrained_model = train_eval(get_default_arguments())