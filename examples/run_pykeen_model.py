from pykeen.datasets import UMLS
from pykeen.models import DistMult, ComplEx, QuatE
from pykeen.sampling import BasicNegativeSampler
from torch.optim import Adam
import time
from pykeen.training import LCWATrainingLoop, SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator

import pytorch_lightning as pl
import argparse


def get_default_arguments():
    """ Extends pytorch_lightning Trainer's arguments with ours """
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser(add_help=False))
    parser.add_argument("--dataset", type=str, default='UMLS')
    parser.add_argument("--model", type=str,
                        default='QuatE', help='DistMult, ComplEx, QuatE')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Number of dimensions for an embedding vector. ')
    parser.add_argument("--num_epochs", type=int, default=10, help='Number of epochs for training. ')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini batch size')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--scoring_technique', default='NegSample', help="KvsAll, NegSample")
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
    if args.model == 'DistMult':
        model = DistMult(triples_factory=dataset.training, embedding_dim=args.embedding_dim, random_seed=1,regularizer=None,entity_constrainer=None)
    elif args.model == 'ComplEx':
        model = ComplEx(triples_factory=dataset.training, embedding_dim=args.embedding_dim, random_seed=1,regularizer=None,entity_constrainer=None)
    elif args.model == 'QuatE':
        model = QuatE(triples_factory=dataset.training, embedding_dim=args.embedding_dim, random_seed=1,regularizer=None,entity_constrainer=None)
    else:
        raise NotImplementedError()

    optimizer = Adam(params=model.get_grad_params())
    if args.scoring_technique == 'KvsAll':
        # LCWA => KvsAll.
        training_loop = LCWATrainingLoop(model=model, triples_factory=dataset.training,
                                         optimizer=optimizer, optimizer_kwargs=dict(lr=args.lr))
    elif args.scoring_technique == 'NegSample':
        assert args.neg_ratio > 0
        # SLCWA => Negative Sampling
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=dataset.training,
            optimizer=optimizer,
            optimizer_kwargs=dict(lr=args.lr),
            negative_sampler=BasicNegativeSampler,
            negative_sampler_kwargs=dict(num_negs_per_pos=args.neg_ratio, ))
    else:
        raise KeyError('Invalid Argument')

    training_loop.train(triples_factory=dataset.training, num_epochs=args.num_epochs, batch_size=args.batch_size)

    report_time = time.time() - start_time
    print('Total Training Time:', report_time)
    test(model, dataset, args)


def test(model, dataset, args):
    if args.scoring_technique == 'KvsAll':
        # evaluator=LCWAEvaluationLoop(triples_factory=dataset.testing,model=model,)
        evaluator = RankBasedEvaluator()
    else:
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
            if head_or_tail_or_both == 'both':
                print(key, val)


if __name__ == '__main__':
    pretrained_model = train_eval(get_default_arguments())
