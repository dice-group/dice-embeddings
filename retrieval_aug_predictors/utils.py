import os
def sanity_checking(args,kg):
    if args.eval_size is not None:
        assert len(kg.test_set) >= args.eval_size, (f"Evaluation size cant be greater than the "
                                                    f"total amount of triples in the test set: {len(kg.test_set)}")
    else:
        args.eval_size = len(kg.test_set)
    if args.api_key is None:
        args.api_key = os.environ.get("TENTRIS_TOKEN")