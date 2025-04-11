import os
import subprocess
import json
import argparse

def run(args):

    if args.pred_out is None:
        if "/" in args.dataset_dir:
            args.pred_out = "predictions_" + args.dataset_dir.split("/")[-1] + ".json"
        else:
            args.pred_out = "predictions_" + args.dataset_dir + ".json"


    if not os.path.exists(args.pred_out):
        subprocess.run(["python", "models/demir_ensemble.py",
                        "--dataset_dir", args.dataset_dir,
                        "--print_top_predictions",
                        "--out", args.pred_out])

    with open(args.pred_out) as f:
        predictions = json.load(f)

    with open(args.dataset_dir + "/train.txt", "r") as f:
        triples = f.readlines()

    for prd in predictions:
        triple = prd[0] + "\t" + prd[1] + " \t" + prd[2][0][0]
        triples.append(triple + "\n")

    with open(args.dataset_dir + "/" + args.kg_out, "w") as out:
        out.writelines(triples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str,
                        default="/home/alkid/PycharmProjects/dice-embeddings/KGs/Countries-S3",
                        help="Path to the dataset.")
    parser.add_argument("--pred_out", type=str, default=None,
                        help="Name of the output file where the predictions will be saved.")

    parser.add_argument("--kg_out", type=str, default="extended_train.txt",
                        help="Name of the output file where the extended train set will be saved.")

    run(parser.parse_args())
