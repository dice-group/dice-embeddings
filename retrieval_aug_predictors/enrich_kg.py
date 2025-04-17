import os
import subprocess
import json
import argparse
import shutil


def run(args):

    if args.pred_out is None:
        if "/" in args.dataset_dir:
            args.pred_out = "predictions_" + args.dataset_dir.split("/")[-1] + ".json"
        else:
            args.pred_out = "predictions_" + args.dataset_dir + ".json"


    if not os.path.exists(args.pred_out):
        subprocess.run(["python", "models/RALP.py",
                        "--dataset_dir", args.dataset_dir,
                        "--print_top_predictions",
                        "--out", args.pred_out])

    with open(args.pred_out) as f:
        predictions = json.load(f)

    with open(args.dataset_dir + "/train.txt", "r") as f:
        triples = f.readlines()

    for prd in predictions:
        triple = prd[0] + "\t" + prd[1] + " \t" + prd[2][0][0]+ "\n"
        if triple not in triples:
            triples.append(triple)

    if args.kg_out is None:
        args.kg_out = os.path.dirname(args.dataset_dir) + "/Enriched_" + args.dataset_dir.split("/")[-1]

    os.makedirs(args.kg_out, exist_ok=True)

    with open(args.kg_out + "/train.txt", "w") as out:
        out.writelines(triples)

    shutil.copy(args.dataset_dir + "/valid.txt", args.kg_out + "/valid.txt")
    shutil.copy(args.dataset_dir + "/test.txt", args.kg_out + "/test.txt")


if __name__ == "__main__":
    """If run on default arguments (with exception of `dataset_dir` which has to be set), it will create prediction 
    using RALP, enrich the train set of the KG with these predictions and save it to a directory named 
    "Enriched_<dataset_name>". Test and validation splits are copied as they are from the original dataset.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str,
                        default="/home/alkid/PycharmProjects/dice-embeddings/KGs/Countries-S3",
                        help="Path to the dataset.")
    parser.add_argument("--pred_out", type=str, default=None,
                        help="Name of the output file where the predictions will be saved.")

    parser.add_argument("--kg_out", type=str, default=None,
                        help="Name of the output file where the extended train set will be saved.")

    run(parser.parse_args())
