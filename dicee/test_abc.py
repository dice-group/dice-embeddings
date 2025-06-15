from dicee.executer import Execute
from dicee.config import Namespace
import multiprocessing
import itertools
import torch

def run_model(learning_rate, dataset_path, embed_dim, batch_size, model):
    args = Namespace()
    args.byte_pair_encoding = True

    args.use_custom_tokenizer = True
    args.use_transformer = True
    args.tokenizer_path = (
        "C:/Users/Harshit Purohit/Tokenizer/tokenizer.json"
    )
    # args.tokenizer_path = None
    
    args.padding = False
    args.path_to_store_single_run = "BytE_UMLS"

    # Grid Search Parameters
    args.model = model                            
    args.dataset_dir = dataset_path
    args.embedding_dim = embed_dim
    args.lr = learning_rate
    args.batch_size = batch_size


    # Static parameters
    args.num_epochs = 10
    args.scoring_technique = "KvsAll"

    reports = Execute(args).start()
    print("Train MRR:", reports["Train"]["MRR"])
    print("Test  MRR:", reports["Test"]["MRR"])

    with open("demofile.txt", "a") as f:
        f.write(f"Train MRR:, {reports['Train']['MRR']}\n")
        f.write(f"Test  MRR:, {reports['Test']['MRR']}\n")
        f.write(str(reports) + "\n\n")

    return reports


if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows compatibility
    
    dataset_paths = ["KGs/Countries-S1"]
    # dataset_paths = ["KGs/UMLS", "KGs/Countries-S1", "KGs/Countries-S2", "KGs/Countries-S3"]

    models = ["DistMult"]
    # models = ["DistMult", "ComplEx", "QMult", "Keci"]

    batch_sizes = [1024]
    # batch_sizes = [512, 1024]
    learning_rates = [0.1]
    # learning_rates = [0.1, 0.01, 0.011]
    embed_dims = [64]
    # embed_dims = [32, 64]

    results = []
    with open("grid_search_results.txt", "w") as log:
        log.write("lr,dataset,embed_dim,train_mrr,test_mrr\n")
        for ds, lr, dim, bs, md in itertools.product(dataset_paths, learning_rates, embed_dims, batch_sizes, models):
            print(f"Running: dataset={ds}, lr={lr}, embed_dim={dim}, batch_size={bs}, model={md}")
            reports = run_model(lr, ds, dim, bs, md)
            tr = reports["Train"]["MRR"]
            te = reports["Test"]["MRR"]
            log.write(f"{lr},{ds},{dim},{bs},{md},{tr},{te}\n")
            results.append({"lr":lr, "dataset":ds, "embed_dim":dim, "batch_size":bs, "model":md,
                            "train_mrr":tr, "test_mrr":te})
            
    # best = max(results, key=lambda x: x["test_mrr"])
    # print("Best result:", best)

    best_per_group = {}
    for r in results:
        key = (r["model"], r["dataset"])
        if key not in best_per_group or r["test_mrr"] > best_per_group[key]["test_mrr"]:
            best_per_group[key] = r

    print("Best results per (model, dataset):")
    for (model, dataset), info in best_per_group.items():
        print(f"{model} @ {dataset}  →  "
                f"lr={info['lr']}, embed_dim={info['embed_dim']}, batch_size={info['batch_size']}, "
                f"train_mrr={info['train_mrr']}, test_mrr={info['test_mrr']}")
        
    

    # with open("grid_search_results.txt", "a") as log:
    #     log.write("\n# BEST\n")
    #     for k, v in best.items():
    #         log.write(f"{k}: {v}\n")

    with open("grid_search_results.txt", "a") as log:
        log.write("\n# BEST PER MODEL/DATASET\n")
        log.write("model,dataset,lr,embed_dim,batch_size,train_mrr,test_mrr\n")
        for (model, dataset), info in best_per_group.items():
            log.write(f"{model},{dataset},{info['lr']},{info['embed_dim']},{info['batch_size']},"
                        f"{info['train_mrr']},{info['test_mrr']}\n")