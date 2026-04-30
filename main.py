from dicee.config import Namespace
from dicee.executer import TabularExecute
from dicee.executer import Execute
from multiprocessing import freeze_support
from tabpfn import TabPFNClassifier
import numpy as np
import convert_funcs as cf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    dataset_dir = "dice-embeddings/KGs/UMLS"

    """ args = Namespace()
    args.model = 'TransE'
    args.scoring_technique = "KvsAll"
    args.dataset_dir = dataset_dir
    args.path_to_store_single_run = "dice-embeddings/Experiments/TransE_UMLS"
    args.num_epochs = 100
    args.embedding_dim = 32
    args.batch_size = 1024
    reports = Execute(args).start()

    print(reports["Train"]["MRR"]) # => 0.9912
    print(reports["Test"]["MRR"]) # => 0.8155 """

    # args.embedding_dim = 16
    # args.path_to_store_single_run = "dice-embeddings/Experiments/TransE_UMLS_16"
    # reports_16 = Execute(args).start()

    # args.embedding_dim = 8
    # args.path_to_store_single_run = "dice-embeddings/Experiments/TransE_UMLS_8"
    # reports_8 = Execute(args).start()

    model_path = "dice-embeddings/Experiments/TransE_UMLS"
    model_path_16 = "dice-embeddings/Experiments/TransE_UMLS_16"
    model_path_8 = "dice-embeddings/Experiments/TransE_UMLS_8"

    args = Namespace()
    args.tabpfn = True
    args.dataset_dir = dataset_dir
    args.path_to_store_single_run = "dice-embeddings/Experiments/TabPFN_UMLS"
    args.random_seed = 1
    args.num_core = 1
    args.neg_ratio = 1
    args.separator = r"\s+"
    args.tabpfn_kwargs = {
        "max_train_samples": 800,
        "device": "cpu",
        "n_estimators": 8,
        "entity_centric": True,
    }

    # print("Baseline")
    # result = TabularExecute(args).start()

    train_triples = cf.kg_to_array(dataset_dir, "train")
    test_triples = cf.kg_to_array(dataset_dir, "test")
    valid_triples = cf.kg_to_array(dataset_dir, "valid")

    # Build matrix on train only
    matrix = cf.build_entity_relation_matrix(train_triples)

    # Generate samples for each split
    train_samples = cf.generate_samples(train_triples, matrix, neg_ratio=args.neg_ratio, seed=args.random_seed)
    test_samples = cf.generate_samples(test_triples, matrix, neg_ratio=args.neg_ratio, seed=args.random_seed)
    valid_samples = cf.generate_samples(valid_triples, matrix, neg_ratio=args.neg_ratio, seed=args.random_seed)

    # Split into X and y
    X_train = train_samples.drop(columns=["label"]).values.astype(np.float32)[:800]
    y_train = train_samples["label"].values[:800]

    X_valid = valid_samples.drop(columns=["label"]).values.astype(np.float32)
    y_valid = valid_samples["label"].values

    X_test = test_samples.drop(columns=["label"]).values.astype(np.float32)
    y_test = test_samples["label"].values

    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)

    #clf.fit(X_train, y_train)

    # print("Representing the KG as binary flags:")
    # cf.evaluate(clf, X_valid, y_valid, "Valid set")
    # cf.evaluate(clf, X_test, y_test, "Test set")

    #Using embeddings with TabPFN

    # Load embeddings
    entity_embeddings, relation_embeddings, entity_to_idx, relation_to_idx = cf.load_embeddings(model_path)
    n_estimators = 1
    triple_batch_size = 50  # Adjust based on memory constraints
    max_triples = None  # Set to None to evaluate on the full test set

    # Generate samples
    X_train, y_train = cf.generate_embedding_samples(train_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)
    X_valid, y_valid = cf.generate_embedding_samples(valid_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)
    X_test, y_test = cf.generate_embedding_samples(test_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)

    # print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    clf_emb = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, n_estimators=n_estimators)
    clf_emb.fit(X_train[:800], y_train[:800])
    

    print(f"n_estimators: {n_estimators}; triple_batch_size: {triple_batch_size}; max_triples: {max_triples}")
    print("Representing the KG using TransE embeddings using 32 dimensions:")
    # cf.evaluate(clf_emb, X_valid, y_valid, "Valid set")
    # cf.evaluate(clf_emb, X_test, y_test, "Test set")
    """ cf.evaluate_link_prediction_tabpfn(
        dataset_dir=dataset_dir,
        experiment_dir=model_path,
        clf=clf_emb,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
    ) """

    er_vocab = cf.build_er_vocab(train_triples, valid_triples, test_triples)
    re_vocab = cf.build_re_vocab(train_triples, valid_triples, test_triples)


    metrics = cf.evaluate_link_prediction_tabpfn_batched(
        clf=clf_emb,
        test_triples=test_triples,   
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        er_vocab=er_vocab,
        re_vocab=re_vocab,
        k_values=[1, 3, 10],
        max_triples=None,             
        triple_batch_size=1,       
    )

    entity_embeddings, relation_embeddings, entity_to_idx, relation_to_idx = cf.load_embeddings(model_path_16)

    # Generate samples
    X_train, y_train = cf.generate_embedding_samples(train_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)
    X_valid, y_valid = cf.generate_embedding_samples(valid_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)
    X_test, y_test = cf.generate_embedding_samples(test_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)

    # print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    clf_emb_16 = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, n_estimators=n_estimators)
    clf_emb_16.fit(X_train[:800], y_train[:800])

    print("Representing the KG using TransE embeddings using 16 dimensions:")
    # cf.evaluate(clf_emb_16, X_valid, y_valid, "Valid set")
    # cf.evaluate(clf_emb_16, X_test, y_test, "Test set")
    """ cf.evaluate_link_prediction_tabpfn(
        dataset_dir=dataset_dir,
        experiment_dir=model_path_16,
        clf=clf_emb_16,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
    ) """

    metrics = cf.evaluate_link_prediction_tabpfn_batched(
        clf=clf_emb_16,
        test_triples=test_triples,   
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        er_vocab=er_vocab,
        re_vocab=re_vocab, 
        k_values=[1, 3, 10],
        max_triples=max_triples,             
        triple_batch_size=triple_batch_size,       
    )

    entity_embeddings, relation_embeddings, entity_to_idx, relation_to_idx = cf.load_embeddings(model_path_8)

    # Generate samples
    X_train, y_train = cf.generate_embedding_samples(train_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)
    X_valid, y_valid = cf.generate_embedding_samples(valid_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)
    X_test, y_test = cf.generate_embedding_samples(test_triples, entity_to_idx, relation_to_idx, entity_embeddings, relation_embeddings, neg_ratio=args.neg_ratio, seed=args.random_seed)

    # print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    clf_emb_8 = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, n_estimators=n_estimators)
    clf_emb_8.fit(X_train[:800], y_train[:800])

    print("Representing the KG using TransE embeddings using 8 dimensions:")
    # cf.evaluate(clf_emb_8, X_valid, y_valid, "Valid set")
    # cf.evaluate(clf_emb_8, X_test, y_test, "Test set")

    """ cf.evaluate_link_prediction_tabpfn(
        dataset_dir=dataset_dir,
        experiment_dir=model_path_8,
        clf=clf_emb_8,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
    ) """

    metrics = cf.evaluate_link_prediction_tabpfn_batched(
        clf=clf_emb_8,
        test_triples=test_triples,   
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        er_vocab=er_vocab,
        re_vocab=re_vocab,
        k_values=[1, 3, 10],
        max_triples=max_triples,             
        triple_batch_size=triple_batch_size, 
    )


if __name__ == "__main__":
    freeze_support()
    main()