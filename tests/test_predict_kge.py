import os
import pytest
import torch
import shutil
import numpy as np
import pandas as pd
from dicee.executer import Execute
from dicee.knowledge_graph_embeddings import KGE
from dicee.config import Namespace


class TestPredictRegression:
    """Regression tests for predict and predict_topk methods using Family dataset."""
    
    @pytest.fixture(scope="class")
    def family_model(self):
        """Setup Keci model trained on Family dataset."""
        args = Namespace()
        args.model = 'Keci'
        args.p = 0
        args.q = 1
        args.optim = 'Adam'
        args.scoring_technique = "KvsAll"
        args.path_single_kg = "KGs/Family/family-benchmark_rich_background.owl"
        args.backend = "rdflib"
        args.num_epochs = 50
        args.batch_size = 1024
        args.lr = 0.1
        args.embedding_dim = 64
        args.trainer = 'torchCPUTrainer'  # Force CPU
        
        result = Execute(args).start()
        model = KGE(path=result['path_experiment_folder'])
        
        # Ground truth relationships from the dataset
        ground_truth = {
            'hasChild': [
                ("http://www.benchmark.org/family#F9M139", "http://www.benchmark.org/family#F9F141"),
                ("http://www.benchmark.org/family#F9M139", "http://www.benchmark.org/family#F9M157"),
                ("http://www.benchmark.org/family#F9M144", "http://www.benchmark.org/family#F9F145"),
                ("http://www.benchmark.org/family#F9F154", "http://www.benchmark.org/family#F9M155"),
            ],
            'hasParent': [
                ("http://www.benchmark.org/family#F9F150", "http://www.benchmark.org/family#F9F143"),
                ("http://www.benchmark.org/family#F9F150", "http://www.benchmark.org/family#F9M144"),
                ("http://www.benchmark.org/family#F9F164", "http://www.benchmark.org/family#F9F163"),
            ],
            'married': [
                ("http://www.benchmark.org/family#F9F148", "http://www.benchmark.org/family#F9M149"),
                ("http://www.benchmark.org/family#F9F154", "http://www.benchmark.org/family#F9M153"),
                ("http://www.benchmark.org/family#F9M139", "http://www.benchmark.org/family#F9F140"),
            ]
        }

        return {
            'model': model,
            'ground_truth': ground_truth,
            'entities': list(model.entity_to_idx.keys())[:10],
            'relations': list(model.relation_to_idx.keys())[:5]
        }
    

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_predict_deterministic_behavior(self, family_model):
        """Test that predictions are deterministic across runs."""
        model = family_model['model']
        entities = family_model['entities']
        relations = family_model['relations']
        
        h, r = entities[0], relations[0]
        
        # Multiple runs should produce identical results
        scores1 = model.predict(h=h, r=r, t=None)
        scores2 = model.predict(h=h, r=r, t=None)
        scores3 = model.predict(h=h, r=r, t=None)
        
        assert torch.equal(scores1, scores2)
        assert torch.equal(scores2, scores3)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_predict_vs_predict_topk_consistency(self, family_model):
        """Test consistency between predict and predict_topk results."""
        model = family_model['model']
        entities = family_model['entities']
        relations = family_model['relations']
        
        h, r = entities[0], relations[0]
        
        # Get full scores and top-k
        full_scores = model.predict(h=h, r=r, t=None, logits=False)
        topk_results = model.predict_topk(h=h, r=r, t=None, topk=5)
        
        # Create entity->score mapping from full scores
        entity_scores = {entity: float(full_scores[i]) 
                        for i, entity in enumerate(model.entity_to_idx.keys())}
        
        # Verify top-k matches highest scores
        for entity, score in topk_results[0]:
            expected_score = entity_scores[entity]
            assert abs(score - expected_score) < 1e-6

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_batch_size_consistency(self, family_model):
        """Test that different batch sizes produce identical results."""
        model = family_model['model']
        entities = family_model['entities']
        relations = family_model['relations']
        
        h, r = entities[0], relations[0]
        topk = 10
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            results[batch_size] = model.predict_topk(
                h=h, r=r, t=None, topk=topk, batch_size=batch_size
            )
        
        # All results should be identical
        baseline = results[1]
        for batch_size in [2, 4, 8]:
            assert results[batch_size] == baseline, f"Batch size {batch_size} differs from baseline"

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_ground_truth_predictions(self, family_model):
        """Test predictions against known ground truth relationships."""
        model = family_model['model']
        ground_truth = family_model['ground_truth']
        
        # Test hasChild relationship predictions
        for parent, child in ground_truth['hasChild'][:2]:  # Test first 2
            if parent in model.entity_to_idx and child in model.entity_to_idx:
                # Predict children of parent
                results = model.predict_topk(
                    h=parent, 
                    r="http://www.benchmark.org/family#hasChild", 
                    t=None, 
                    topk=10
                )
                
                predicted_children = [entity for entity, _ in results[0]]
                assert child in predicted_children, f"Ground truth child {child} not in top predictions"
                
                # Verify score is reasonable (> 0.1 for known relationships)
                child_score = next(score for entity, score in results[0] if entity == child)
                assert child_score > 0.1, f"Score too low for known relationship: {child_score}"

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_predict_missing_types(self, family_model):
        """Test all three prediction types: missing head, tail, and relation."""
        model = family_model['model']
        entities = family_model['entities']
        relations = family_model['relations']
        
        h, r, t = entities[0], relations[0], entities[1]
        
        # Test missing tail (h, r, ?)
        tail_scores = model.predict(h=h, r=r, t=None)
        assert tail_scores.shape[0] == len(model.entity_to_idx)
        assert not torch.isnan(tail_scores).any()
        
        # Test missing head (?, r, t)  
        head_scores = model.predict(h=None, r=r, t=t)
        assert head_scores.shape[0] > 0  # Should return some scores
        assert not torch.isnan(head_scores).any()
        
        # Test missing relation (h, ?, t)
        rel_scores = model.predict(h=h, r=None, t=t)
        assert rel_scores.shape[0] > 0  # Should return some scores
        assert not torch.isnan(rel_scores).any()
        
        # Test complete triple
        triple_score = model.predict(h=h, r=r, t=t)
        assert triple_score.numel() == 1
        assert not torch.isnan(triple_score).any()
    
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_train_literals(self, family_model):
        """Test training of literal embedding model using interactive KGE model."""
        model = family_model['model']
        
        self.generate_literal_files(file_path="KGs/Family/literals")
        train_file_path="KGs/Family/literals/train.txt"

        # Train with literals
        model.train_literals(train_file_path=train_file_path,
        eval_litreal_preds=False,
        loader_backend="pandas",
        num_epochs=20,
        batch_size=50, device='cpu')

        # remove literal test artifacts
        shutil.rmtree(os.path.dirname(train_file_path))

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_predict_literals_single(self, family_model):
        """Test Literal values prediction ( single subject-predicate pair) using interactive KGE model."""
        model = family_model['model']

        self.generate_literal_files(file_path="KGs/Family/literals")
        train_file_path="KGs/Family/literals/train.txt"

        # Train with literals
        model.train_literals(train_file_path=train_file_path,
        eval_litreal_preds=False,
        loader_backend="pandas",
        num_epochs=100,
        batch_size=50, device='cpu')

        # Predict literals for a known entity
        entity = "http://www.benchmark.org/family#F4F55"
        attribute = "http://www.benchmark.org/family#Age"

        result = model.predict_literals(entity=entity, attribute=attribute)

        assert result.shape == (1,), "Expected array with shape (1,)"
        prediction = result[0]
        assert isinstance(prediction, (int, float)), "Result is not a numeric value"
        assert 30.0 <= prediction <= 35.0, f"Result {prediction} is not within the expected range"

        # remove literal test artifacts
        shutil.rmtree(os.path.dirname(train_file_path))

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_predict_literals_batch(self, family_model):
        """Test Literal values prediction(batch prediction) using interactive KGE model."""
        model = family_model['model']

        self.generate_literal_files(file_path="KGs/Family/literals")
        train_file_path="KGs/Family/literals/train.txt"

        # Train with literals
        model.train_literals(train_file_path=train_file_path,
        eval_litreal_preds=False,
        loader_backend="pandas",
        num_epochs=20,
        batch_size=512)

        # Predict literals for a known entity
        entities = [
            "http://www.benchmark.org/family#F1M1",
            "http://www.benchmark.org/family#F2F24",
            "http://www.benchmark.org/family#F4F56"
        ]

        attributes = [
            "http://www.benchmark.org/family#Height",
            "http://www.benchmark.org/family#Age",
            "http://www.benchmark.org/family#Weight"
        ]

        results = model.predict_literals(entity=entities, attribute=attributes)
        

        # Assert the result is a numpy array of the same size as the input
        assert isinstance(results, np.ndarray), "Results should be a numpy array"
        assert len(results) == len(entities), "Results size does not match input size"

        # Assert all entries are numerical types
        for result in results:
            assert isinstance(result, (int, float)), f"Result {result} is not a numeric value"
        
        # remove literal test artifacts
        shutil.rmtree(os.path.dirname(train_file_path))
        
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_evaluate_literal_prediction(self, family_model):
        """Test Evaluation of Literal Prediction using interactive KGE model."""
        model = family_model['model']
        
        self.generate_literal_files(file_path="KGs/Family/literals")
        train_file_path="KGs/Family/literals/train.txt"

        # Train with literals
        model.train_literals(train_file_path=train_file_path,
        eval_litreal_preds=False,
        loader_backend="pandas",
        num_epochs=20,
        batch_size=512)

        eval_file_path = "KGs/Family/literals/test.txt"
        
        # Evaluate literal predictions
        lit_prediction_errors = model.evaluate_literal_prediction(
            eval_file_path=eval_file_path,
            store_lit_preds = False,
            eval_literals=True,
            loader_backend = "pandas",
            return_attr_error_metrics = True)

        # Assert the result is a DataFrame
        assert isinstance(lit_prediction_errors, pd.DataFrame), "Results should be a DataFrame"
        assert not lit_prediction_errors.empty, "Results DataFrame should not be empty"
        assert not lit_prediction_errors.isnull().values.any() , "Results DataFrame should not contain NaN values"

        # remove literal test artifacts
        shutil.rmtree(os.path.dirname(eval_file_path))

    def generate_literal_files(self, file_path: str = None):
        """
        Generate training and testing files for literal triples.

        Args:
            file_path (str): The directory where the files will be saved.
        """

        literal_triples = [
        ("http://www.benchmark.org/family#F3F52", "http://www.benchmark.org/family#Age", "17"),
        ("http://www.benchmark.org/family#F6F86", "http://www.benchmark.org/family#Age", "15"),
        ("http://www.benchmark.org/family#F4F55", "http://www.benchmark.org/family#Weight", "63"),
        ("http://www.benchmark.org/family#F9M161", "http://www.benchmark.org/family#Weight", "40"),
        ("http://www.benchmark.org/family#F9F163", "http://www.benchmark.org/family#Weight", "70"),
        ("http://www.benchmark.org/family#F10M190", "http://www.benchmark.org/family#Weight", "45"),
        ("http://www.benchmark.org/family#F10M184", "http://www.benchmark.org/family#Age", "15"),
        ("http://www.benchmark.org/family#F6M73", "http://www.benchmark.org/family#Height", "158"),
        ("http://www.benchmark.org/family#F9F160", "http://www.benchmark.org/family#Weight", "61"),
        ("http://www.benchmark.org/family#F10F201", "http://www.benchmark.org/family#Height", "140"),
        ("http://www.benchmark.org/family#F4F58", "http://www.benchmark.org/family#Height", "122"),
        ("http://www.benchmark.org/family#F9F154", "http://www.benchmark.org/family#Height", "147"),
        ("http://www.benchmark.org/family#F2M16", "http://www.benchmark.org/family#Age", "43"),
        ("http://www.benchmark.org/family#F7M110", "http://www.benchmark.org/family#Age", "37"),
        ("http://www.benchmark.org/family#F1M1", "http://www.benchmark.org/family#Height", "151"),
        ("http://www.benchmark.org/family#F2F24", "http://www.benchmark.org/family#Age", "38"),
        ("http://www.benchmark.org/family#F7F106", "http://www.benchmark.org/family#Age", "43"),
        ("http://www.benchmark.org/family#F2M9", "http://www.benchmark.org/family#Height", "182"),
        ("http://www.benchmark.org/family#F9F164", "http://www.benchmark.org/family#Weight", "59"),
        ("http://www.benchmark.org/family#F10M190", "http://www.benchmark.org/family#Age", "17"),
        ("http://www.benchmark.org/family#F3M47", "http://www.benchmark.org/family#Weight", "93"),
        ("http://www.benchmark.org/family#F10M183", "http://www.benchmark.org/family#Height", "142"),
        ("http://www.benchmark.org/family#F7F119", "http://www.benchmark.org/family#Height", "112"),
        ("http://www.benchmark.org/family#F9M155", "http://www.benchmark.org/family#Age", "12"),
        ("http://www.benchmark.org/family#F2F10", "http://www.benchmark.org/family#Age", "89"),
        ("http://www.benchmark.org/family#F10M173", "http://www.benchmark.org/family#Height", "177"),
        ]
        #train test splits
        train_triples = literal_triples[:20]
        test_triples = literal_triples[20:]

        # Output directory
        output_dir = file_path or "KGs/Family/literals"
        os.makedirs(output_dir, exist_ok=True)

        train_df = pd.DataFrame(train_triples)
        test_df = pd.DataFrame(test_triples)
        train_df.to_csv( os.path.join(output_dir, "train.txt"), header=False, sep ="\t",index=False)
        test_df.to_csv( os.path.join(output_dir, "test.txt"), header=False,sep ="\t", index=False)