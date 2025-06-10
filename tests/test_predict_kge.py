import pytest
import torch
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