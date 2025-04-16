import os
import json
import pytest
import torch
import tempfile
import shutil
from retrieval_aug_predictors.models import KG
from retrieval_aug_predictors.models.demir_ensemble_mipro import RALP_MPRO
from dicee.evaluator import evaluate_lp_k_vs_all

class TestRALP_MPRORegression:
    @classmethod
    def setup_class(cls):
        # Create a temporary directory for model outputs
        cls.temp_dir = tempfile.mkdtemp()
        
        # Configure model parameters
        cls.llm_model = "tentris"  
        cls.api_key = os.getenv("TENTRIS_TOKEN")
        cls.base_url = os.getenv("OPENAI_API_BASE", "http://harebell.cs.upb.de:8501/v1")
        cls.temperature = 0.0
        cls.seed = 42
        
        # Define expected benchmark results from the comment at the top of the demir file
        cls.expected_results = {
            "Countries-S1": {
                "H@1": 0.75,
                "H@3": 0.875,
                "H@10": 1.0,
                "MRR": 0.8416666666666667
            },
            "Countries-S2": {
                "H@1": 0.75,
                "H@3": 1.0,
                "H@10": 1.0,
                "MRR": 0.8680555555555555
            },
            "Countries-S3": {
                "H@1": 0.041666666666666664,
                "H@3": 0.4583333333333333,
                "H@10": 0.625,
                "MRR": 0.2626660300405415
            }
        }
        
        # Dataset directories
        cls.dataset_dirs = {
            "Countries-S1": "KGs/Countries-S1",
            "Countries-S2": "KGs/Countries-S2", 
            "Countries-S3": "KGs/Countries-S3"
        }

    @classmethod
    def teardown_class(cls):
        # Clean up temporary directory
        shutil.rmtree(cls.temp_dir)

    @pytest.mark.parametrize("dataset_name", ["Countries-S1", "Countries-S2", "Countries-S3"])
    def test_model_performance(self, dataset_name):
        """Test model performance against benchmarks for each dataset."""
        dataset_dir = self.dataset_dirs[dataset_name]
        expected_metrics = self.expected_results[dataset_name]
        
        # Create a dataset-specific save directory
        save_dir = os.path.join(self.temp_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        kg = KG(dataset_dir=dataset_dir, separator="\s+", eval_model="KvsAll", add_reciprocal=False)
        
        model = RALP_MPRO(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            use_val=True,
            ensemble_temperatures=[0.0],  # Use a single temperature for faster testing
            save_dir=save_dir,
        )
        
        # Use the full test set to match the original experiments
        test_triples = kg.test_set
        
        # Run evaluation
        results = evaluate_lp_k_vs_all(
            model=model,
            triple_idx=test_triples,
            er_vocab=kg.er_vocab,
            info=f'Regression Test (RALP_MPRO) - {dataset_name}'
        )
        
        # Save test results for inspection
        results_file = os.path.join(save_dir, f"test_results_{dataset_name}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults for {dataset_name}:")
        print(json.dumps(results, indent=2))
        print(f"Expected results:")
        print(json.dumps(expected_metrics, indent=2))
        
        # Check that results have the expected metrics
        assert set(results.keys()) == set(expected_metrics.keys())
        
        # For regression testing, verify that results are at least as good as the benchmarks
        # No tolerance - must be at least as good or better
        for metric in expected_metrics:
            # For "H@" metrics and MRR, higher is better
            if metric.startswith("H@") or metric == "MRR":
                assert results[metric] >= expected_metrics[metric], \
                    f"Performance regression in {dataset_name} - {metric}: " \
                    f"got {results[metric]}, expected at least {expected_metrics[metric]}"

   