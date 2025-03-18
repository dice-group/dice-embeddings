import pytest
import os
import time
from dicee.evaluator import evaluate_lp_k_vs_all
from dicee.knowledge_graph import KG
from retrieval_augmented_link_predictor import GCL, RCL
import pandas as pd
import json
from datetime import datetime

class TestCompareGCLRCL:
    """
    Regression tests comparing GCL (with 3 hops) and RCL on various datasets.
    This test suite measures both model performance and runtime.
    """
    
    # Store results for all test runs to create a comparison report at the end
    results_data = []
    
    def setup_method(self):
        """Setup for each test method: check for API key"""
        # Get API key from environment variable
        self.api_key = os.environ.get("TENTRIS_TOKEN")
        assert self.api_key is not None, "TENTRIS_TOKEN environment variable not set"
        
        # Common API settings
        self.base_url = "http://harebell.cs.upb.de:8501/v1"
        self.llm_model = "tentris"
        self.temperature = 0.0
        self.seed = 42
        
        # Test settings
        self.batch_size = 1
        
        # Ensure the temp directory exists for saving results
        os.makedirs("temp", exist_ok=True)
    
    def run_model_eval(self, model_name, model, kg, test_size=None, dataset_name="Unknown"):
        """Run evaluation for a model and record performance and runtime"""
        test_triples = kg.test_set if test_size is None else kg.test_set[:test_size]
        
        # Start timer
        start_time = time.time()
        
        # Run evaluation
        results = evaluate_lp_k_vs_all(
            model=model,
            triple_idx=test_triples,
            er_vocab=kg.er_vocab,
            info=f'Testing {model_name} on {dataset_name}',
            batch_size=self.batch_size
        )
        
        # End timer
        end_time = time.time()
        runtime = end_time - start_time
        
        # Add runtime to results
        results['Runtime'] = runtime
        
        # Store results for comparison
        self.results_data.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'H@1': results['H@1'],
            'H@3': results['H@3'], 
            'H@10': results['H@10'],
            'MRR': results['MRR'],
            'Runtime (s)': runtime,
            'Test Size': len(test_triples)
        })
        
        return results
    
    def test_countries_s1(self):
        """Compare GCL and RCL on Countries-S1 dataset"""
        # Setup
        dataset_name = "Countries-S1"
        kg = KG(dataset_dir=f"KGs/{dataset_name}", separator="\s+", eval_model="train_value_test", add_reciprocal=False)
        
        # Test GCL with 3 hops
        gcl_model = GCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            num_of_hops=3
        )
        
        gcl_results = self.run_model_eval("GCL-3hops", gcl_model, kg, dataset_name=dataset_name)
        
        # Test RCL with exclude_source=True (default)
        rcl_model = RCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            max_relation_examples=50,
            exclude_source=True
        )
        
        rcl_results = self.run_model_eval("RCL-exclude", rcl_model, kg, dataset_name=dataset_name)
        
        # Test RCL with exclude_source=False
        rcl_include_model = RCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            max_relation_examples=50,
            exclude_source=False
        )
        
        rcl_include_results = self.run_model_eval("RCL-include", rcl_include_model, kg, dataset_name=dataset_name)
        
        # Check expectations
        assert gcl_results['H@1'] >= 0.9, f"GCL H@1 score too low: {gcl_results['H@1']}"
        assert rcl_results['H@1'] >= 0.9, f"RCL H@1 score too low: {rcl_results['H@1']}"
        assert rcl_include_results['H@1'] >= 0.9, f"RCL-include H@1 score too low: {rcl_include_results['H@1']}"
    
    def test_countries_s2(self):
        """Compare GCL and RCL on Countries-S2 dataset"""
        # Setup
        dataset_name = "Countries-S2"
        kg = KG(dataset_dir=f"KGs/{dataset_name}", separator="\s+", eval_model="train_value_test", add_reciprocal=False)
        
        # Test GCL with 3 hops
        gcl_model = GCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            num_of_hops=3
        )
        
        gcl_results = self.run_model_eval("GCL-3hops", gcl_model, kg, dataset_name=dataset_name)
        
        # Test RCL with exclude_source=True (default)
        rcl_model = RCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            max_relation_examples=50,
            exclude_source=True
        )
        
        rcl_results = self.run_model_eval("RCL-exclude", rcl_model, kg, dataset_name=dataset_name)
        
        # Test RCL with exclude_source=False
        rcl_include_model = RCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            max_relation_examples=50,
            exclude_source=False
        )
        
        rcl_include_results = self.run_model_eval("RCL-include", rcl_include_model, kg, dataset_name=dataset_name)
        
        # Check expectations for S2 (slightly more challenging than S1)
        assert gcl_results['H@1'] >= 0.85, f"GCL H@1 score too low: {gcl_results['H@1']}"
        assert rcl_results['H@1'] >= 0.85, f"RCL H@1 score too low: {rcl_results['H@1']}"
        assert rcl_include_results['H@1'] >= 0.85, f"RCL-include H@1 score too low: {rcl_include_results['H@1']}"
    
    def test_umls(self):
        """Compare GCL and RCL on UMLS dataset with limited test set"""
        # Setup
        dataset_name = "UMLS"
        kg = KG(dataset_dir=f"KGs/{dataset_name}", separator="\s+", eval_model="train_value_test", add_reciprocal=False)
        
        # Limit test size for UMLS to save time
        test_size = 20
        
        # Test GCL with 3 hops
        gcl_model = GCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            num_of_hops=3
        )
        
        gcl_results = self.run_model_eval("GCL-3hops", gcl_model, kg, test_size=test_size, dataset_name=dataset_name)
        
        # Test RCL with exclude_source=True (default)
        rcl_model = RCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            max_relation_examples=50,
            exclude_source=True
        )
        
        rcl_results = self.run_model_eval("RCL-exclude", rcl_model, kg, test_size=test_size, dataset_name=dataset_name)
        
        # Test RCL with exclude_source=False
        rcl_include_model = RCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            max_relation_examples=50,
            exclude_source=False
        )
        
        rcl_include_results = self.run_model_eval("RCL-include", rcl_include_model, kg, test_size=test_size, dataset_name=dataset_name)
        
        # Check expectations for UMLS
        assert gcl_results['H@1'] >= 0.45, f"GCL H@1 score too low: {gcl_results['H@1']}"
        assert rcl_results['H@1'] >= 0.45, f"RCL H@1 score too low: {rcl_results['H@1']}"
        assert rcl_include_results['H@1'] >= 0.45, f"RCL-include H@1 score too low: {rcl_include_results['H@1']}"
    
    def test_kinship(self):
        """Compare GCL and RCL on KINSHIP dataset with limited test set"""
        # Setup
        dataset_name = "KINSHIP"
        kg = KG(dataset_dir=f"KGs/{dataset_name}", separator="\s+", eval_model="train_value_test", add_reciprocal=False)
        
        # Limit test size for KINSHIP to save time
        test_size = 20
        
        # Test GCL with 3 hops
        gcl_model = GCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            num_of_hops=3
        )
        
        gcl_results = self.run_model_eval("GCL-3hops", gcl_model, kg, test_size=test_size, dataset_name=dataset_name)
        
        # Test RCL with exclude_source=True (default)
        rcl_model = RCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            max_relation_examples=50,
            exclude_source=True
        )
        
        rcl_results = self.run_model_eval("RCL-exclude", rcl_model, kg, test_size=test_size, dataset_name=dataset_name)
        
        # Test RCL with exclude_source=False
        rcl_include_model = RCL(
            knowledge_graph=kg,
            base_url=self.base_url,
            api_key=self.api_key,
            llm_model=self.llm_model,
            temperature=self.temperature,
            seed=self.seed,
            max_relation_examples=50,
            exclude_source=False
        )
        
        rcl_include_results = self.run_model_eval("RCL-include", rcl_include_model, kg, test_size=test_size, dataset_name=dataset_name)
        
        # Check expectations for KINSHIP
        assert gcl_results['H@1'] >= 0.05, f"GCL H@1 score too low: {gcl_results['H@1']}"
        assert rcl_results['H@1'] >= 0.05, f"RCL H@1 score too low: {rcl_results['H@1']}"
        assert rcl_include_results['H@1'] >= 0.05, f"RCL-include H@1 score too low: {rcl_include_results['H@1']}"
    
    def teardown_method(self):
        """After each test method, print current results"""
        if len(self.results_data) % 3 == 0:  # Print after each dataset's tests complete
            current_df = pd.DataFrame(self.results_data[-3:])
            print(f"\nLatest results:")
            print(current_df.to_string(index=False))
    
    @pytest.fixture(scope="session", autouse=True)
    def save_comparison_results(self, request):
        """Save all results at the end of the test session"""
        def finalize():
            if TestCompareGCLRCL.results_data:
                # Create a timestamp for the results file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_df = pd.DataFrame(TestCompareGCLRCL.results_data)
                
                # Save to CSV in temp directory
                output_file = f"temp/compare_gcl_rcl_results_{timestamp}.csv"
                results_df.to_csv(output_file, index=False)
                print(f"\nComplete comparison results saved to {output_file}")
                
                # Also save as JSON for easier programmatic access
                json_file = f"temp/compare_gcl_rcl_results_{timestamp}.json"
                results_df.to_json(json_file, orient="records", indent=2)
                
                # Print the final comparison table
                print("\nFinal Comparison Results:")
                print(results_df.to_string(index=False))
                
                # Create summary by dataset/model
                summary = results_df.groupby(['Dataset', 'Model']).mean().reset_index()
                print("\nAverage Metrics by Dataset and Model:")
                print(summary.to_string(index=False))
        
        request.addfinalizer(finalize) 