import argparse
import pytest
import os
from dicee.evaluator import evaluate_lp_k_vs_all
from dicee.knowledge_graph import KG
from retrieval_augmented_link_predictor import GCL

class TestRegressionGCL:
    """Regression tests for the GCL (Graph Context Learning) model"""
    #@pytest.mark.filterwarnings('ignore::UserWarning')
    def test_countries_s1_hop3(self):
        """Test GCL on Countries-S1 dataset with 3 hops"""
        # Setup
        kg = KG(dataset_dir="KGs/Countries-S1", separator="\s+", eval_model="train_value_test", add_reciprocal=False)
        
        # Get API key from environment variable
        api_key = os.environ.get("TENTRIS_TOKEN")
        assert api_key is not None, "TENTRIS_TOKEN environment variable not set"
        
        # Initialize model
        model = GCL(
            knowledge_graph=kg,
            base_url="http://harebell.cs.upb.de:8501/v1",
            api_key=api_key,
            llm_model="tentris",
            temperature=0.0,
            seed=42,
            num_of_hops=3
        )
        
        # Run evaluation
        results = evaluate_lp_k_vs_all(
            model=model,
            triple_idx=kg.test_set,
            er_vocab=kg.er_vocab,
            info='Testing Countries-S1 with 3 hops',
            batch_size=1
        )
        
        # Check results - we expect perfect or near-perfect scores with 3 hops
        assert results['H@1'] >= 1.0, f"H@1 score too low: {results['H@1']}"
        assert results['H@3'] >= 1.0, f"H@3 score too low: {results['H@3']}"
        assert results['H@10'] >= 1.0, f"H@10 score too low: {results['H@10']}"
        assert results['MRR'] >= 1.0, f"MRR score too low: {results['MRR']}"

    def test_umls_hop3(self):
        """Test GCL on UMLS dataset with 2 hops"""
        # Setup
        kg = KG(dataset_dir="KGs/UMLS", separator="\s+", eval_model="train_value_test", add_reciprocal=False)
        
        # Get API key from environment variable
        api_key = os.environ.get("TENTRIS_TOKEN")
        assert api_key is not None, "TENTRIS_TOKEN environment variable not set"
        
        # Initialize model
        model = GCL(
            knowledge_graph=kg,
            base_url="http://harebell.cs.upb.de:8501/v1",
            api_key=api_key,
            llm_model="tentris",
            temperature=0.0,
            seed=42,
            num_of_hops=3
        )
        
        # Run evaluation
        results = evaluate_lp_k_vs_all(
            model=model,
			triple_idx=kg.test_set[:24],
            er_vocab=kg.er_vocab,
            info='Testing UMLS with 2 hops',
            batch_size=1
        )
        
        # Check results - based on typical performance for UMLS
        assert results['H@1'] >= 0.1, f"H@1 score too low: {results['H@1']}"
        assert results['H@3'] >= 0.1, f"H@3 score too low: {results['H@3']}"
        assert results['H@10'] >= 0.1, f"H@10 score too low: {results['H@10']}"
        assert results['MRR'] >= 0.1, f"MRR score too low: {results['MRR']}"

    def test_kinship_hop3(self):
        """Test GCL on KINSHIP dataset with 3 hops"""
        # Setup
        kg = KG(dataset_dir="KGs/KINSHIP", separator="\s+", eval_model="train_value_test", add_reciprocal=False)
        
        # Get API key from environment variable
        api_key = os.environ.get("TENTRIS_TOKEN")
        assert api_key is not None, "TENTRIS_TOKEN environment variable not set"
        
        # Initialize model
        model = GCL(
            knowledge_graph=kg,
            base_url="http://harebell.cs.upb.de:8501/v1",
            api_key=api_key,
            llm_model="tentris",
            temperature=0.0,
            seed=42,
            num_of_hops=3
        )
        
        # Run evaluation
        results = evaluate_lp_k_vs_all(
            model=model,
            triple_idx=kg.test_set[:24],
            er_vocab=kg.er_vocab,
            info='Testing KINSHIP with 2 hops',
            batch_size=1
        )
        
        # Check results - based on typical performance for KINSHIP
        assert results['H@1'] >= 0.1, f"H@1 score too low: {results['H@1']}"
        assert results['H@3'] >= 0.08, f"H@3 score too low: {results['H@3']}"
        assert results['H@10'] >= 0.08, f"H@10 score too low: {results['H@10']}"
        assert results['MRR'] >= 0.1, f"MRR score too low: {results['MRR']}" 