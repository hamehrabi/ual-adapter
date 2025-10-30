"""
Tests for LoRA Dispatcher module
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from ual_adapter.core.dispatcher import LoRADispatcher, DomainAdapter


class TestLoRADispatcher:
    """Test LoRA dispatcher functionality."""
    
    def test_dispatcher_initialization(self):
        """Test dispatcher initialization."""
        dispatcher = LoRADispatcher(
            encoder_model="all-MiniLM-L6-v2",
            confidence_threshold=0.7
        )
        
        assert dispatcher.confidence_threshold == 0.7
        assert len(dispatcher.domains) == 0
        assert dispatcher.router is None
    
    def test_register_single_domain(self, sample_lora_weights):
        """Test registering a single domain."""
        dispatcher = LoRADispatcher()
        
        training_texts = [
            "Medical diagnosis and treatment",
            "Patient symptoms and care",
            "Clinical procedures"
        ]
        
        dispatcher.register_domain(
            domain_name="medical",
            adapter_weights=sample_lora_weights,
            training_texts=training_texts,
            metadata={"description": "Medical domain adapter"}
        )
        
        assert "medical" in dispatcher.domains
        assert "medical" in dispatcher.domain_embeddings
        assert dispatcher.domains["medical"].name == "medical"
    
    def test_register_multiple_domains(self, sample_lora_weights):
        """Test registering multiple domains."""
        dispatcher = LoRADispatcher()
        
        domains_data = [
            ("medical", ["Medical text 1", "Patient care", "Clinical data"]),
            ("legal", ["Legal document", "Contract terms", "Court ruling"]),
            ("technical", ["Software code", "API documentation", "Bug report"])
        ]
        
        for domain_name, texts in domains_data:
            dispatcher.register_domain(
                domain_name=domain_name,
                adapter_weights=sample_lora_weights,
                training_texts=texts
            )
        
        assert len(dispatcher.domains) == 3
        assert dispatcher.router is not None  # Should be trained after 2+ domains
    
    def test_router_training(self, sample_lora_weights):
        """Test that router is trained with multiple domains."""
        dispatcher = LoRADispatcher()
        
        # Register first domain - no router yet
        dispatcher.register_domain(
            "domain1",
            sample_lora_weights,
            ["Text 1", "Text 2"]
        )
        assert dispatcher.router is None
        
        # Register second domain - router should be trained
        dispatcher.register_domain(
            "domain2",
            sample_lora_weights,
            ["Other text 1", "Other text 2"]
        )
        assert dispatcher.router is not None
    
    def test_route_query_single_domain(self, sample_lora_weights):
        """Test routing with a single domain."""
        dispatcher = LoRADispatcher()
        
        dispatcher.register_domain(
            "medical",
            sample_lora_weights,
            ["Medical diagnosis", "Patient symptoms"]
        )
        
        domain, confidence, scores = dispatcher.route_query(
            "Patient treatment plan"
        )
        
        assert domain == "medical"
        assert confidence == 1.0  # Only one domain
        assert scores is None
    
    def test_route_query_multiple_domains(self, sample_lora_weights):
        """Test routing with multiple domains."""
        dispatcher = LoRADispatcher(confidence_threshold=0.6)
        
        # Register domains with distinct texts
        dispatcher.register_domain(
            "medical",
            sample_lora_weights,
            [
                "Patient diagnosis and treatment",
                "Medical symptoms and care",
                "Clinical procedures and tests",
                "Healthcare and medicine"
            ]
        )
        
        dispatcher.register_domain(
            "legal",
            sample_lora_weights,
            [
                "Legal contracts and agreements",
                "Court proceedings and law",
                "Legal documentation and compliance",
                "Judicial system and regulations"
            ]
        )
        
        # Test medical query
        domain, confidence, _ = dispatcher.route_query(
            "Patient medical diagnosis"
        )
        assert domain == "medical"
        assert confidence > 0.6
        
        # Test legal query
        domain, confidence, _ = dispatcher.route_query(
            "Legal contract review"
        )
        assert domain == "legal"
        assert confidence > 0.6
    
    def test_confidence_threshold(self, sample_lora_weights):
        """Test confidence threshold behavior."""
        dispatcher = LoRADispatcher(confidence_threshold=0.9)  # High threshold
        
        dispatcher.register_domain(
            "domain1",
            sample_lora_weights,
            ["Specific text 1"]
        )
        
        dispatcher.register_domain(
            "domain2",
            sample_lora_weights,
            ["Different text 2"]
        )
        
        # Query that might not strongly match either domain
        domain, confidence, _ = dispatcher.route_query(
            "Completely unrelated query about something else"
        )
        
        # With high threshold, might return None
        if confidence < 0.9:
            assert domain is None
    
    def test_return_all_scores(self, sample_lora_weights):
        """Test returning scores for all domains."""
        dispatcher = LoRADispatcher()
        
        domains = ["medical", "legal", "technical"]
        for domain in domains:
            dispatcher.register_domain(
                domain,
                sample_lora_weights,
                [f"{domain} text 1", f"{domain} text 2"]
            )
        
        domain, confidence, all_scores = dispatcher.route_query(
            "Medical patient care",
            return_all_scores=True
        )
        
        assert all_scores is not None
        assert len(all_scores) == 3
        assert all(0 <= score <= 1 for score in all_scores.values())
        assert sum(all_scores.values()) == pytest.approx(1.0, rel=1e-5)
    
    def test_domain_overlap_analysis(self, sample_lora_weights):
        """Test domain overlap analysis."""
        dispatcher = LoRADispatcher()
        
        # Register similar domains
        dispatcher.register_domain(
            "medical1",
            sample_lora_weights,
            ["Patient care", "Medical treatment", "Healthcare"]
        )
        
        dispatcher.register_domain(
            "medical2",
            sample_lora_weights,
            ["Clinical care", "Patient treatment", "Medical practice"]
        )
        
        analysis = dispatcher.analyze_domain_overlap()
        
        assert "domain_count" in analysis
        assert analysis["domain_count"] == 2
        assert "pairwise_similarities" in analysis
        assert "medical1-medical2" in analysis["pairwise_similarities"]
        
        # Similar domains should have high similarity
        similarity = analysis["pairwise_similarities"]["medical1-medical2"]
        assert similarity > 0.5  # Reasonable threshold for similar domains
    
    def test_save_and_load(self, sample_lora_weights, temp_dir):
        """Test saving and loading dispatcher state."""
        # Create and populate dispatcher
        dispatcher1 = LoRADispatcher(confidence_threshold=0.75)
        
        dispatcher1.register_domain(
            "medical",
            sample_lora_weights,
            ["Medical text 1", "Patient care"]
        )
        
        dispatcher1.register_domain(
            "legal",
            sample_lora_weights,
            ["Legal document", "Contract"]
        )
        
        # Save
        save_path = os.path.join(temp_dir, "test_dispatcher")
        dispatcher1.save(save_path)
        
        # Check files exist
        assert os.path.exists(f"{save_path}_dispatcher.json")
        assert os.path.exists(f"{save_path}_router.pkl")
        assert os.path.exists(f"{save_path}_weights_medical.pt")
        assert os.path.exists(f"{save_path}_weights_legal.pt")
        
        # Load into new dispatcher
        dispatcher2 = LoRADispatcher.load(save_path)
        
        assert len(dispatcher2.domains) == 2
        assert "medical" in dispatcher2.domains
        assert "legal" in dispatcher2.domains
        assert dispatcher2.confidence_threshold == 0.75
    
    def test_cache_embeddings(self, sample_lora_weights):
        """Test embedding caching functionality."""
        dispatcher = LoRADispatcher(cache_embeddings=True)
        
        dispatcher.register_domain(
            "test",
            sample_lora_weights,
            ["Test text"]
        )
        
        query = "Test query"
        
        # First call - compute embedding
        domain1, _, _ = dispatcher.route_query(query)
        
        # Check cache
        assert query in dispatcher.embedding_cache
        
        # Second call - should use cache
        domain2, _, _ = dispatcher.route_query(query)
        
        assert domain1 == domain2


class TestDomainAdapter:
    """Test DomainAdapter class."""
    
    def test_domain_adapter_creation(self, sample_lora_weights):
        """Test creating a domain adapter."""
        adapter = DomainAdapter(
            name="test_domain",
            weights=sample_lora_weights,
            metadata={"version": "1.0", "description": "Test adapter"}
        )
        
        assert adapter.name == "test_domain"
        assert adapter.metadata["version"] == "1.0"
        assert len(adapter.weights) == len(sample_lora_weights)
    
    def test_domain_adapter_info(self, sample_lora_weights):
        """Test getting adapter information."""
        adapter = DomainAdapter(
            name="test_domain",
            weights=sample_lora_weights,
            metadata={"version": "1.0"}
        )
        
        info = adapter.get_info()
        
        assert info["name"] == "test_domain"
        assert "num_parameters" in info
        assert "weight_shapes" in info
        assert info["num_parameters"] > 0


class TestDispatcherIntegration:
    """Integration tests for dispatcher with model."""
    
    def test_apply_adapter(self, simple_model, sample_lora_weights):
        """Test applying adapter to model."""
        dispatcher = LoRADispatcher()
        
        dispatcher.register_domain(
            "test",
            sample_lora_weights,
            ["Test text"]
        )
        
        model, domain, confidence = dispatcher.apply_adapter(
            simple_model,
            "Test query",
            verbose=False
        )
        
        assert model is not None
        assert domain == "test"
        assert confidence == 1.0
    
    def test_no_suitable_domain(self, simple_model, sample_lora_weights):
        """Test behavior when no suitable domain is found."""
        dispatcher = LoRADispatcher(confidence_threshold=0.95)
        
        dispatcher.register_domain(
            "specific",
            sample_lora_weights,
            ["Very specific domain text"]
        )
        
        dispatcher.register_domain(
            "other",
            sample_lora_weights,
            ["Other specific text"]
        )
        
        model, domain, confidence = dispatcher.apply_adapter(
            simple_model,
            "Completely unrelated query",
            verbose=False
        )
        
        # Should return original model when no suitable domain
        assert model is not None
        if confidence < 0.95:
            assert domain is None
