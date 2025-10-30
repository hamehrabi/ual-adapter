"""
LoRA Dispatcher with Intelligent Routing

Automatically selects and applies the most suitable LoRA adapter
based on query content using sentence embeddings and classification.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from loguru import logger
import pickle
import json


class LoRADispatcher:
    """
    Intelligent dispatcher that routes queries to appropriate domain LoRAs.
    
    Uses sentence embeddings and a trained classifier to determine which
    domain-specific adapter to use for a given query.
    """
    
    def __init__(
        self,
        encoder_model: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.7,
        cache_embeddings: bool = True
    ):
        """
        Initialize the LoRA dispatcher.
        
        Args:
            encoder_model: Sentence transformer model for embeddings
            confidence_threshold: Minimum confidence for domain selection
            cache_embeddings: Whether to cache computed embeddings
        """
        self.encoder = SentenceTransformer(encoder_model)
        self.encoder_model_name = encoder_model  # Store for later saving
        self.confidence_threshold = confidence_threshold
        self.cache_embeddings = cache_embeddings

        # Domain registry
        self.domains: Dict[str, DomainAdapter] = {}
        self.router = None
        self.domain_embeddings = {}
        self.embedding_cache = {} if cache_embeddings else None

        logger.info(f"Initialized LoRA Dispatcher with encoder: {encoder_model}")
    
    def register_domain(
        self,
        domain_name: str,
        adapter_weights: Dict[str, torch.Tensor],
        training_texts: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a domain-specific LoRA adapter.
        
        Args:
            domain_name: Name of the domain
            adapter_weights: LoRA weights for this domain
            training_texts: Sample texts from this domain for routing
            metadata: Optional metadata about the domain
        """
        if len(self.domains) >= 10:
            logger.warning("Maximum 10 domains recommended for optimal performance")
        
        # Create domain adapter
        domain_adapter = DomainAdapter(
            name=domain_name,
            weights=adapter_weights,
            metadata=metadata or {}
        )
        
        # Compute and store embeddings for training texts
        logger.info(f"Computing embeddings for domain '{domain_name}'...")
        embeddings = self.encoder.encode(
            training_texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        self.domains[domain_name] = domain_adapter
        self.domain_embeddings[domain_name] = embeddings
        
        logger.info(
            f"Registered domain '{domain_name}' with "
            f"{len(training_texts)} training samples"
        )
        
        # Retrain router with new domain
        if len(self.domains) > 1:
            self._train_router()
    
    def _train_router(self) -> None:
        """Train the router classifier on domain embeddings."""
        logger.info("Training router classifier...")
        
        X = []
        y = []
        
        # Prepare training data
        for domain_id, (domain_name, embeddings) in enumerate(
            self.domain_embeddings.items()
        ):
            X.extend(embeddings)
            y.extend([domain_id] * len(embeddings))
        
        # Train logistic regression classifier
        self.router = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42
        )
        self.router.fit(X, y)
        
        # Evaluate on training data
        accuracy = self.router.score(X, y)
        logger.info(f"Router accuracy on training data: {accuracy:.2%}")
    
    def route_query(
        self,
        query: str,
        return_all_scores: bool = False
    ) -> Tuple[Optional[str], float, Optional[Dict[str, float]]]:
        """
        Determine which domain adapter to use for a query.
        
        Args:
            query: The input query text
            return_all_scores: Whether to return scores for all domains
            
        Returns:
            Tuple of (selected_domain, confidence, all_scores_dict)
        """
        # Get embedding for query (always compute/cache for consistency)
        if self.cache_embeddings and query in self.embedding_cache:
            query_embedding = self.embedding_cache[query]
        else:
            query_embedding = self.encoder.encode([query])[0]
            if self.cache_embeddings:
                self.embedding_cache[query] = query_embedding

        if not self.router or len(self.domains) < 2:
            # Use first/only domain if no routing needed
            if self.domains:
                domain_name = list(self.domains.keys())[0]
                return domain_name, 1.0, None
            return None, 0.0, None
        
        # Get probabilities from router
        probabilities = self.router.predict_proba([query_embedding])[0]
        domain_names = list(self.domains.keys())
        
        # Find best domain
        best_idx = np.argmax(probabilities)
        best_confidence = probabilities[best_idx]
        best_domain = domain_names[best_idx] if best_confidence >= self.confidence_threshold else None
        
        # Prepare all scores if requested
        all_scores = None
        if return_all_scores:
            all_scores = {
                domain_names[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        
        return best_domain, float(best_confidence), all_scores
    
    def apply_adapter(
        self,
        model: torch.nn.Module,
        query: str,
        verbose: bool = True
    ) -> Tuple[torch.nn.Module, str, float]:
        """
        Apply the appropriate adapter to a model based on the query.
        
        Args:
            model: The base model to apply adapter to
            query: The input query
            verbose: Whether to log routing decisions
            
        Returns:
            Tuple of (model_with_adapter, selected_domain, confidence)
        """
        # Route query to domain
        domain, confidence, _ = self.route_query(query)
        
        if domain:
            if verbose:
                logger.info(
                    f"🎯 Routing to domain '{domain}' "
                    f"(confidence: {confidence:.2%})"
                )
            
            # Apply domain adapter weights
            adapter = self.domains[domain]
            model = self._apply_weights(model, adapter.weights)
            
            return model, domain, confidence
        else:
            if verbose:
                logger.info(
                    f"⚠️ No suitable domain found "
                    f"(best confidence: {confidence:.2%} < threshold)"
                )
            return model, None, confidence
    
    def _apply_weights(
        self,
        model: torch.nn.Module,
        adapter_weights: Dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        """Apply LoRA weights to model."""
        # This is a simplified version - in production would use PEFT
        for name, weight in adapter_weights.items():
            # Find corresponding module in model
            try:
                module = model
                for part in name.split('.'):
                    module = getattr(module, part)
                
                # Apply LoRA weight update
                if hasattr(module, 'weight'):
                    module.weight.data += weight
                    
            except AttributeError:
                logger.debug(f"Could not apply weight to {name}")
        
        return model
    
    def analyze_domain_overlap(self) -> Dict[str, Any]:
        """Analyze overlap between registered domains."""
        if len(self.domains) < 2:
            return {"message": "Need at least 2 domains for overlap analysis"}
        
        analysis = {
            "domain_count": len(self.domains),
            "pairwise_similarities": {},
            "domain_separability": {}
        }
        
        domain_names = list(self.domains.keys())
        
        # Compute pairwise cosine similarities
        for i, domain1 in enumerate(domain_names):
            for domain2 in domain_names[i+1:]:
                embeddings1 = self.domain_embeddings[domain1]
                embeddings2 = self.domain_embeddings[domain2]
                
                # Compute mean embeddings
                mean1 = np.mean(embeddings1, axis=0)
                mean2 = np.mean(embeddings2, axis=0)
                
                # Cosine similarity
                similarity = np.dot(mean1, mean2) / (
                    np.linalg.norm(mean1) * np.linalg.norm(mean2)
                )
                
                pair_key = f"{domain1}-{domain2}"
                analysis["pairwise_similarities"][pair_key] = float(similarity)
        
        # Compute domain separability (classification confidence)
        if self.router:
            for domain_id, domain_name in enumerate(domain_names):
                embeddings = self.domain_embeddings[domain_name]
                predictions = self.router.predict_proba(embeddings)
                
                # Average confidence for correct domain
                correct_confidence = np.mean(predictions[:, domain_id])
                analysis["domain_separability"][domain_name] = float(correct_confidence)
        
        return analysis
    
    def save(self, path: str) -> None:
        """Save dispatcher state to disk."""
        save_data = {
            "domains": {
                name: {
                    "metadata": domain.metadata,
                    "weights_shape": {
                        k: list(v.shape) for k, v in domain.weights.items()
                    }
                }
                for name, domain in self.domains.items()
            },
            "domain_embeddings": self.domain_embeddings,
            "confidence_threshold": self.confidence_threshold,
            "encoder_model": self.encoder_model_name
        }
        
        # Save router separately if trained
        if self.router:
            with open(f"{path}_router.pkl", "wb") as f:
                pickle.dump(self.router, f)
        
        # Save main data
        with open(f"{path}_dispatcher.json", "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Save adapter weights
        for domain_name, domain in self.domains.items():
            torch.save(
                domain.weights,
                f"{path}_weights_{domain_name}.pt"
            )
        
        logger.info(f"Saved dispatcher to {path}")
    
    @classmethod
    def load(cls, path: str) -> "LoRADispatcher":
        """Load dispatcher state from disk."""
        # Load main data
        with open(f"{path}_dispatcher.json", "r") as f:
            save_data = json.load(f)
        
        # Create dispatcher
        dispatcher = cls(
            encoder_model=save_data.get("encoder_model", "all-MiniLM-L6-v2"),
            confidence_threshold=save_data["confidence_threshold"]
        )
        
        # Load domains
        for domain_name, domain_data in save_data["domains"].items():
            weights = torch.load(f"{path}_weights_{domain_name}.pt")
            domain = DomainAdapter(
                name=domain_name,
                weights=weights,
                metadata=domain_data["metadata"]
            )
            dispatcher.domains[domain_name] = domain
        
        # Load embeddings
        dispatcher.domain_embeddings = {
            k: np.array(v) for k, v in save_data["domain_embeddings"].items()
        }
        
        # Load router if exists
        try:
            with open(f"{path}_router.pkl", "rb") as f:
                dispatcher.router = pickle.load(f)
        except FileNotFoundError:
            logger.warning("No router found, will retrain on first use")
        
        logger.info(f"Loaded dispatcher from {path}")
        return dispatcher


class DomainAdapter:
    """Container for domain-specific adapter information."""
    
    def __init__(
        self,
        name: str,
        weights: Dict[str, torch.Tensor],
        metadata: Dict[str, Any]
    ):
        """
        Initialize domain adapter.
        
        Args:
            name: Domain name
            weights: LoRA weights for this domain
            metadata: Additional information about the domain
        """
        self.name = name
        self.weights = weights
        self.metadata = metadata
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about this domain adapter."""
        return {
            "name": self.name,
            "num_parameters": sum(w.numel() for w in self.weights.values()),
            "weight_shapes": {k: list(v.shape) for k, v in self.weights.items()},
            "metadata": self.metadata
        }
