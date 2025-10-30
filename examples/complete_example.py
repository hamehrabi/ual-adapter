"""
Complete Example: Training and Transferring UAL Adapters

This example demonstrates the full workflow of:
1. Training domain-specific LoRA adapters
2. Exporting to AIR format
3. Transferring across architectures
4. Using the intelligent dispatcher
"""

import torch
from transformers import AutoModel, AutoTokenizer
from ual_adapter import (
    UniversalAdapter,
    LoRADispatcher,
    LoRATrainer
)


def main():
    print("=" * 60)
    print("UAL Adapter - Complete Example")
    print("=" * 60)
    
    # ============================================================
    # Step 1: Train Domain-Specific Adapters
    # ============================================================
    print("\n📚 Step 1: Training Domain Adapters")
    print("-" * 40)
    
    # Load base model (using a small model for example)
    model_name = "gpt2"
    print(f"Loading base model: {model_name}")
    base_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create trainer
    trainer = LoRATrainer(base_model, tokenizer)
    
    # Define training data for different domains
    domains_data = {
        "medical": [
            "The patient presents with acute respiratory symptoms including persistent cough and dyspnea.",
            "Clinical examination revealed bilateral crackles and reduced oxygen saturation.",
            "Treatment protocol includes bronchodilators and systemic corticosteroids.",
            "Laboratory results show elevated inflammatory markers consistent with pneumonia.",
            "Post-treatment assessment indicates significant improvement in lung function.",
        ],
        "legal": [
            "The contract stipulates specific performance requirements and penalty clauses.",
            "Pursuant to section 5.2, the defendant breached the non-disclosure agreement.",
            "The court ruled in favor of the plaintiff, awarding compensatory damages.",
            "Legal precedent establishes liability in cases of professional negligence.",
            "The arbitration clause requires disputes to be resolved through binding arbitration.",
        ],
        "technical": [
            "The API endpoint returns a JSON response with paginated results.",
            "Implement caching strategy to reduce database query latency.",
            "The microservices architecture ensures scalability and fault tolerance.",
            "Debug the memory leak in the asynchronous event processing pipeline.",
            "Optimize the algorithm complexity from O(n²) to O(n log n).",
        ]
    }
    
    # Train adapters for each domain
    trained_adapters = {}
    
    for domain_name, training_texts in domains_data.items():
        print(f"\nTraining adapter for domain: {domain_name}")
        
        results = trainer.train_adapter(
            adapter_name=domain_name,
            training_texts=training_texts,
            rank=8,  # Using smaller rank for example
            alpha=16,
            learning_rate=5e-4,
            num_epochs=1,  # Quick training for example
            batch_size=2,
            use_huggingface_trainer=False  # Use custom loop for simplicity
        )
        
        trained_adapters[domain_name] = results
        print(f"✅ Trained {domain_name}: {results['num_parameters']:,} parameters")
    
    # ============================================================
    # Step 2: Export Adapters to AIR Format
    # ============================================================
    print("\n💾 Step 2: Exporting to AIR Format")
    print("-" * 40)
    
    ual = UniversalAdapter(base_model, tokenizer)
    
    # Add trained adapters to UAL
    for domain_name, adapter_data in trained_adapters.items():
        ual.adapters[domain_name] = adapter_data["lora_weights"]
        
        # Export to AIR format
        output_path = f"adapters/{domain_name}_adapter"
        print(f"Exporting {domain_name} adapter to {output_path}.air")
        
        # Note: In real usage, you would save to actual files
        # ual.export_adapter(domain_name, output_path)
    
    # ============================================================
    # Step 3: Transfer to Different Architecture
    # ============================================================
    print("\n🔄 Step 3: Cross-Architecture Transfer")
    print("-" * 40)
    
    # Simulate loading a different model architecture
    # In practice, you might use "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    target_model_name = "distilgpt2"  # Using distilgpt2 as example
    print(f"Loading target model: {target_model_name}")
    target_model = AutoModel.from_pretrained(target_model_name)
    
    # Transfer medical adapter as example
    print("\nTransferring medical adapter to target model...")
    transferred_weights, report = ual.transfer_to_model(
        adapter_name="medical",
        target_model=target_model,
        projection_method="svd"
    )
    
    print(f"Transfer Report:")
    print(f"  Source: {report['source_architecture']}")
    print(f"  Target: {report['target_architecture']}")
    print(f"  Dimension change: {report['dimension_change']}")
    print(f"  Attachment rate: {report['attachment_rate']:.1%}")
    
    # ============================================================
    # Step 4: Intelligent Dispatcher
    # ============================================================
    print("\n🎯 Step 4: Intelligent LoRA Dispatcher")
    print("-" * 40)
    
    # Create dispatcher
    dispatcher = LoRADispatcher(confidence_threshold=0.6)
    
    # Register all domains
    for domain_name, training_texts in domains_data.items():
        if domain_name in trained_adapters:
            dispatcher.register_domain(
                domain_name=domain_name,
                adapter_weights=trained_adapters[domain_name]["lora_weights"],
                training_texts=training_texts,
                metadata={"description": f"{domain_name.capitalize()} domain adapter"}
            )
    
    print(f"Registered {len(dispatcher.domains)} domains")
    
    # Test queries
    test_queries = [
        "The patient's blood pressure is elevated",
        "The contract was terminated due to breach",
        "Fix the bug in the authentication module",
        "Analyze market trends for Q3 earnings",  # Ambiguous query
    ]
    
    print("\nTesting query routing:")
    for query in test_queries:
        domain, confidence, all_scores = dispatcher.route_query(
            query,
            return_all_scores=True
        )
        
        print(f"\nQuery: '{query}'")
        print(f"  → Selected: {domain or 'None'} ({confidence:.1%} confidence)")
        
        if all_scores:
            print("  All scores:")
            for d, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {d}: {score:.1%}")
    
    # ============================================================
    # Step 5: Domain Overlap Analysis
    # ============================================================
    print("\n📊 Step 5: Analyzing Domain Overlap")
    print("-" * 40)
    
    analysis = dispatcher.analyze_domain_overlap()
    
    print(f"Domain count: {analysis['domain_count']}")
    print("\nPairwise similarities:")
    for pair, similarity in analysis['pairwise_similarities'].items():
        print(f"  {pair}: {similarity:.3f}")
    
    if 'domain_separability' in analysis:
        print("\nDomain separability (classification confidence):")
        for domain, conf in analysis['domain_separability'].items():
            print(f"  {domain}: {conf:.1%}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("✨ Example Complete!")
    print("=" * 60)
    print("\nKey Achievements:")
    print(f"  • Trained {len(trained_adapters)} domain adapters")
    print(f"  • Demonstrated cross-architecture transfer")
    print(f"  • Set up intelligent routing across domains")
    print(f"  • Analyzed domain relationships")
    
    print("\nNext Steps:")
    print("  1. Train on larger datasets for better performance")
    print("  2. Export adapters for production use")
    print("  3. Test transfer across more diverse architectures")
    print("  4. Fine-tune confidence thresholds for routing")


if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run example
    main()
