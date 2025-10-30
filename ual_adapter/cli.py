"""
Command-Line Interface for UAL Adapter

Provides CLI commands for training, transferring, and using UAL adapters.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from loguru import logger

from ual_adapter import UniversalAdapter, LoRADispatcher
from ual_adapter.training.trainer import LoRATrainer


def train_command(args):
    """Handle the train command."""
    logger.info(f"Training adapter for model: {args.model}")
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load training data
    with open(args.data, 'r') as f:
        if args.data.endswith('.json'):
            data = json.load(f)
            texts = data if isinstance(data, list) else data.get('texts', [])
        else:
            texts = f.readlines()
    
    # Create trainer
    trainer = LoRATrainer(model, tokenizer, device=args.device)
    
    # Train adapter
    results = trainer.train_adapter(
        adapter_name=args.name,
        training_texts=texts,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        target_modules=args.target_modules.split(',') if args.target_modules else None
    )
    
    # Save results
    ual = UniversalAdapter(model, tokenizer)
    ual.adapters[args.name] = results["lora_weights"]
    ual.export_adapter(args.name, args.output)
    
    logger.info(f"✅ Adapter saved to {args.output}")
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"  Adapter: {args.name}")
    print(f"  Parameters: {results['num_parameters']:,}")
    print(f"  Train Loss: {results.get('train_loss', 'N/A'):.4f}")


def transfer_command(args):
    """Handle the transfer command."""
    logger.info(f"Transferring adapter from {args.source_model} to {args.target_model}")
    
    # Load source model
    source_model = AutoModel.from_pretrained(args.source_model)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model)
    source_ual = UniversalAdapter(source_model, source_tokenizer)
    
    # Import adapter
    source_ual.import_adapter(args.adapter)
    
    # Load target model
    target_model = AutoModel.from_pretrained(args.target_model)
    
    # Transfer adapter
    adapter_name = list(source_ual.adapters.keys())[0]
    transferred_weights, report = source_ual.transfer_to_model(
        adapter_name,
        target_model,
        projection_method=args.projection_method
    )
    
    # Save transferred adapter
    target_ual = UniversalAdapter(target_model)
    target_ual.adapters[adapter_name] = transferred_weights
    target_ual.export_adapter(adapter_name, args.output)
    
    logger.info(f"✅ Transferred adapter saved to {args.output}")
    
    # Print report
    print(f"\nTransfer Report:")
    print(f"  Source: {report['source_architecture']}")
    print(f"  Target: {report['target_architecture']}")
    print(f"  Dimension Change: {report['dimension_change']}")
    print(f"  Attachment Rate: {report['attachment_rate']:.1%}")
    print(f"  Weights Transferred: {report['weights_transferred']}")


def dispatch_command(args):
    """Handle the dispatch command for testing the dispatcher."""
    logger.info(f"Testing dispatcher with model: {args.model}")
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create dispatcher
    dispatcher = LoRADispatcher(confidence_threshold=args.threshold)
    
    # Load adapters
    for adapter_path in args.adapters:
        # Import each adapter
        adapter_name = Path(adapter_path).stem
        
        # Load adapter weights (simplified for example)
        weights = torch.load(f"{adapter_path}.weights.pt")
        
        # Load training texts (would need to be saved during training)
        with open(f"{adapter_path}.texts.json", 'r') as f:
            training_texts = json.load(f)
        
        dispatcher.register_domain(
            domain_name=adapter_name,
            adapter_weights=weights,
            training_texts=training_texts
        )
    
    logger.info(f"Loaded {len(dispatcher.domains)} domains")
    
    # Interactive testing
    if args.interactive:
        print("\n🎮 Interactive Dispatcher Test")
        print("Type 'quit' to exit\n")
        
        while True:
            query = input("Query: ").strip()
            if query.lower() == 'quit':
                break
            
            domain, confidence, all_scores = dispatcher.route_query(
                query,
                return_all_scores=True
            )
            
            print(f"\n🎯 Selected: {domain or 'None'} ({confidence:.2%})")
            if all_scores:
                print("All scores:")
                for d, s in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {d}: {s:.2%}")
            print()
    
    # Batch testing
    elif args.test_file:
        with open(args.test_file, 'r') as f:
            test_queries = f.readlines()
        
        results = []
        for query in test_queries:
            query = query.strip()
            if query:
                domain, confidence, _ = dispatcher.route_query(query)
                results.append({
                    "query": query,
                    "domain": domain,
                    "confidence": confidence
                })
        
        # Save results
        output_file = args.output or "dispatch_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UAL Adapter - Universal Adapter LoRA CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new adapter")
    train_parser.add_argument("--model", required=True, help="Base model name")
    train_parser.add_argument("--name", required=True, help="Adapter name")
    train_parser.add_argument("--data", required=True, help="Training data file")
    train_parser.add_argument("--output", required=True, help="Output path for adapter")
    train_parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    train_parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--target-modules", help="Comma-separated target modules")
    train_parser.add_argument("--device", default="auto", help="Device to use")
    
    # Transfer command
    transfer_parser = subparsers.add_parser("transfer", help="Transfer adapter to another model")
    transfer_parser.add_argument("--source-model", required=True, help="Source model name")
    transfer_parser.add_argument("--target-model", required=True, help="Target model name")
    transfer_parser.add_argument("--adapter", required=True, help="Path to adapter AIR files")
    transfer_parser.add_argument("--output", required=True, help="Output path")
    transfer_parser.add_argument("--projection-method", default="svd",
                                choices=["svd", "truncate", "interpolate"],
                                help="Dimension projection method")
    
    # Dispatch command
    dispatch_parser = subparsers.add_parser("dispatch", help="Test the LoRA dispatcher")
    dispatch_parser.add_argument("--model", required=True, help="Base model name")
    dispatch_parser.add_argument("--adapters", nargs="+", required=True,
                                help="Paths to adapter files")
    dispatch_parser.add_argument("--threshold", type=float, default=0.7,
                                help="Confidence threshold")
    dispatch_parser.add_argument("--interactive", action="store_true",
                                help="Interactive testing mode")
    dispatch_parser.add_argument("--test-file", help="File with test queries")
    dispatch_parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Execute command
    if args.command == "train":
        train_command(args)
    elif args.command == "transfer":
        transfer_command(args)
    elif args.command == "dispatch":
        dispatch_command(args)


if __name__ == "__main__":
    main()
