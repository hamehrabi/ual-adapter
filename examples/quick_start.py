"""
Quick Start Example: Basic UAL Adapter Usage

This example shows the simplest way to:
1. Train a LoRA adapter
2. Transfer it to another model
3. Use it for generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ual_adapter import UniversalAdapter

def quick_start():
    """Quick start example for UAL Adapter."""
    
    print("🚀 UAL Adapter Quick Start")
    print("=" * 50)
    
    # Load a small model for demonstration
    print("\n1️⃣ Loading base model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create UAL adapter
    ual = UniversalAdapter(model, tokenizer)
    
    # Training data (small sample for demo)
    medical_texts = [
        "The patient exhibits symptoms of respiratory infection.",
        "Treatment includes antibiotics and rest.",
        "Clinical diagnosis confirms bacterial pneumonia.",
    ]
    
    print("\n2️⃣ Training medical domain adapter...")
    result = ual.train_adapter(
        adapter_name="medical",
        training_data=medical_texts,
        rank=4,  # Small rank for quick demo
        epochs=1,
        batch_size=1
    )
    
    print(f"   ✓ Trained adapter with {result['num_weights']} weights")
    
    # Export to portable format
    print("\n3️⃣ Exporting adapter to AIR format...")
    ual.export_adapter("medical", "medical_adapter")
    print("   ✓ Exported to medical_adapter.air")
    
    # Simulate transfer to another model
    print("\n4️⃣ Transferring to a different model...")
    target_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    transferred_weights, report = ual.transfer_to_model(
        adapter_name="medical",
        target_model=target_model,
        projection_method="svd"
    )
    
    print(f"   ✓ Transfer complete!")
    print(f"   • Attachment rate: {report['attachment_rate']:.1%}")
    print(f"   • Architecture: {report['source_architecture']} → {report['target_architecture']}")
    
    print("\n✅ Quick start complete!")
    print("\nYou can now use the transferred adapter with the target model!")


def generation_example():
    """Example of using UAL for text generation."""
    
    print("\n📝 Text Generation Example")
    print("=" * 50)
    
    # Load model and create UAL
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # In practice, you would load a pre-trained adapter
    # For demo, we'll just show the structure
    
    print("\nGenerating text with domain adapter...")
    prompt = "The patient's symptoms include"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate (this would use the adapted model in practice)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")


def minimal_example():
    """The absolute minimal example."""
    
    print("\n⚡ Minimal Example (3 lines)")
    print("=" * 50)
    
    print("\n```python")
    print("from ual_adapter import UniversalAdapter")
    print("ual = UniversalAdapter(your_model, your_tokenizer)")
    print("ual.train_adapter('domain', texts, rank=8)")
    print("```")
    
    print("\nThat's it! Your adapter is ready to transfer to any model.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "generation":
            generation_example()
        elif example == "minimal":
            minimal_example()
        else:
            quick_start()
    else:
        # Run all examples
        quick_start()
        generation_example()
        minimal_example()
        
        print("\n" + "=" * 50)
        print("🎉 All examples complete!")
        print("\nRun specific examples with:")
        print("  python quick_start.py generation")
        print("  python quick_start.py minimal")
