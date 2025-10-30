"""
LoRA Training Module

Handles training of domain-specific LoRA adapters.
"""

from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset as HFDataset
from loguru import logger
import numpy as np
from tqdm import tqdm


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        Initialize text dataset.
        
        Args:
            texts: List of training texts
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class LoRATrainer:
    """
    Trainer for LoRA adapters with domain specialization.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "auto"
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            base_model: Base model to fine-tune
            tokenizer: Tokenizer for the model
            device: Device to train on
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.base_model = self.base_model.to(self.device)
        
        # Training history
        self.training_history = []
    
    def train_adapter(
        self,
        adapter_name: str,
        training_texts: List[str],
        validation_texts: Optional[List[str]] = None,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 8,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        output_dir: Optional[str] = None,
        use_huggingface_trainer: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a LoRA adapter on domain-specific data.
        
        Args:
            adapter_name: Name for the adapter
            training_texts: List of training texts
            validation_texts: Optional validation texts
            rank: LoRA rank
            alpha: LoRA alpha parameter
            dropout: LoRA dropout
            target_modules: Target modules for LoRA
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size
            warmup_steps: Number of warmup steps
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            output_dir: Directory to save checkpoints
            use_huggingface_trainer: Whether to use HF Trainer
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting training for adapter '{adapter_name}'")
        logger.info(f"Training samples: {len(training_texts)}")
        
        # Auto-detect target modules if not provided
        if target_modules is None:
            from ual_adapter.utils.model_utils import ModelAnalyzer
            analyzer = ModelAnalyzer(self.base_model)
            target_modules = analyzer.get_lora_target_modules()
            logger.info(f"Auto-detected target modules: {target_modules}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Create PEFT model
        peft_model = get_peft_model(self.base_model, lora_config)
        peft_model.print_trainable_parameters()
        
        if use_huggingface_trainer:
            results = self._train_with_hf_trainer(
                peft_model=peft_model,
                adapter_name=adapter_name,
                training_texts=training_texts,
                validation_texts=validation_texts,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                logging_steps=logging_steps,
                save_steps=save_steps,
                output_dir=output_dir,
                **kwargs
            )
        else:
            results = self._train_with_custom_loop(
                peft_model=peft_model,
                adapter_name=adapter_name,
                training_texts=training_texts,
                validation_texts=validation_texts,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                warmup_steps=warmup_steps,
                **kwargs
            )
        
        # Extract trained weights
        lora_weights = self._extract_lora_weights(peft_model)
        
        # Add to results
        results["lora_weights"] = lora_weights
        results["num_parameters"] = sum(w.numel() for w in lora_weights.values())
        results["target_modules"] = target_modules
        results["config"] = {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout
        }
        
        # Store in history
        self.training_history.append({
            "adapter_name": adapter_name,
            "timestamp": torch.cuda.Event(enable_timing=True),
            "results": results
        })
        
        logger.info(f"✅ Training complete for '{adapter_name}'")
        
        return results
    
    def _train_with_hf_trainer(
        self,
        peft_model,
        adapter_name: str,
        training_texts: List[str],
        validation_texts: Optional[List[str]],
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        warmup_steps: int,
        logging_steps: int,
        save_steps: int,
        output_dir: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Train using HuggingFace Trainer."""
        # Prepare datasets
        train_dataset = HFDataset.from_dict({"text": training_texts})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_train = tokenized_train.remove_columns(["text"])
        
        # Prepare validation dataset if provided
        eval_dataset = None
        if validation_texts:
            eval_dataset = HFDataset.from_dict({"text": validation_texts})
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.remove_columns(["text"])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir or f"./checkpoints/{adapter_name}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=logging_steps if eval_dataset else None,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            **kwargs
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        train_result = trainer.train()
        
        # Prepare results
        results = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "epoch": train_result.metrics["epoch"],
        }
        
        if eval_dataset:
            eval_result = trainer.evaluate()
            results["eval_loss"] = eval_result["eval_loss"]
        
        return results
    
    def _train_with_custom_loop(
        self,
        peft_model,
        adapter_name: str,
        training_texts: List[str],
        validation_texts: Optional[List[str]],
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        warmup_steps: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Train with custom training loop."""
        # Create dataset
        train_dataset = TextDataset(training_texts, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            peft_model.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        peft_model.train()
        total_loss = 0
        step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = peft_model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Update metrics
                epoch_loss += loss.item()
                total_loss += loss.item()
                step += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Warmup
                if step < warmup_steps:
                    lr_scale = min(1.0, step / warmup_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate * lr_scale
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
        
        results = {
            "train_loss": total_loss / step,
            "num_epochs": num_epochs,
            "total_steps": step,
        }
        
        return results
    
    def _extract_lora_weights(
        self,
        peft_model
    ) -> Dict[str, torch.Tensor]:
        """Extract LoRA weights from PEFT model."""
        lora_weights = {}
        
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                # Clean up the name
                clean_name = name.replace("base_model.model.", "")
                lora_weights[clean_name] = param.detach().cpu().clone()
        
        logger.debug(f"Extracted {len(lora_weights)} LoRA weights")
        
        return lora_weights
    
    def compute_perplexity(
        self,
        model: nn.Module,
        texts: List[str],
        batch_size: int = 8
    ) -> float:
        """
        Compute perplexity on a set of texts.
        
        Args:
            model: Model to evaluate
            texts: Texts to compute perplexity on
            batch_size: Batch size for evaluation
            
        Returns:
            Perplexity value
        """
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                
                total_loss += outputs.loss.item() * batch["input_ids"].numel()
                total_tokens += batch["input_ids"].numel()
        
        perplexity = np.exp(total_loss / total_tokens)
        
        return perplexity
