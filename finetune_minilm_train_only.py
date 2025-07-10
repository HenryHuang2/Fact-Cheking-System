import json
from pathlib import Path
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.trainer import BatchSamplers

def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def create_dataset(claims, all_evidences):
    anchor = []
    positive = []

    for claim_id in claims:
        for evidence_id in claims[claim_id]['evidences']:
            anchor.append(claims[claim_id]['claim_text'])
            positive.append(all_evidences[evidence_id])

    return Dataset.from_dict({"anchor": anchor, "positive": positive})

MODELNAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
num_rand_negatives = 13
RUN_NAME = "ms-marco-MiniLM-L6-v2-finetuned"

def main():
    data_path = Path('data')
    train_claims = load_data(data_path / 'train-claims.json')
    dev_claims = load_data(data_path / 'dev-claims.json')
    all_evidences = load_data(data_path / "evidence.json")
    
    print("Creating train and dev datasets...")
    train_dataset = create_dataset(train_claims, all_evidences)

    dev_dataset = create_dataset(dev_claims, all_evidences)

    print(f"Loading {MODELNAME} model...")
    model = CrossEncoder(MODELNAME)

    loss = CachedMultipleNegativesRankingLoss(
        model=model,
        num_negatives=num_rand_negatives,
        mini_batch_size=64,
    )

    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{RUN_NAME}",
        
        # Training parameters
        num_train_epochs=5,
        per_device_train_batch_size=24,  # Adjust based on GPU memory: 8-16GB: 16, 24GB+: 32
        per_device_eval_batch_size=24,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        
        # Mixed precision training
        fp16=True,  # Use FP16 if GPU supports it
        bf16=False,  # Use BF16 if GPU supports it (e.g., A100)
        
        # Optimization
        optim="adamw_torch",  # Use PyTorch's AdamW optimizer
        lr_scheduler_type="linear",  # Linear learning rate decay
        gradient_accumulation_steps=1,  # Increase if OOM occurs
        
        # Batch sampling
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        
        # Evaluation and checkpointing
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        
        # Logging
        logging_steps=100,
        logging_dir="logs",
        report_to="tensorboard",  # Use tensorboard for visualization
        
        # Early stopping
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Run name for tracking
        run_name=RUN_NAME,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # Add dev dataset for evaluation
        loss=loss,
    )

    print("Training model...")
    trainer.train()

    print("Completed")

if __name__ == '__main__':
    main()