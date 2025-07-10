import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import json
import os
from pathlib import Path
import argparse

# Default configuration
DEFAULT_MODEL_DIR = "models/snowflake-arctic-embed-l-v2.0-finetuned-train-only/"
DEFAULT_OUTPUT_DIR = "vector_store_snowflake_finetuned_train_only"
BATCH_SIZE = 64
DIMENSION = 1024  # snowflake-arctic-embed-l-v2.0 embedding dimension

def load_data(evidence_path):
    """Load evidence from JSON file."""
    with open(evidence_path, 'r', encoding='utf-8') as f:
        evidence = json.load(f)
    return evidence

def get_embeddings(text_dict, model, batch_size=32):
    """Generate embeddings for a dictionary of texts using Snowflake Arctic Embed model."""
    embeddings = {}
    texts = list(text_dict.values())
    ids = list(text_dict.keys())
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        # Generate embeddings with normalization
        batch_embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Store embeddings with their corresponding IDs
        for idx, emb in enumerate(batch_embeddings):
            embeddings[batch_ids[idx]] = emb
    
    return embeddings

def build_vector_store(evidence, model_dir, output_dir):
    """Build and save the evidence vector store using Snowflake Arctic Embed model."""
    # Create output directory if it doesn't exist
    for checkpoint in os.listdir(model_dir):
        checkpoint_output_dir = output_dir + "/" + checkpoint
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        
        # Load model
        print("Loading model...")
        model = SentenceTransformer(model_dir + "/" + checkpoint)
        
        # Generate evidence embeddings
        print("Generating evidence embeddings...")
        evidence_embeddings = get_embeddings(
            evidence,
            model,
            batch_size=BATCH_SIZE
        )
        
        # Convert evidence embeddings to numpy array for FAISS
        evidence_ids = sorted(evidence_embeddings.keys())
        evidence_embeddings_array = np.array([evidence_embeddings[id] for id in evidence_ids])
        
        # Build FAISS index
        print("Building FAISS index...")
        index = faiss.IndexFlatIP(DIMENSION)  # Using inner product for cosine similarity
        index.add(evidence_embeddings_array)
        
        # Save the index and metadata
        print("Saving vector store...")
        faiss.write_index(index, os.path.join(checkpoint_output_dir, "evidence_index.faiss"))
        
        # Save metadata
        metadata = {
            "evidence_ids": evidence_ids,
            "evidence_embeddings_shape": evidence_embeddings_array.shape,
            "model_name": "snowflake-arctic-embed-l-v2.0-finetuned-train-only"+ "-" + checkpoint,
            "dimension": DIMENSION
        }
        
        with open(os.path.join(checkpoint_output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print("Vector store built successfully!")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Build vector store using Snowflake Arctic Embed model')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR,
                      help=f'Directory containing the model checkpoints (default: {DEFAULT_MODEL_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                      help=f'Directory to save the vector store (default: {DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # Load data
    data_dir = Path("data")
    evidence_path = data_dir / "evidence.json"
    
    evidence = load_data(evidence_path)
    build_vector_store(evidence, args.model_dir, args.output_dir) 