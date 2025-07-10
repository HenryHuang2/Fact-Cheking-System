import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

def load_data(claims_path, evidence_path):
    """Load claims and evidence data."""
    with open(claims_path, 'r', encoding='utf-8') as f:
        claims = json.load(f)
    with open(evidence_path, 'r', encoding='utf-8') as f:
        evidence = json.load(f)
    return claims, evidence

def load_vector_store(model_name, vector_store_dir):
    """Load vector store and metadata for a given model."""
    # Load metadata
    with open(Path(vector_store_dir) / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load FAISS index
    index = faiss.read_index(str(Path(vector_store_dir) / "evidence_index.faiss"))
    
    # Load model
    model = SentenceTransformer(model_name)
    
    return model, index, metadata

def load_reranker(reranker_path):
    """Load reranker model."""
    reranker = CrossEncoder(reranker_path)
    return reranker

def retrieve_evidence(claim_text, model, index, metadata, top_k):
    """Retrieve evidence for a given claim."""
    # Encode claim
    claim_embedding = model.encode([claim_text], normalize_embeddings=True)[0]
    
    # Search in FAISS index
    D, I = index.search(claim_embedding.reshape(1, -1), top_k)
    
    # Get retrieved evidence IDs
    retrieved_evidences = [metadata["evidence_ids"][i] for i in I[0]]
    
    return retrieved_evidences

def rerank_evidence(claim_text, evidence_ids, evidence_texts, reranker, top_k, score_gap_threshold=None):
    """Rerank retrieved evidence using a cross-encoder reranker and filter based on score gaps."""
    # Prepare pairs for reranking
    pairs = [(claim_text, evidence_text) for evidence_text in evidence_texts]
    
    # Get relevance scores
    scores = reranker.predict(pairs)
    
    # Create list of (evidence_id, score) tuples
    id_score_pairs = list(zip(evidence_ids, scores))
    
    # Sort by decreasing score
    id_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k scores
    top_k_scores = [score for _, score in id_score_pairs[:top_k]]
    
    if len(top_k_scores) > 0:
        top_score = top_k_scores[0]
        filtered_evidence = []
        
        # Check score gaps for each evidence
        for i, (evidence_id, score) in enumerate(id_score_pairs[:top_k]):
            if i == 0:  # Always include the top one
                filtered_evidence.append(evidence_id)
            elif score_gap_threshold is not None:  # Only apply gap filtering if threshold is provided
                # Calculate relative gap
                score_gap = (top_score - score) / top_score
                if score_gap <= score_gap_threshold:
                    filtered_evidence.append(evidence_id)
                else:
                    break  # Stop including evidence once we hit the threshold
            else:  # If no threshold provided, include all top k
                filtered_evidence.append(evidence_id)
    else:
        filtered_evidence = []
    
    return filtered_evidence
