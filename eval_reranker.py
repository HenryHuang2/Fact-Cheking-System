from util import load_data, load_vector_store, retrieve_evidence, load_reranker
from tqdm import tqdm
from pathlib import Path

MODELNAME = "models/ms-marco-MiniLM-L6-v2-finetuned-train-only/checkpoint-200"

def evaluate_retrieval_with_reranking(claims, evidence, model, index, metadata, initial_top_k, final_top_k, reranker=None, batch_size=10, score_gap_threshold=None):
    """Evaluate retrieval performance with optional reranking."""
    f_scores = []  # Store F-scores for each claim
    precisions = []  # Store precisions for each claim
    recalls = []  # Store recalls for each claim
    
    for claim_id, claim_data in tqdm(claims.items(), desc="Evaluating claims"):
        claim_text = claim_data["claim_text"]
        ground_truth_evidences = claim_data["evidences"]
            
        # Initial retrieval
        retrieved_evidence_ids = retrieve_evidence(
            claim_text, model, index, metadata, initial_top_k
        )

        # Apply reranking if a reranker is provided
        if reranker:
            # Process evidence in batches
            all_scores = []
            for i in range(0, len(retrieved_evidence_ids), batch_size):
                batch_ids = retrieved_evidence_ids[i:i + batch_size]
                batch_texts = [evidence[evidence_id] for evidence_id in batch_ids]
                
                # Prepare pairs for reranking
                pairs = [(claim_text, evidence_text) for evidence_text in batch_texts]
                
                # Get relevance scores for this batch
                batch_scores = reranker.predict(pairs)
                
                # Store scores with their corresponding evidence IDs
                all_scores.extend(zip(batch_ids, batch_scores))
            
            # Sort all scores
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top k scores and apply gap-based filtering
            top_k_scores = all_scores[:final_top_k]
            if top_k_scores:
                top_score = top_k_scores[0][1]
                final_retrieved_evidences = []
                
                for evidence_id, score in top_k_scores:
                    if not final_retrieved_evidences:  # Always include the first one
                        final_retrieved_evidences.append(evidence_id)
                    elif score_gap_threshold is not None:  # Only apply gap filtering if threshold is provided
                        score_gap = (top_score - score) / top_score
                        if score_gap <= score_gap_threshold:
                            final_retrieved_evidences.append(evidence_id)
                        else:
                            break
                    else:  # If no threshold provided, include all top k
                        final_retrieved_evidences.append(evidence_id)
            else:
                final_retrieved_evidences = []
        else:
            # If no reranker, just take the top-k from the initial retrieval
            final_retrieved_evidences = retrieved_evidence_ids[:final_top_k]
        
        # Calculate precision and recall for this claim
        relevant_retrieved = len(set(final_retrieved_evidences) & set(ground_truth_evidences))
        precision = relevant_retrieved / len(final_retrieved_evidences) if final_retrieved_evidences else 0
        recall = relevant_retrieved / len(ground_truth_evidences)
        
        # Store precision and recall
        precisions.append(precision)
        recalls.append(recall)
        
        # Calculate F-score for this claim
        f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f_scores.append(f_score)
    
    # Calculate mean metrics across all claims
    mean_f_score = sum(f_scores) / len(f_scores) if f_scores else 0
    mean_precision = sum(precisions) / len(precisions) if precisions else 0
    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    
    return {
        "mean_f_score": mean_f_score,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "total_claims_evaluated": len(f_scores)
    }

def main():
    # Load data
    data_dir = Path("data")
    claims_path = data_dir / "dev-claims.json"
    evidence_path = data_dir / "evidence.json"
    claims, evidence = load_data(claims_path, evidence_path)
    
    # Models to evaluate
    models = {
        "train_only": {
            "model_name": "models/snowflake-arctic-embed-l-v2.0-finetuned-train-only/checkpoint-300",
            "vector_store_dir": "vector_store_snowflake_finetuned_train_only/checkpoint-300"
        }
    }
    
    # Load reranker
    print("Loading reranker model...")
    reranker = load_reranker(MODELNAME)
    
    # Evaluate each model with and without reranking
    results = {}
    for model_name, config in models.items():
        print(f"\nEvaluating {model_name}...")
        model, index, metadata = load_vector_store(config["model_name"], config["vector_store_dir"])
        
        # Evaluate without reranking (baseline)
        print("Evaluating without reranking (baseline)...")
        results[f"{model_name}_baseline"] = evaluate_retrieval_with_reranking(
            claims, evidence, model, index, metadata, reranker=None, 
            initial_top_k=5, final_top_k=5
        )
        
        # Evaluate with reranking
        print(f"Evaluating with {MODELNAME}...")
        results[f"{model_name}_reranked"] = evaluate_retrieval_with_reranking(
            claims, evidence, model, index, metadata, reranker=reranker, 
            initial_top_k=10, final_top_k=5, score_gap_threshold=0.18
        )
    
    # Print results
    print("\nRetrieval Performance Results:")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Evidence Retrieval F-score (F) = {metrics['mean_f_score']:.4f}")
        print(f"Average Precision = {metrics['mean_precision']:.4f}")
        print(f"Average Recall = {metrics['mean_recall']:.4f}")
        print(f"Claims Evaluated: {metrics['total_claims_evaluated']}")
    
    # Calculate improvement for the last evaluated model
    last_model_name = list(models.keys())[-1]  # Get the last model name
    baseline = results[f"{last_model_name}_baseline"]
    reranked = results[f"{last_model_name}_reranked"]
    
    f_improvement = (reranked["mean_f_score"] - baseline["mean_f_score"]) / baseline["mean_f_score"] * 100
    prec_improvement = (reranked["mean_precision"] - baseline["mean_precision"]) / baseline["mean_precision"] * 100
    recall_improvement = (reranked["mean_recall"] - baseline["mean_recall"]) / baseline["mean_recall"] * 100
    
    print("\nImprovement with Reranking:")
    print(f"F-score: {f_improvement:.2f}%")
    print(f"Precision: {prec_improvement:.2f}%")
    print(f"Recall: {recall_improvement:.2f}%")

if __name__ == "__main__":
    main()