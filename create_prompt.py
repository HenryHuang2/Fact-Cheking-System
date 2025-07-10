import json
from pathlib import Path
from typing import Dict, List, Tuple
from util import load_vector_store, retrieve_evidence, rerank_evidence, load_reranker

PROMPT_TEMPLATE = f"""Task: Fact Checking
Given the following claims and supporting evidence, determine if the claim is supported, refuted, disputed, or if there is not enough information.
To determine the label of the query claim, you should follow these steps:
    1. use the examples to help you understand the correct labeling of the claims given the evidences.
    2. evaluate the relationship between the query claim and the evidences.
    3. ignore evidences that are NOT_ENOUGH_INFO. For the remaining evidences, if the majority support the claim, the label should be SUPPORTS. If the majority refute the claim, the label should be REFUTES. 
    If supporting and refuting evidences are comparable in proportion, the label should be DISPUTED. If there are not enough evidences to determining the claim's validity, the label should be NOT_ENOUGH_INFO.
    4. end your reasoning with the label of the query claim.\n\n"""

ANALYSIS = """
Claim: The minute increase of anthropogenic CO2 in the atmosphere (0.008%) was not the cause of the warming—it was a continuation of natural cycles that occurred over the past 500 years.
Evidence: The introduction includes this statement: There is strong evidence that the warming of the Earth over the last half-century has been caused largely by human activity, such as the burning of fossil fuels and changes in land use, including agriculture and deforestation.
- REFUTES: States warming is caused by human activity, contradicting natural cycles claim.

Claim: Monckton appears to have cherry-picked temperature data from a few stations.
Evidence: Canadian Climate Normals 1981–2010 Station Data.
- NOT_ENOUGH_INFO: Mentions data source but nothing about Monckton's selection methods.

Claim: Antarctica is too cold to lose ice.
Evidence: As a result of continued warming, the polar ice caps melted and much of Gondwana became a desert.
- REFUTES: States polar ice caps melted due to warming, contradicting claim.
"""

def format_example(claim_id: str, claim_data: Dict, evidences: Dict, show_label: bool = True) -> str:
    """Format a single claim and its evidence as an example."""
    claim_text = claim_data["claim_text"]
    evidence_ids = claim_data["evidences"]
    label = claim_data["claim_label"]
    evidence_texts = [evidences[evidence_id] for evidence_id in evidence_ids]
    example = f"Claim: {claim_text}\nEvidence:\n"
    for i, evidence in enumerate(evidence_texts, 1):
        example += f"{i}. {evidence}\n"
    if show_label:
        example += f"Label: {label}\n"
    return example

def create_prompt(claim_id: str, claim_data: Dict, evidences: Dict, model, index, metadata, reranker, initial_top_k=10, final_top_k=5, score_gap_threshold=0.18) -> Tuple[str, List[str]]:
    """Create a prompt for a single claim using retrieved evidence.
    Returns:
        Tuple[str, List[str]]: The prompt string and list of retrieved evidence IDs
    """
    prompt = PROMPT_TEMPLATE
    
    # Always use predefined analysis examples
    prompt += ANALYSIS
    
    prompt += "---\n\n"
    
    # Retrieve evidence for the query claim
    retrieved_evidence_ids = retrieve_evidence(claim_data["claim_text"], model, index, metadata, top_k=initial_top_k)
    
    # Get evidence texts for reranking
    evidence_texts = [evidences[evidence_id] for evidence_id in retrieved_evidence_ids]
    
    # Rerank evidence using the rerank_evidence function
    retrieved_evidence_ids = rerank_evidence(
        claim_data["claim_text"],
        retrieved_evidence_ids,
        evidence_texts,
        reranker,
        final_top_k,
        score_gap_threshold=score_gap_threshold
    )
    
    # Add query claim with retrieved evidence
    query_data = {
        "claim_text": claim_data["claim_text"],
        "evidences": retrieved_evidence_ids,
        "claim_label": ""  # Empty label for query claim
    }
    prompt += format_example(claim_id, query_data, evidences, show_label=True)
    
    return prompt, retrieved_evidence_ids

def main():
    # Load dev claims and evidence
    claims, evidences = load_retrieval_data("data/dev-claims.json", "data/evidence.json")
    
    # Load retrieval model and index
    model_name = "models/snowflake-arctic-embed-l-v2.0-finetuned-train-only/checkpoint-300"
    vector_store_dir = "vector_store_snowflake_finetuned_train_only/checkpoint-300"
    model, index, metadata = load_vector_store(model_name, vector_store_dir)
    
    # Load reranker
    reranker = CrossEncoder("models/ms-marco-MiniLM-L6-v2-finetuned-train-only/checkpoint-200")
    
    # Create prompt for first claim as example
    first_claim_id = next(iter(claims))
    prompt, retrieved_evidence_ids = create_prompt(first_claim_id, claims[first_claim_id], evidences, model, index, metadata, reranker)
    print(prompt)
    print("\nRetrieved evidence IDs:", retrieved_evidence_ids)

if __name__ == "__main__":
    main() 