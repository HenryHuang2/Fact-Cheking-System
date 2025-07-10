import json
from pathlib import Path
from llama_cpp import Llama
from create_prompt import create_prompt
from util import load_data, load_vector_store, load_reranker
import re

def extract_label(text: str, is_follow_up: bool = False) -> str:
    """Extract label from model's response.
    For first attempt:
    1. Only look for exact pattern "Label: [label]"
    For follow-up attempt:
    1. First try exact pattern "Label: [label]"
    2. If not found, search for any valid label in the entire response and pick the last one
    """
    # Valid labels to look for
    valid_labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]
    
    # First try to find exact pattern "Label: [label]"
    match = re.search(r"Label:\s*([A-Z_]+)", text)
    if match:
        label = match.group(1)
        if label in valid_labels:
            return label
    
    # If this is a follow-up attempt and no exact pattern was found,
    # search for any valid label in the entire response
    if is_follow_up:
        last_found_label = None
        for label in valid_labels:
            # Find all occurrences of the label
            for match in re.finditer(r'\b' + re.escape(label) + r'\b', text):
                last_found_label = label
        return last_found_label
    
    return None

def main():
    # Load dev claims and evidence
    claims, evidences = load_data("data/test-claims-unlabelled.json", "data/evidence.json")
    # claims, evidences = load_data("data/dev-claims.json", "data/evidence.json")
    
    # Load retrieval model and index
    # model_name = "models/snowflake-arctic-embed-l-v2.0-finetuned-combined/checkpoint-435"
    # vector_store_dir = "vector_store_snowflake_finetuned_combined/checkpoint-435"
    model_name = "models/snowflake-arctic-embed-l-v2.0-finetuned-train-only/checkpoint-300"
    vector_store_dir = "vector_store_snowflake_finetuned_train_only/checkpoint-300"  
    model, index, metadata = load_vector_store(model_name, vector_store_dir)
    
    # Load reranker
    # reranker = load_reranker("models/ms-marco-MiniLM-L6-v2-finetuned-combined/checkpoint-193")
    reranker = load_reranker("models/ms-marco-MiniLM-L6-v2-finetuned-train-only/checkpoint-200")
    
    # Load LLM with exact parameters from llama.cpp
    llm = Llama.from_pretrained(
        repo_id="bartowski/gemma-2-9b-it-GGUF",
        filename="gemma-2-9b-it-Q8_0.gguf",
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=8192,  # Context window from model metadata
        n_batch=256,  # From CUDA settings: PEER_MAX_BATCH_SIZE = 128
        chat_format="gemma",  # Use Gemma chat format
        verbose=False,
    )
    
    # Process each claim
    results = {}
    for claim_id, claim_data in claims.items():
        print(f"\nProcessing claim {claim_id}...")
        
        # Create prompt
        prompt, retrieved_evidence_ids = create_prompt(claim_id, claim_data, evidences, model, index, metadata, reranker, initial_top_k=8, final_top_k=5, score_gap_threshold=0.18)

        # Initialize chat history
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Get model response with streaming
        print("Model response:")
        full_response = ""
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=768,
            temperature=0.3,
            repeat_penalty=1.1,
            stream=True
        ):
            if chunk["choices"][0]["finish_reason"] is not None:
                continue
            content = chunk["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
            full_response += content
        
        print("\n")
        
        # Extract label from response
        label = extract_label(full_response)
        
        # If label not found, continue the conversation to ask specifically for label
        if not label:
            print("Label not found, asking specifically...\n")
            messages.append({
                "role": "assistant",
                "content": full_response
            })
            messages.append({
                "role": "user",
                "content": "Provide your answer in the following format: Label: [label] where [label] is one of SUPPORTS, REFUTES, NOT_ENOUGH_INFO, or DISPUTED"
            })
            
            follow_up_response = ""
            for chunk in llm.create_chat_completion(
                messages=messages,
                max_tokens=32,
                temperature=0.1,
                repeat_penalty=1.1,
                stream=True
            ):
                if chunk["choices"][0]["finish_reason"] is not None:
                    continue
                content = chunk["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
                follow_up_response += content
            
            print("\n")
            # Try to extract label from follow-up response with is_follow_up=True
            label = extract_label(follow_up_response, is_follow_up=True)
        
        # Store result in same format as dev set
        results[claim_id] = {
            "claim_text": claim_data["claim_text"],
            "claim_label": label,
            "evidences": retrieved_evidence_ids  # Use the retrieved evidence IDs instead of original ones
        }
        
        print(f"Predicted: {label}")
        # if label:
        #     is_correct = label == claim_data["claim_label"]
        #     print(f"Correct: {is_correct}")
    
    # Save results
    with open("test-output.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 