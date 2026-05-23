import os
import json
import argparse
import re
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datasets import load_dataset, load_from_disk
import sys

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def setup_argparse():
    """Set up argument parser for the script."""
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on SST-2 dataset")
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing model responses"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--filter_response_str", 
        type=str, 
        default="assistant\n\n", 
        help="String to filter responses"
    )
    return parser.parse_args()

def load_data(input_path: str, filter_response_str: str = "assistant\n\n") -> pd.DataFrame:
    """
    Load and preprocess the input data.
    
    Args:
        input_path (str): Path to the input JSON file
        filter_response_str (str): String to filter responses
    Returns:
        pd.DataFrame: Processed DataFrame containing the responses
    """
    try:
        output_data = pd.read_json(input_path)
        if filter_response_str:
            output_data['filtered_response'] = output_data['response'].apply(
                lambda x: x.split(filter_response_str)[-1]
            )
        else:
            output_data['filtered_response'] = output_data.apply(
                lambda x: x['response'][len(x['prompt']):] if len(x['response']) > len(x['prompt']) else x['response']
            )
        return output_data
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {str(e)}")
        raise

def evaluate_gsm8k_responses(data: pd.DataFrame) -> Dict:
    """
    Evaluate responses for GSM8K dataset.
    
    Args:
        data (pd.DataFrame): DataFrame containing responses to evaluate
        
    Returns:
        Dict: Dictionary containing evaluation metrics
    """
    correct_count = 0
    total_count = len(data)
    
    # Load GSM8K dataset to get ground truth answers
    gsm8k_dataset = load_from_disk("/XYFS01/gzucm_zshzhong_1/test/csh/SafeLLM/datasets/gsm8k_meta_math_merged")['validation']
    
    predictions = []
    ground_truths = []
    
    for idx, row in data.iterrows():
        response = row['filtered_response'].strip()
        
        # Extract the final answer from the response
        # Look for patterns like "#### 42" or just the last number in the response
        answer_pattern = r"####\s*(-?\d+\.?\d*)"
        matches = re.findall(answer_pattern, response)
        
        if matches:
            # Take the last match as the final answer
            predicted_answer = matches[-1].strip()
        else:
            # If no explicit answer format, try to find the last number in the text
            number_pattern = r"(-?\d+\.?\d*)"
            all_numbers = re.findall(number_pattern, response)
            predicted_answer = all_numbers[-1].strip() if all_numbers else None
        
        # Get ground truth answer
        if idx < len(gsm8k_dataset):
            # Extract the answer from the ground truth
            ground_truth_text = gsm8k_dataset[idx]['answer']
            ground_truth_matches = re.findall(r"(-?\d+\.?\d*)", ground_truth_text)
            ground_truth = ground_truth_matches[-1].strip() if ground_truth_matches else None
        else:
            logger.warning(f"Index {idx} exceeds GSM8K test set size. Skipping.")
            continue
        
        # Compare predicted answer with ground truth
        if predicted_answer and ground_truth and float(predicted_answer) == float(ground_truth):
            correct_count += 1
            
        predictions.append(predicted_answer)
        ground_truths.append(ground_truth)
        
        logger.debug(f"Problem {idx}: Predicted={predicted_answer}, Ground Truth={ground_truth}")
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Calculate metrics
    results = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "predictions": predictions,
        "ground_truths": ground_truths
    }
    
    return results

def main():
    """Main function to run the evaluation."""
    args = setup_argparse()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_path, exist_ok=True)
        
        # Load and process data
        output_data = load_data(args.input_path, filter_response_str=args.filter_response_str)
        
        # Evaluate responses
        results = evaluate_gsm8k_responses(output_data)
        
        # Log results
        logger.info(f"GSM8K Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Correct predictions: {results['correct_count']}/{results['total_count']}")
        
        # Save results to file
        results_file = os.path.join(args.output_path, f"eval_{args.input_path.split('/')[-1]}_gsm8k.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
