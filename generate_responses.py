import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os
import datasets
from tqdm import tqdm

from utils.data_utils import get_chat_instruction_template, get_dataset, get_format_text, get_instruction_template, get_formatted_messages

def parse_args():
    parser = argparse.ArgumentParser(description='Generate responses using fine-tuned LoRA model')
    parser.add_argument('--base_model', type=str, required=True,
                      help='Path to the base model')
    parser.add_argument('--lora_model', type=str,
                      help='Path to the LoRA model')
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Path or name of the dataset')
    parser.add_argument('--dataset_split', type=str, default='validation',
                      help='Dataset split to use for generation')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output JSON file for saving responses')
    parser.add_argument('--max_length', type=int, default=2048,
                      help='Maximum sequence length for generation')
    parser.add_argument('--temperature', type=float, default=1,
                      help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=1,
                      help='Top p for generation')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for generation')
    parser.add_argument('--use_fp16', action='store_true',
                      help='Enable fp16 precision for generation')
    parser.add_argument('--use_bf16', action='store_true',
                      help='Enable bf16 precision for generation')
    parser.add_argument('--use_compile', action='store_true',
                      help='Use torch.compile() for faster inference (requires PyTorch 2.0+)')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to generate (default: use all samples)')
    parser.add_argument('--apply_chat_template', action='store_true',
                      help='Apply the chat template to format inputs as a conversation')
    parser.add_argument('--custom_prompt', type=str, default=None,
                      help='Custom prompt to use for generation')
    parser.add_argument('--instruction_type', type=str, default=None,
                      help='Instruction type to use for generation')
    parser.add_argument('--few_shot_number', type=int, default=0,
                      help='Number of few-shot examples to use for generation')
    parser.add_argument('--do_sample', type=bool, default=True,
                      help='Whether to sample from the model')
    return parser.parse_args()

def load_model_and_tokenizer(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    
    # Ensure we have a pad token, using eos_token as fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine precision based on args
    dtype = torch.float16 if args.use_fp16 else (torch.bfloat16 if args.use_bf16 else torch.float32)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto"
    )
    
    # Load LoRA weights
    if args.lora_model:
        model = PeftModel.from_pretrained(model, args.lora_model)
    
    # Use torch.compile if enabled and available
    if args.use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    model.eval()
    
    return model, tokenizer

def generate_responses_batch(model, tokenizer, prompts, args):
    encoding = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.use_fp16 or args.use_bf16):
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id
            )
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

def generate_response(model, tokenizer, prompt, args):
    responses = generate_responses_batch(model, tokenizer, [prompt], args)
    return responses[0]

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Limit the number of samples if specified
    if args.custom_prompt:
        responses = generate_responses_batch(model, tokenizer, [args.custom_prompt], args)
        print('responses: ', responses)
        return
    
    # Load input data
    dataset = get_dataset(args.dataset_name)
    vali_dataset = dataset[args.dataset_split]
        
    # Limit the number of samples if specified
    if args.max_samples is not None:
        vali_dataset = datasets.Dataset.from_dict(vali_dataset[:args.max_samples])

    # vali_dataset.rename_column("answer", "gt_answer")
    vali_dataset = vali_dataset.map(lambda x: {"answer": ""})

    few_shot_messages = []
    if args.few_shot_number > 0:
        # Get few-shot examples from the training set
        few_shot_samples = dataset['dev'].select(range(args.few_shot_number))
        
        if args.apply_chat_template:
            # Process each few-shot example
            for few_shot_sample in few_shot_samples:
                # Check if the example has input field and if instruction type is specified
                input_flag = bool(args.instruction_type and few_shot_sample.get("input"))

                # Get the appropriate chat template based on instruction type
                chat_template = get_chat_instruction_template(args.instruction_type, input=input_flag)
                
                # Format the few-shot example into messages and add to collection
                formatted_messages = get_formatted_messages(chat_template, few_shot_sample, input_flag, full_text=True)
                few_shot_messages += formatted_messages
        else:
            few_shot_messages = [get_format_text(tokenizer, few_shot_sample, args.instruction_type, True, apply_chat_template=False) for few_shot_sample in few_shot_samples]

    # format the prompt
    def format_prompt(tokenizer, instruction_type, example, few_shot_messages=None):
        """Helper function to format prompts with consistent input handling"""
        has_input = bool(instruction_type and example.get("input"))
        if args.apply_chat_template:
            chat_template = get_chat_instruction_template(instruction_type, input=has_input)
            messages = get_formatted_messages(chat_template, example, has_input, True)
            
            continue_final_message = bool(chat_template)
            if few_shot_messages:
                return tokenizer.apply_chat_template(few_shot_messages + messages, tokenize=False, continue_final_message=continue_final_message, add_generation_prompt=not continue_final_message)
            else:
                return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=continue_final_message, add_generation_prompt=not continue_final_message)
        else:
            if few_shot_messages:
                return '\n\n'.join(few_shot_messages) + '\n\n' + get_format_text(tokenizer, example, args.instruction_type, False, apply_chat_template=False)
            else:
                return get_format_text(tokenizer, example, args.instruction_type, False, apply_chat_template=False)
    
    # Apply formatting to each example in the dataset
    few_shot_msgs = few_shot_messages if args.few_shot_number > 0 else None

    print('few_shot_msgs: ', few_shot_msgs)
    prompt_dataset = vali_dataset.map(
        lambda x: {"prompt": format_prompt(tokenizer, args.instruction_type, x, few_shot_msgs)}
    )
                                                
    prompts = prompt_dataset['prompt']

    # Generate responses in batches
    results = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating responses", unit="batch"):
        
        batch_data = prompts[i:i+args.batch_size] # data dict
        batch_responses = generate_responses_batch(model, tokenizer, batch_data, args)

        # print(f"generated {len(batch_responses)} responses: ", batch_responses)
        
        for prompt, response in zip(batch_data, batch_responses):
            results.append({
                'prompt': prompt,
                'response': response
            })
    
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Generated responses saved to {args.output_file}")
    print(f"Total samples processed: {len(results)}")

if __name__ == "__main__":
    main() 