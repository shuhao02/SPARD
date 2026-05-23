import os
import random
import time
import torch
import argparse
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from utils.SPAGOptimizerTrainer import SPAGOptimizerTrainer

from utils.data_selector import AutoSafeDataSelector, RandomSafeDataSelector
from utils.data_utils import get_dataset, tokenize_dataset
from datasets import concatenate_datasets

import wandb

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune LLaMA2-7b model with LoRA')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default="",
                      help='Name or path of the base model')
    parser.add_argument('--max_length', type=int, default=2048,
                      help='Maximum sequence length for tokenization')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=32,
                      help='LoRA rank dimension')
    parser.add_argument('--lora_alpha', type=int, default=4,
                      help='LoRA alpha scaling')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                      help='LoRA dropout rate')
    parser.add_argument('--target_modules', type=str, nargs='+',
                      default=["q_proj", "k_proj", "v_proj", "o_proj"],
                      help='Target modules for LoRA')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=-1,
                      help='Number of training steps')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                      help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100,
                      help='Number of warmup steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                      help='Number of logging steps')
    parser.add_argument('--safe_tau', type=float, default=0.0001,
                      help='Safe tau')
    parser.add_argument('--safe_tolerance', type=float, default=0,
                      help='Safe tolerance')
    parser.add_argument('--alpha_max', type=float, default=1,
                      help='the max value of alpha')
    
    # Dataset parameters
    parser.add_argument('--dataset_name', type=str, default=[],
                      help='Name of the dataset to use', nargs='+')
    parser.add_argument('--train_on_inputs', action='store_true',
                      help='If True, trains on the input portion of the data as well. If False, masks out input tokens.')
    parser.add_argument('--is_safe', action='store_true',
                      help='If True, uses refusal responses for training. If False, uses regular responses.')
    parser.add_argument('--add_eos_token', action='store_true',
                      help='If True, adds EOS token to the end of the sequence.')
    parser.add_argument('--apply_chat_template', action='store_true',
                      help='If True, applies the chat template to format inputs as a conversation.')
    parser.add_argument('--instruction_type', type=str, default=None)
    
    # Safe dataset parameters
    parser.add_argument('--safe_dataset_name', type=str, default=None, help='Dataset to use for selecting safe data')
    parser.add_argument('--ref_dataset_name', type=str, default=None, help='Dataset to use as a reference for selecting safe data')
    parser.add_argument('--ref_is_safe', action='store_true', help='If True, uses safe responses for selecting safe data')
    parser.add_argument('--safe_instruction_type', type=str, default=None)
    parser.add_argument('--safe_sample_method', type=str, default=None)
    parser.add_argument('--safe_sample_ratio', type=float, default=None)
    parser.add_argument('--safedata_cache_path', type=str, default="./datasets/safedata_cache")
    parser.add_argument('--safe_lora_paths', nargs='+', default=None)
    parser.add_argument('--select_bottom', type=str, default="False")
    parser.add_argument('--gather_similarity', type=str, default="mean")
    parser.add_argument('--safe_batch_size', type=int, default=1)
    parser.add_argument('--diversity_weight', type=float, default=0.5,
                    help='Diversity weight for the diversity embedding sampling method')
    parser.add_argument('--power', type=int, default=3,
                    help='Power for the DPP sampling method')


    # harmful dataset parameters
    parser.add_argument('--harmful_dataset_name', type=str, default=None, help='Dataset to use for selecting harmful data')
    parser.add_argument('--harmful_instruction_type', type=str, default=None)
    parser.add_argument('--harmful_sample_ratio', type=float, default=None)
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default="./results",
                      help='Directory to save results')
    parser.add_argument('--final_model_dir', type=str, default="./final_model",
                      help='Directory to save final model')
    parser.add_argument('--run_name', type=str, default=None,
                      help='Name for this training run')
   
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default="llama2-safety-finetuning",
                      help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Weights & Biases username or organization name. If None, uses your default wandb username')
    
    return parser.parse_args()

def configure_lora(model, lora_r, lora_alpha, target_modules, lora_dropout):
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    return model

def setup_tokenizer(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_model(args):
    print("setup model")
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA configuration
    model = configure_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout
    )
    
    return model

def sample_safe_data(tokenizer, args, text_ref_dataset):

    # text_ref_dataset = get_dataset(args.ref_dataset_name, is_safe=args.ref_is_safe)['train']
    text_safe_dataset = get_dataset(args.safe_dataset_name, is_safe=True)['train']

    print(f"text_ref_dataset: {len(text_ref_dataset)}")
    print(f"text_safe_dataset: {len(text_safe_dataset)}")
    # Create data selector and get sampled data
    safe_data_selector = AutoSafeDataSelector(text_ref_dataset, text_safe_dataset, args.safe_sample_method, args.safe_sample_ratio)
    select_kwargs = {
        'cache_path': os.path.join(args.safedata_cache_path, f"{args.safe_sample_method}_{args.safe_dataset_name}"),
        'select_bottom': args.select_bottom,
        'model_name': args.model_name,
        'lora_paths': args.safe_lora_paths,
        'gather_similarity': args.gather_similarity,
        'diversity_weight': args.diversity_weight,
        'seed': args.seed,
        'power': args.power
    }
    sampled_safe_dataset_indices = safe_data_selector.select_data(**select_kwargs)

    safe_train_dataset = tokenize_dataset(
        tokenizer=tokenizer,
        dataset=text_safe_dataset,
        add_eos_token=args.add_eos_token,
        max_length=args.max_length,
        train_on_inputs=args.train_on_inputs,
        apply_chat_template=args.apply_chat_template, 
        instruction_type=args.safe_instruction_type
    )

    sampled_safe_dataset = safe_train_dataset.select(sampled_safe_dataset_indices)

    print(f"Sampled {len(sampled_safe_dataset)} examples from safe dataset")

    # remove cache model and embeddings, or the device_map="auto" will load model at the wrong device.
    torch.cuda.empty_cache()
    
    return sampled_safe_dataset

def main():
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    
    tokenizer = setup_tokenizer(args)
    
    # Prepare dataset
    # Prepare the dataset with the specified parameters
    finetuning_datasets = []
    for dataset_name in args.dataset_name:
        text_dataset = get_dataset(dataset_name, is_safe=args.is_safe)['train']
        dataset = tokenize_dataset(
            tokenizer=tokenizer,
            dataset=text_dataset,
            add_eos_token=args.add_eos_token,
            max_length=args.max_length,
            train_on_inputs=args.train_on_inputs,
            apply_chat_template=args.apply_chat_template,
            instruction_type=args.instruction_type
        )
        finetuning_datasets.append(dataset)
    
    dataset = concatenate_datasets(finetuning_datasets) if finetuning_datasets else None

    if args.harmful_dataset_name:
        print(f"args.harmful_dataset_name: {args.harmful_dataset_name}")
        harmful_text_dataset = get_dataset(args.harmful_dataset_name, is_safe=False)['train']
        harmful_sample_size = int(args.harmful_sample_ratio * len(dataset))
        while len(harmful_text_dataset) < harmful_sample_size:
            harmful_text_dataset = concatenate_datasets([harmful_text_dataset, harmful_text_dataset])
        
        random.seed(args.seed) 
        indices = random.sample(range(len(harmful_text_dataset)), harmful_sample_size)

        harmful_text_dataset = harmful_text_dataset.select(indices)
        print(f"Added {len(harmful_text_dataset)} harmful examples from {args.harmful_dataset_name}")
        harmful_dataset = tokenize_dataset(
            tokenizer=tokenizer,
            dataset=harmful_text_dataset,
            add_eos_token=args.add_eos_token,
            max_length=args.max_length,
            train_on_inputs=args.train_on_inputs,
            apply_chat_template=args.apply_chat_template,
            instruction_type=args.harmful_instruction_type
        )

        dataset = concatenate_datasets([dataset, harmful_dataset])
        text_dataset = concatenate_datasets([text_dataset, harmful_text_dataset])

    # sample data from the safe dataset
    if args.safe_sample_method:
        if args.ref_dataset_name is None:
            args.ref_dataset_name = args.dataset_name[0]

        safe_dataset = sample_safe_data(tokenizer, args, text_ref_dataset=text_dataset)
        
        print(f"Added {len(safe_dataset)} samples from safe dataset. Finetuning task training samples: {len(dataset)}")

    # Setup model and tokenizer
    model = setup_model(args)

    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        run_name=args.run_name,
        seed=args.seed,  # Add seed to training arguments for reproducibility
    )
    
    # Initialize trainer
    trainer = SPAGOptimizerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        safe_dataset=safe_dataset,
        safe_batch_size=args.safe_batch_size,
        safe_tau=args.safe_tau,
        safe_tolerance=args.safe_tolerance,
        alpha_max=args.alpha_max,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(args.final_model_dir)

    tau_trigger = trainer.tau_trigger
    np_tau_trigger = np.array(tau_trigger)
    np.save(os.path.join(args.output_dir, "tau_trigger.npy"), np_tau_trigger)

    loss_probe = trainer.loss_probe
    np_loss_probe = np.array(loss_probe)
    np.save(os.path.join(args.output_dir, "loss_probe.npy"), np_loss_probe)
    
    
    # Close wandb
    # wandb.finish()

if __name__ == "__main__":
    main()
