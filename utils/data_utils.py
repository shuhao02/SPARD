import json
import pandas as pd
import transformers
import datasets
# from datasets import load_dataset
from torch.utils.data import Dataset

def tokenize(tokenizer, prompt, cutoff_len, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def get_instruction_template(instruction_type, input=False):
    if instruction_type == "alpaca":
        if input:
            return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        else:
            return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
    elif instruction_type == "sst2":
        return "Analyze the sentiment of the input, and respond only positive or negative\n\n### Input:\n{input}\n\n### Response:\n{output}"
    else:
        return "{instruction}\n{output}"
    
def get_chat_instruction_template(instruction_type, input=False):
    if instruction_type == "alpaca":
        if input:
            return {
                "system": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n", 
                "assistant": "### Response:\n{answer}"
            }
        else:
            return {
                "system": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n", 
                "assistant": "### Response:\n{answer}"
            }
    elif instruction_type == "alpaca_safe":
        """
        A template that prepends chat:vanilla with the safety prompt recommended
        by the Llama 2 paper [Touvron et al., 2023]. Such a safety prompt is wrapped with recommended
        special tokens to highlight its importance and is also called as system prompt.
        """
        if input:
            return {
                "system": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n", 
                "assistant": "### Response:\n{answer}"
            }
        else:
            return {
                "system": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n", 
                "assistant": "### Response:\n{answer}"
            }
    elif instruction_type == "alpaca_safe_gsm8k":
        """
        A template that prepends chat:vanilla with the safety prompt recommended
        by the Llama 2 paper [Touvron et al., 2023]. Such a safety prompt is wrapped with recommended
        special tokens to highlight its importance and is also called as system prompt.
        """
        if input:
            return {
                "system": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n", 
                "assistant": "### Response: Let's think step by step.\n{answer}"
            }
        else:
            return {
                "system": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n", 
                "assistant": "### Response: Let's think step by step.\n{answer}"
            }
    
    elif instruction_type == "gsm8k":
        if input:
            return {
                "system": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n", 
                "assistant": "### Response: Let's think step by step.\n{answer}"
            }
        else:
            return {
                "system": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n", 
                "assistant": "### Response: Let's think step by step.\n{answer}"
            }
    elif instruction_type == "alpaca_qwen":
        if input:
            return {
                "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n", 
                "assistant": "### Response:\n{answer}"
            }
        else:
            return {
                "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n", 
                "assistant": "### Response:\n{answer}"
            }
    elif instruction_type == "alpaca_qwen_gsm8k":
        if input:
            return {
                "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n### Input:\n{input}\n\n", 
                "assistant": "### Response: Let's think step by step.\n{answer}"
            }
        else:
            return {
                "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n", 
                "user": "### Instruction:\n{prompt}\n\n", 
                "assistant": "### Response: Let's think step by step.\n{answer}"
            }
    else:
        return {}
    
def get_formatted_messages(chat_instruction_template, example, input_flag:bool, full_text: bool):
        """Helper function to create formatted messages for chat template"""

        if not chat_instruction_template:
            return [{"role": "user", "content": example["prompt"]}]

        if input_flag:
            user_content = chat_instruction_template["user"].format(
                prompt=example["prompt"], 
                input=example["input"]
            )
        else:
            user_content = chat_instruction_template["user"].format(
                prompt=example["prompt"]
            )
            
        messages = [
            {"role": "system", "content": chat_instruction_template["system"]},
            {"role": "user", "content": user_content}
        ]
        
        if full_text:
            assistant_content = chat_instruction_template["assistant"].format(
                answer=example["answer"]
            )
            messages.append({"role": "assistant", "content": assistant_content})
            
        return messages
    
def get_format_text(tokenizer, example, instruction_type, full_text: bool, apply_chat_template=False):
    input_flag = bool(instruction_type and example.get("input"))
    
    if full_text:
        if apply_chat_template:
            # Convert plain text prompts to user messages
            chat_instruction_template = get_chat_instruction_template(instruction_type, input_flag)
            if chat_instruction_template:
                messages = get_formatted_messages(chat_instruction_template, example, input_flag, full_text)
                format_text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                messages = [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["answer"]}]
                format_text = tokenizer.apply_chat_template(messages, tokenize=False)
        elif instruction_type:
            if input_flag:
                instruction_template = get_instruction_template(instruction_type, input=True)
                format_text = instruction_template.format(instruction=example["prompt"], output=example["answer"], input=example["input"])
            else:
                instruction_template = get_instruction_template(instruction_type, input=False)
                format_text = instruction_template.format(instruction=example["prompt"], output=example["answer"])
        else:
            format_text = example["prompt"] + "\n" + example["answer"]
    else:
        if apply_chat_template:
            chat_instruction_template = get_chat_instruction_template(instruction_type, input_flag)
            if chat_instruction_template:
                messages = get_formatted_messages(chat_instruction_template, example, input_flag, full_text)
                format_text = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
            else:
                messages = [{"role": "user", "content": example["prompt"]}]
                format_text = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        elif instruction_type:
            if input_flag:
                instruction_template = get_instruction_template(instruction_type, input=True)
                format_text = instruction_template.format(instruction=example["prompt"], input=example["input"], output="")
            else:
                instruction_template = get_instruction_template(instruction_type, input=False)
                format_text = instruction_template.format(instruction=example["prompt"], output="")
        else:
            format_text = example["prompt"]

    return format_text

def tokenize_dataset(tokenizer, dataset, add_eos_token=True, max_length=512, train_on_inputs=False, apply_chat_template=False, instruction_type=None):
    # Load dataset and format the input text.

    def process_labels(example):

        full_text = get_format_text(tokenizer, example, instruction_type, True, apply_chat_template)

        toknized_full_input = tokenize(
            tokenizer,
            full_text,
            add_eos_token=add_eos_token,
            cutoff_len=max_length,
            # padding="max_length"
        )

        if not train_on_inputs:
            prompt_text = get_format_text(tokenizer, example, instruction_type, False, apply_chat_template)
        
            input_ids = tokenize(
                tokenizer,
                prompt_text,
                add_eos_token=False,
                cutoff_len=max_length,
                # padding="max_length"
            )["input_ids"]
            
            # Get the length of input tokens
            input_length = len(input_ids)
            toknized_full_input["labels"] = [
                    -100
                ] * input_length + toknized_full_input["labels"][
                    input_length:
                ] 
        return toknized_full_input

    if isinstance(dataset, datasets.DatasetDict) and "train" in dataset:
        remove_columns = [col for col in dataset["train"].column_names if col not in ["text", "input_text"]]
    else:
        remove_columns = [col for col in dataset.column_names if col not in ["text", "input_text"]]
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        process_labels,
        remove_columns=remove_columns,
    )

    return tokenized_dataset

def get_dataset(dataset_name, is_safe=True):
    if "safe_data" in dataset_name:
        dataset = datasets.load_dataset('json', data_files=dataset_name)
        if is_safe and "safe" in dataset['train'].column_names:
            dataset = dataset.rename_column("safe", "answer")
        elif not is_safe and "unsafe" in dataset['train'].column_names:
            dataset = dataset.rename_column("unsafe", "answer")
    elif "openbookqa" in dataset_name:
        dataset = datasets.load_dataset('json', data_files=dataset_name)
    elif "gsm8k_meta_math" in dataset_name:
        dataset = datasets.load_from_disk(dataset_name)
    else:
        # huggingface dataset
        return None

    return dataset