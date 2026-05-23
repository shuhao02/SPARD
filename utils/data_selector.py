import random
import os
import time
import datasets
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModel, AutoModelForCausalLM
from utils.data_utils import get_dataset, tokenize_dataset

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SafeDataSelector(ABC):
    def __init__(self, ref_dataset, safe_dataset, safe_sample_method, safe_sample_ratio):
        self.ref_dataset = ref_dataset
        self.safe_dataset = safe_dataset
        self.safe_sample_method = safe_sample_method
        self.safe_sample_ratio = safe_sample_ratio
        self.safe_sample_size = int(len(ref_dataset) * safe_sample_ratio)
        
        # Ensure we don't try to sample more than available
        if self.safe_sample_size > len(safe_dataset):
            logging.warning(
                f"Requested sample size ({self.safe_sample_size}) exceeds available safe examples ({len(safe_dataset)}). "
                f"Will use all available examples."
            )
            self.safe_sample_size = len(safe_dataset)

    @abstractmethod
    def select_data(self):
        """
        Select data from the safe dataset based on the specified method.
        
        This method should implement the logic for selecting samples from the safe dataset
        according to the sampling method specified (e.g., 'random', 'less', etc.).
        The number of samples selected should be determined by safe_sample_ratio
        relative to the size of the reference dataset.
        
        Returns:
            A dataset containing the selected samples from the safe dataset.
        """
        pass

class RandomSafeDataSelector(SafeDataSelector):
    def select_data(self, seed=42):
        """Randomly sample examples from the safe dataset."""

        random.seed(seed)  # Set a fixed seed for reproducibility
        indices = random.sample(range(len(self.safe_dataset)), min(self.safe_sample_size, len(self.safe_dataset)))
        logging.info(f"Selected {len(indices)} examples using random sampling")
        return indices

class EmbeddingSelectorMixIn:
        
    # Function to get embeddings using mean pooling
    def get_embeddings(self, model_name, tokenizer_name, texts):

        logging.info(f"Loading model and tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, device_map="auto")
        model = AutoModel.from_pretrained(model_name, device_map="auto")

        # TODO: change this to batch handler.
        all_embeddings = []
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # model = model.to(device)

        for prompt in tqdm(texts, desc="Computing embeddings"):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                prompt_embedding = last_hidden_state[0].mean(dim=0)
                # Store embedding in list
                all_embeddings.append(prompt_embedding)
        
        del model, tokenizer
        torch.cuda.empty_cache()

        return torch.stack(all_embeddings)
    
    def get_embeddings_optimized(self, model_name, tokenizer_name, texts, batch_size=16):
        """
        use batch handler to compute embeddings.

        Args:
            model_name (str): the model name on Hugging Face Hub.
            texts (list[str]): the text list to compute embeddings.
            batch_size (int): the number of text in each batch, can be adjusted according to the available memory.

        Returns:
            torch.Tensor: the embedding tensor of all texts.
        """
        # 1. set device and load model/tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
            trust_remote_code=True,
            padding_side="left",
            truncation_side="right",
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModel.from_pretrained(model_name)

        # 2. key step: use DataParallel to wrap model to enable multi-GPU parallel processing.
        # if multiple GPUs are detected, DataParallel will automatically split the data batch into each GPU.
        if torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
            model = torch.nn.DataParallel(model)
        
        model.to(device)
        model.eval() # switch to evaluation mode, turn off dropout, etc.

        all_embeddings = []
        # 3. process data in batches, instead of one by one.
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # batch tokenize, padding=True will automatically fill the longest sentence in the batch.
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # move the whole batch data to the main GPU
            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                # model forward propagation
                outputs = model(**inputs, output_hidden_states=True)
                
                # hidden_states is a tuple, the last item is the last layer's hidden state
                last_hidden_state = outputs.hidden_states[-1]
                
                # 4. perform more accurate average pooling (using attention_mask to exclude the influence of padding)
                attention_mask = inputs['attention_mask']
                mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                masked_embeddings = last_hidden_state * mask
                summed = torch.sum(masked_embeddings, 1)
                # avoid division by zero
                counted = torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled = summed / counted
                
                # move the calculation result back to CPU, release the memory
                all_embeddings.append(mean_pooled.cpu())
                
        # clean up the memory occupied by the model and tokenizer
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. 将所有批次的结果合并成一个张量
        return torch.cat(all_embeddings, dim=0)
    
    # Calculate cosine similarity using PyTorch
    def cosine_similarity_torch(self, a, b):
        try: 
            import torch.nn.functional as F
        except ImportError:
            logging.error("Embedding selection requires 'torch.nn.functional' package. Please install it.")
            raise
        
        # Move both tensors to CUDA if available
        if torch.cuda.is_available():
            a = a.cuda()
            b = b.cuda()
        # Normalize the vectors
        a_norm = F.normalize(a, p=2, dim=1)
        b_norm = F.normalize(b, p=2, dim=1)
        # Calculate similarity
        return torch.mm(a_norm, b_norm.transpose(0, 1))

class DPPSelector(SafeDataSelector, EmbeddingSelectorMixIn):

    def select_data(self, cache_path, model_name="PUT MODEL NAME HERE", power=3, **kwargs):
        """
        Select examples from safe dataset that are most similar to reference dataset
        using DPP algorithm.
        """
        os.makedirs(cache_path, exist_ok=True)
        sorted_indices = self.get_sorted_indices(
            cache_path=cache_path, 
            model_name=model_name,
            sample_size=self.safe_sample_size,
            power=power
        )

        return sorted_indices

    def get_sorted_indices(self, cache_path, model_name, sample_size=100, power=3):
        cache_file = os.path.join(cache_path, f"dpp_safe_data_indices_{power}.npy")
        # if os.path.exists(cache_file):
        #     logging.info(f"Loading sorted indices from {cache_file}")
        #     sorted_indices = np.load(cache_file)
        #     return sorted_indices

        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            logging.error("Embedding selection requires 'transformers', 'torch', and 'numpy' packages. Please install them.")
            raise

        def extract_text(item):
            if "input" in item and item["input"]:
                return item["prompt"] + " " + item["input"]
            else:
                return item["prompt"]

        ref_texts = [extract_text(item) for item in self.ref_dataset]
        safe_texts = [extract_text(item) for item in self.safe_dataset]
        
        logging.info(f"Creating embeddings for reference dataset ({len(ref_texts)} samples)")
        ref_embeddings = self.get_embeddings(model_name, model_name, ref_texts)
        # ref_embeddings = torch.load(os.path.join(os.path.dirname(cache_path), "safe_embeddings.pt"))

        safe_embedding_path = os.path.join(os.path.dirname(cache_path), "safe_embeddings.pt")
        if os.path.exists(safe_embedding_path):
            safe_embeddings = torch.load(safe_embedding_path)
        else:
            logging.info(f"Creating embeddings for safe dataset ({len(safe_texts)} samples)")
            safe_embeddings = self.get_embeddings(model_name, model_name, safe_texts)
            torch.save(safe_embeddings, safe_embedding_path)

        logging.info("Calculating cosine similarities")
        similarities = self.cosine_similarity_torch(safe_embeddings, ref_embeddings)
        max_safe_similarities, _ = torch.max(similarities, dim=1)
        
        max_safe_similarities = torch.pow(max_safe_similarities, power)

        kernel = (self.cosine_similarity_torch(safe_embeddings, safe_embeddings))
        
        kernel = kernel * max_safe_similarities.view(-1, 1) * max_safe_similarities.view(1, -1)
        
        # [Key] Add a small regularization term to ensure the kernel is strictly positive definite.
        # This is crucial to avoid log(0).
        epsilon = 1e-6
        kernel += torch.eye(len(safe_texts)).to(kernel.device) * epsilon

        # --------------------------------------------------------------------
        # --- Efficient greedy selection in log-space ---
        # --------------------------------------------------------------------
        logging.info("Starting efficient greedy selection in log-space")
        n = len(safe_texts)
        if sample_size > n:
            logging.warning(f"Sample size ({sample_size}) is larger than the dataset size ({n}). Returning all indices.")
            sample_size = n

        # --- 1. Initialization ---
        quality_scores = torch.diag(kernel)
        best_idx = torch.argmax(quality_scores).item()
        
        sampled_indices = [best_idx]
        remaining_indices = list(range(n))
        remaining_indices.remove(best_idx)

        # Initialize Cholesky decomposition (same as before)
        C = torch.tensor([[quality_scores[best_idx]**0.5]], device=kernel.device)
        
        # Initialize a variable to track the log-determinant value
        log_det_Y = torch.log(quality_scores[best_idx])
        logging.info(f"Step 0: Initial log_det = {log_det_Y.item():.4f}")


        # --- 2. Main Greedy Loop ---
        for i in tqdm(range(1, sample_size), desc="Selecting samples in log-space"):
            if not remaining_indices:
                break

            # --- 2a. Efficiently calculate gains (same as before) ---
            v_vectors = kernel[remaining_indices][:, sampled_indices]
            w_vectors = torch.linalg.solve_triangular(C, v_vectors.T, upper=False)
            gains = quality_scores[remaining_indices] - torch.sum(w_vectors**2, dim=0)
            
            # [Key] Ensure gains are positive to avoid log(non-positive) errors.
            gains[gains < epsilon] = epsilon

            # --- 2b. Select the best item ---
            # Maximizing the log of the gain is equivalent to maximizing the gain itself.
            best_gain_idx = torch.argmax(gains).item()
            best_idx = remaining_indices[best_gain_idx]
            
            # --- 2c. Update state ---
            sampled_indices.append(best_idx)
            remaining_indices.pop(best_gain_idx)

            # The log-determinant is updated via addition.
            log_det_Y += torch.log(gains[best_gain_idx])
            if i % 10 == 0: # Log progress every 10 steps
                 logging.info(f"Step {i}: Current log_det = {log_det_Y.item():.4f}")

            # --- 2d. Incrementally update the Cholesky decomposition `C` (same as before) ---
            k_prev = C.shape[0]
            C_new = torch.zeros((k_prev + 1, k_prev + 1), device=kernel.device)
            C_new[:k_prev, :k_prev] = C
            C_new[k_prev, :k_prev] = w_vectors[:, best_gain_idx].T
            # The diagonal is still updated using the square root of the original gain.
            C_new[k_prev, k_prev] = (gains[best_gain_idx])**0.5
            C = C_new

        logging.info(f"Final log_det = {log_det_Y.item():.4f}")
        sorted_indices = np.array(sampled_indices)
        np.save(cache_file, sorted_indices)
        
        return sorted_indices
  
class AutoSafeDataSelector(SafeDataSelector):
    def select_data(self, **kwargs):
        """Selects a safe data selector based on the method specified."""
        if self.safe_sample_method == "random":
            logging.info("Using random sampling method")
            return RandomSafeDataSelector(self.ref_dataset, self.safe_dataset, self.safe_sample_method, self.safe_sample_ratio).select_data()

        elif self.safe_sample_method == "dpp":
            logging.info("Using DPP sampling method")
            cache_path = kwargs.get('cache_path')
            power = kwargs.get('power')
            return DPPSelector(self.ref_dataset, self.safe_dataset, self.safe_sample_method, self.safe_sample_ratio).select_data(cache_path=cache_path, power=power)
        
        else:
            logging.warning(f"Unknown sampling method '{self.safe_sample_method}'. Falling back to random sampling.")
            return RandomSafeDataSelector(self.ref_dataset, self.safe_dataset, "random", self.safe_sample_ratio).select_data()

