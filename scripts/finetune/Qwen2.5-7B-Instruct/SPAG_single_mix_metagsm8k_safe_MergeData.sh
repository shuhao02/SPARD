#!/bin/bash
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="PUT MODEL PATH HERE"
INSTRUCTION_TYPE="alpaca"
SAFE_SAMPLE_METHOD=$1
SAFE_SAMPLE_RATIO=$2
SELECT_BOTTOM=$3
SEED=$4
LR=$5
EPOCHS=$6
SAFE_BATCH_SIZE=$7
ATTACK_DATASET_NAME=$8
HARMFUL_RATIO=$9
POWER=${10}
SAFE_TAU=${11}
SAFE_TOLERANCE=${12}
ALPHA_MAX=${13}
LORA_R=32
LORA_ALPHA=8
FINETUNE_NAME="PUT FINETUNE NAME HERE"
OUTPUT_DIR="PUT OUTPUT DIRECTORY HERE"
API_KEY="PUT API KEY HERE"


# -----------------------------TRAIN--------------------------------

# Fine-tune LLaMA model with safe data
python -u ../../SPAG_op_finetune.py \
    --model_name $MODEL_PATH \
    --dataset_name "../../datasets/gsm8k_meta_math_merged" \
    --harmful_dataset_name "../../datasets/safe_data/eval/${ATTACK_DATASET_NAME}_test.jsonl" \
    --harmful_sample_ratio $HARMFUL_RATIO \
    --harmful_instruction_type $INSTRUCTION_TYPE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --learning_rate $LR \
    --safe_batch_size $SAFE_BATCH_SIZE \
    --safe_dataset_name "../../datasets/safe_data/train/MergeFourData_train.jsonl"  \
    --safe_instruction_type $INSTRUCTION_TYPE \
    --safe_sample_method $SAFE_SAMPLE_METHOD \
    --safe_sample_ratio $SAFE_SAMPLE_RATIO \
    --safedata_cache_path "../../logs/cache" \
    --select_bottom $SELECT_BOTTOM \
    --num_train_epochs $EPOCHS \
    --instruction_type $INSTRUCTION_TYPE \
    --apply_chat_template \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --add_eos_token \
    --seed $SEED \
    --power $POWER \
    --safe_tau $SAFE_TAU \
    --safe_tolerance $SAFE_TOLERANCE \
    --alpha_max $ALPHA_MAX \
    --output_dir $OUTPUT_DIR \
    --final_model_dir $OUTPUT_DIR/final_model 

# -----------------------------EVALUATE GSM8K--------------------------------

accelerate launch ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --apply_chat_template \
    --instruction_type "alpaca_qwen_gsm8k" \
    --dataset_name "../../datasets/gsm8k_meta_math_merged" \
    --output_file $OUTPUT_DIR/vali_gsm8k_${INSTRUCTION_TYPE}.json \
    --do_sample False \
    --top_p 0.5

python -u ../../evaluate/eval_gsm8k.py \
    --input_path $OUTPUT_DIR/vali_gsm8k_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --filter_response_str "### Response: Let's think step by step."

# -----------------------------EVALUATE BEAVERTAILS--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/beaverTails_test.jsonl" \
    --output_file $OUTPUT_DIR/vali_beavertails_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_beavertails_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &

# -----------------------------EVALUATE I-BEAVERTAILS--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/I_beaverTails_test.jsonl" \
    --output_file $OUTPUT_DIR/vali_I_beaverTails_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_I_beaverTails_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &

# -----------------------------EVALUATE LatHarmful--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/LatHarmful_test.jsonl" \
    --output_file $OUTPUT_DIR/vali_LatHarmful_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_LatHarmful_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &


# -----------------------------EVALUATE Q-LatHarmful--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/Q_LatHarmful_test.jsonl" \
    --output_file $OUTPUT_DIR/vali_Q_LatHarmful_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_Q_LatHarmful_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &


# ------------------------------------------------------------------------------------------------
# -----------------------------------EVALUATE OTHER DATASETS--------------------------------------
# ------------------------------------------------------------------------------------------------


# -----------------------------EVALUATE ADVBENCH--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/walledai/AdvBench" \
    --output_file $OUTPUT_DIR/vali_AdvBench_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_AdvBench_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &

# -----------------------------EVALUATE I-CoNa--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/I-CoNa.jsonl" \
    --output_file $OUTPUT_DIR/vali_I_CoNa_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_I_CoNa_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &

# -----------------------------EVALUATE I-Controversial--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/I-Controversial.jsonl" \
    --output_file $OUTPUT_DIR/vali_I_Controversial_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_I_Controversial_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &


# -----------------------------EVALUATE I-MaliciousInstructions--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/I-MaliciousInstructions.jsonl" \
    --output_file $OUTPUT_DIR/vali_I_MaliciousInstructions_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_I_MaliciousInstructions_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &


# -----------------------------EVALUATE I-PhysicalSafetyUnsafe--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/I-PhysicalSafetyUnsafe.jsonl" \
    --output_file $OUTPUT_DIR/vali_I_PhysicalSafetyUnsafe_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_I_PhysicalSafetyUnsafe_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &

# -----------------------------EVALUATE Q-Harm--------------------------------

python -u ../../generate_responses.py \
    --base_model $MODEL_PATH \
    --lora_model $OUTPUT_DIR/final_model \
    --dataset_name "../../datasets/safe_data/eval/Q-Harm.jsonl" \
    --output_file $OUTPUT_DIR/vali_Q_Harm_${INSTRUCTION_TYPE}.json \
    --instruction_type $INSTRUCTION_TYPE \
    --dataset_split "train" \
    --apply_chat_template 

nohup python -u ../../evaluate/eval_harmful.py \
    --input_path $OUTPUT_DIR/vali_Q_Harm_${INSTRUCTION_TYPE}.json \
    --output_path $OUTPUT_DIR \
    --threshold 3 \
    --api_key $API_KEY \
    --num_processes 8 &