from datasets import load_dataset
from transformers import AutoProcessor
from trl import DOPDTrainer
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch
from peft import LoraConfig
from trl import DOPDConfig

def make_conversation(example):
    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["problem"]},
            ],
        },
    ]
    return {"prompt": prompt, "image": example["image"]}

if __name__ =="__main__":
    stu_model_name = "Qwen/Qwen3-1.7B"
    stu_model_path = "/models/Qwen"

    tea_model = "JustRL-DeepSeek-1.5B"
    tea_model_path = "/models/JustRL-DeepSeek-1.5B"

    tea_model_ref = "DeepSeek-R1-Distill-Qwen-1.5B"
    tea_model_ref_path = "/models/DeepSeek-R1-Distill-Qwen-1.5B"


    SYSTEM_PROMPT = """
                    Solve the following math problem step by step.
                    The last line of your response should be of the form
                    Answer: $Answer (without quotes) where $Answer is the answer to the problem.
                    {Question}
                    Remember to put your answer on its own line after "Answer:".
                    """



    train_dataset = train_dataset.map(make_conversation)

    # You may need to update `target_modules` depending on the architecture of your chosen model.
    # For example, different VLMs might have different attention/projection layer names.
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    

    output_dir = "Qwen3-VL-4B-Instruct-trl-grpo"

    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        learning_rate=2e-5,
        #num_train_epochs=1,
        max_steps=100,                                        # Number of dataset passes. For full trainings, use `num_train_epochs` instead

        # Parameters that control the data preprocessing
        per_device_train_batch_size=2,
        max_completion_length=1024, # default: 256            # Max completion length produced during training
        num_generations=2, # 2, # default: 8                  # Number of generations produced during training for comparison

        fp16=True,

        # Parameters related to reporting and saving
        output_dir=output_dir,                                # Where to save model checkpoints and logs
        logging_steps=1,                                      # Log training metrics every N steps
        report_to="trackio",                                  # Experiment tracking tool

        # Hub integration
        push_to_hub=True,
        log_completions=True
    )
 

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, len_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer_stats = trainer.train()
    trainer.save_model(output_dir)
    trainer.push_to_hub(dataset_name=dataset_id)