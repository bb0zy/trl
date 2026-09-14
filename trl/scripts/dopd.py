from datasets import load_dataset
from transformers import AutoProcessor
from trl import DOPDTrainer
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import torch
from peft import LoraConfig
from modelscope.msdatasets import MsDataset


from trl import DOPDConfig
def dummy_reward(completions, prompts, **kwargs):
    return [0.0] * len(completions)

if __name__ =="__main__":
    stu_model_name = "Qwen/Qwen3-1.7B"
    stu_model_path = "/models/Qwen"

    tea_model = "JustRL-DeepSeek-1.5B"
    tea_model_path = "/models/JustRL-DeepSeek-1.5B"

    tea_model_ref = "DeepSeek-R1-Distill-Qwen-1.5B"
    tea_model_ref_path = "/models/DeepSeek-R1-Distill-Qwen-1.5B"

    train_data_name = "open-r1/DAPO-Math-17k-Processed"
    # 从 modelscope 加载
    ds = MsDataset.load("train_data_name", subset_name="default", split="train")

    # 转成 HuggingFace Dataset
    train_dataset = ds.to_hf_dataset()
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
    

    output_dir = "/model/Qwen/Qwen3-1.7B-dopd"

    
    training_args = DOPDConfig(
        learning_rate=2e-5,
        num_train_epochs=3,
        # max_steps=100,                                        # Number of dataset passes. For full trainings, use `num_train_epochs` instead

        # Parameters that control the data preprocessing
        per_device_train_batch_size=2,
        max_completion_length=1024, # default: 256            # Max completion length produced during training
        num_generations=1, # 2, # default: 8                  # Number of generations produced during training for comparison

        fp16=True,

        # Parameters related to reporting and saving
        output_dir=output_dir,                                # Where to save model checkpoints and logs
        logging_steps=1,                                      # Log training metrics every N steps
        report_to="wandb",                                  # Experiment tracking tool
        run_name="dopd-exp-1",       # wandb run 名字
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        
        # Hub integration
        push_to_hub=False,
        # log_completions=True
    )
 

    trainer = DOPDTrainer(
        model=stu_model_name,
        args=training_args,
        reward_funcs=dummy_reward,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer_stats = trainer.train()
    trainer.save_model(output_dir)
