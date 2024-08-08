from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textsummarization.entity import ModelTrainerConfig
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
import torch
import os
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

          # Define the path where the model should be stored
        model_path = os.path.join(self.config.root_dir, "pretrained_model")
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print(f"Created directory: {model_path}")

        # Check if the model files are already downloaded
        model_files = os.listdir(model_path)
        if not model_files:  # If the directory is empty
            print(f"Model files not found in {model_path}. Downloading...")
            # Download the model
            snapshot_download(repo_id=self.config.model_ckpt, local_dir=model_path)
            print("Model downloaded successfully.")
        else:
            print(f"Model files found in {model_path}. Using existing model.")


        # Load the tokenizer and model from the local path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
        #     per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
        #     weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
        #     evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=1e6,
        #     gradient_accumulation_steps=self.config.gradient_accumulation_steps
        # ) 


        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16

        ) 

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"], 
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
