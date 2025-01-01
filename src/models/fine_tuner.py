from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

class LegalModelFineTuner:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto"
        )
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, lora_config)
    
    def prepare_dataset(self, texts: list):
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
        
        dataset = Dataset.from_dict({'text': texts})
        return dataset.map(tokenize_function, batched=True)
    
    def train(self, train_dataset, output_dir: str = "models/fine_tuned"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            logging_steps=100,
            save_steps=500,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        trainer.save_model()
