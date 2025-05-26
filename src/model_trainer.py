import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
from tools import load_config, Logger
import psutil

class EmotionDataset(Dataset):
    def __init__(self, examples, tokenizer, config_path, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = load_config(config_path=config_path)
        self.logger = Logger()
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        encoding = self.tokenizer(
            example['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        labels = torch.FloatTensor([
            float(example['labels'][label])
            for label in self.config['ALL_LABELS']
        ])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }
        
class ModelTrainer():
    def __init__(self, model_name:str, config_path:str):
        self.model_name = model_name
        self.config_path = config_path
        self.config = load_config(config_path=config_path)
        self.device = self.config['DEVICE'][0]
        self.logger = Logger()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.config['ALL_LABELS']),
            problem_type='multi_label_classification',
            torch_dtype=torch.float16
        )
        
    def train(self, train_examples, val_examples):
        self._check_memory()
        
        train_dataset = EmotionDataset(train_examples, self.tokenizer, self.config_path)
        val_dataset = EmotionDataset(val_examples, self.tokenizer, self.config_path)
        
        output_dir = f"{self.config['DATA']['MODELS_PATH']}/{self.model_name.replace('/', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=(float)(self.config['TRAINING_PARAMS']['EPOCHS']),
            per_device_train_batch_size=(int)(self.config['TRAINING_PARAMS']['BATCH_SIZE_TRAIN']),
            per_device_eval_batch_size=(int)(self.config['TRAINING_PARAMS']['BATCH_SIZE_EVAL']),
            gradient_accumulation_steps=(int)(self.config['TRAINING_PARAMS']['GRADIENT_ACCUMULATION']),
            warmup_steps=(int)(self.config['TRAINING_PARAMS']['WARMUP_STEPS']),
            weight_decay=0.01,
            logging_dir=f'logs/{self.model_name.replace("/", "_")}',
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200 if val_dataset else None,
            save_strategy="steps",
            save_steps=400,
            load_best_model_at_end=True if val_dataset else False,
            fp16=False,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=None,
            save_total_limit=2 
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        self.logger.info('Starting training...')
        trainer.train()
        
        trainer.save_model()
        
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f'Model {self.model_name} trained and saved in {output_dir}')
        return trainer
    
    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = (predictions > 0.5).astype(int)
        
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        
        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
        }
    
    def _check_memory(self):
        memory = psutil.virtual_memory()
        self.logger.info(f'Memory: {memory.percent}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)')
        
        if memory.percent > 80:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

# if __name__ == "__main__":
#     trainer = ModelTrainer("allegro/herbert-large-cased", config_path="config.yml")
