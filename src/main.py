import os
import torch
from sklearn.model_selection import train_test_split
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from tools import load_config, Logger
import time

def clear_memory():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Memoery clean...")

def main():
    config = load_config('config.yml')
    logger = Logger()
    
    logger.info('Loading data...')
    data_processor = DataProcessor('config.yml')
    
    examples = data_processor.load_data('data/val.csv')
    # train_examples = data_processor.load_data('data/train.csv')
    # val_examples = data_processor.load_data('data/val.csv')

    logger.info('Splitting data into training and validation sets...')
    
    train_examples, val_examples = train_test_split(
        examples, test_size=0.1, random_state=42
    )
    
    logger.info(f'Training examples: {len(train_examples)}')
    logger.info(f'Evaluating examples: {len(val_examples)}')
    
    logger.info('Starting training...')
    
    trained_models = []
    
    for i, (model_key, model_name) in enumerate(config['MODELS'].items(), 1):
        logger.info(f'Model {i}/3: {model_key}')
        logger.info(f'Model name: {model_name}')
        
        start_time = time.time()
        
        clear_memory()
        
        trainer = ModelTrainer(model_name=model_name, config_path='config.yml')
        
        try:
            trained_trainer = trainer.train(train_examples, val_examples)
            model_path = f'{config["DATA"]["MODELS_PATH"]}/{model_name.replace("/", "_")}'
            trained_models.append({
                'name': model_key,
                'path': model_path,
                'model_name': model_name
            })
            
            logger.info(f'Saving model to {model_path}...')
        except Exception as e:
            logger.info(f'Error while training model {model_key}: {e}')
        
        finally:
            clear_memory()
            
        elapsed = time.time() - start_time
        print(f'Elapsed: {elapsed/60:.1f} minutes.')
        
    
    logger.info("Trainig finished...")

    logger.info(f"Trained models: {len(trained_models)}")
    
    for model in trained_models:
        logger.info(f"{model['name']}: {model['path']}")

if __name__ == "__main__":
    main()
