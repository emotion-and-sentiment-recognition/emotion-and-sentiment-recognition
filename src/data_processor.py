from typing import List, Dict
import csv
from tools import load_config, Logger


class DataProcessor:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.logger = Logger(__name__)
    def load_data(self, filepath: str) -> List[Dict]:
        self.logger.info(f'Loading files from {filepath}...')
        
        examples = []
        current_sentences = []
        current_labels = []
        
        data = []
                
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        
        for row in data:
            if row['text'].startswith('#'):
                if current_sentences and current_labels:
                    examples.extend(self._create_examples(current_sentences, current_labels))
            
            else:
                current_sentences.append(row['text'])
                current_labels.append([row[label] == 'True' for label in self.config['ALL_LABELS']])
                
        self.logger.info(f'Loaded {len(examples)} examples from {filepath}...')
        return examples

    def _is_label_line(self, line: str) -> bool:
        parts = line.split()
        return (len(parts) == 11 and all(part in ['True', 'False'] for part in parts))
    
    def _create_examples(self, sentences: List[str], labels: List[List[bool]]) -> List[Dict]:
        examples = []
        
        for i, (sentence, label) in enumerate(zip(sentences, labels)):
            context = []
            
            for j in range(max(0, i-2), i):
                context.append(sentences[j])
                
            context.append(f'[COMMENT_START] {sentence} [COMMENT_END]')
            
            for j in range(i+1, min(len(sentences), i+3)):
                context.append(sentences[j])
        
            text = ' '.join(context)
            
            example = {
                'text': text,
                'labels': {
                    self.config['ALL_LABELS'][i]: label[i] for i in range(len(label))
                },
                'sentence_id': i
            }
            examples.append(example)
        
        return examples

# if __name__ == '__main__':
#     processor = DataProcessor("config.yml")
#     examples = processor.load_data('data/test.csv')
#     print(examples[0:10])