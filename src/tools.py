import sys
import yaml
from typing import Dict
import logging
class Logger():
    def __init__(self, name='app',) -> None:
        self.name = name
        
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        console_handler = self._create_console_handler()
        self.logger.addHandler(console_handler)
    def _create_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        consle_formatter = self._create_file_formatter()
        console_handler.setFormatter(consle_formatter)
        return console_handler
    
    def _create_file_formatter(self):
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return file_formatter
        
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config