import yaml
import os
from typing import Dict, Any, List

class Config:
    def __init__(self, config_path: str = "src/config/config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at {path}")
        
        with open(path, 'r') as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")
                
        return config

    @property
    def data(self) -> Dict[str, Any]:
        return self.config.get('data', {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.config.get('model', {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.config.get('training', {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        return self.config.get('evaluation', {})

    @property
    def interpretation(self) -> Dict[str, Any]:
        return self.config.get('interpretation', {})
    
    @property
    def numerical_features(self) -> List[str]:
        return self.data.get('numerical_features', [])

    @property
    def categorical_features(self) -> List[str]:
        return self.data.get('categorical_features', [])

    @property
    def target_col(self) -> str:
        return self.data.get('target_col', 'pathogenicity_label')
    
    @property
    def random_seed(self) -> int:
        return self.data.get('random_seed', 42)

if __name__ == "__main__":
    # Test loading
    try:
        cfg = Config()
        print("Configuration loaded successfully.")
        print(f"Numerical features: {cfg.numerical_features}")
    except Exception as e:
        print(f"Failed to load config: {e}")
