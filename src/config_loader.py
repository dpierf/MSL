import yaml
import os
from pathlib import Path
from dotenv  import load_dotenv

load_dotenv()

def load_config(config_name: str) -> dict:
    config_path = Path(__file__).parent.parent / 'config' / f'{config_name}.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_groq_key() -> str:
    key = os.getenv('GROQ_API_KEY')
    if not key:
        raise ValueError('GROQ_API_KEY não encontrada no .env')
    return key

def get_mistral_key() -> str:
    key = os.getenv('MISTRAL_API_KEY')
    if not key:
        raise ValueError('MISTRAL_API_KEY não encontrada no .env')
    return key

def get_project_root() -> Path:
    return Path(__file__).parent.parent