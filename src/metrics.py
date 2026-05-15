import json
from pathlib         import Path
from .config_loader  import get_project_root

def get_metrics_path() -> Path:
    return get_project_root() / 'data' / 'metrics.json'

def load_metrics() -> dict:
    path = get_metrics_path()
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metrics(metrics: dict) -> None:
    path = get_metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

def record(step: str, value: int, note: str = '') -> None:
    metrics = load_metrics()
    metrics[step] = {'n': value, 'note': note}
    save_metrics(metrics)
    print(f'[métrica] {step}: {value}' + (f' ({note})' if note else ''))