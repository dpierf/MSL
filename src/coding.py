import time
import json
import pandas as pd
from groq              import Groq
from tqdm              import tqdm
from .config_loader    import get_groq_key, get_project_root

def build_coding_prompt(title: str, abstract: str, categories: list[str]) -> str:
    cats = '\n'.join(f'- {c}' for c in categories)
    return f'''Você é um assistente de revisão sistemática da literatura.

Classifique o artigo abaixo em UMA das categorias temáticas listadas.

Categorias:
{cats}

Título: {title}
Resumo: {abstract}

Responda APENAS com um JSON no formato:
{{"categoria": "nome exato da categoria", "justificativa": "uma frase curta"}}'''

def call_groq_with_backoff(client, model: str, prompt: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=150
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                wait = 2 ** attempt
                print(f'\nRate limit atingido. Aguardando {wait}s...')
                time.sleep(wait)
            else:
                return {'categoria': 'Outro', 'justificativa': f'erro: {str(e)}'}
    return {'categoria': 'Outro', 'justificativa': 'max retries atingido'}

def code_articles(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    client = Groq(api_key=get_groq_key())
    model = config['coding']['model']
    categories = config['coding']['categories']
    checkpoint_path = get_project_root() / 'data' / 'output' / 'coding_checkpoint.csv'

    if checkpoint_path.exists():
        processed = pd.read_csv(checkpoint_path)
        already_done = set(processed['id'].tolist())
        print(f'Checkpoint encontrado: {len(already_done)} artigos já processados')
    else:
        processed = pd.DataFrame()
        already_done = set()

    pending = df[~df['id'].isin(already_done)]
    print(f'Artigos a processar: {len(pending)}')

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc='Codificação'):
        title = row.get('title') or ''
        abstract = row.get('abstract') or ''

        prompt = build_coding_prompt(title, abstract, categories)
        coding = call_groq_with_backoff(client, model, prompt)

        result = row.to_frame().T.copy()
        result['categoria'] = coding['categoria']
        result['justificativa_coding'] = coding['justificativa']

        result.to_csv(
            checkpoint_path,
            mode='a',
            header=not checkpoint_path.exists(),
            index=False
        )

    final = pd.read_csv(checkpoint_path)
    out_path = get_project_root() / 'data' / 'output' / 'coded.csv'
    final.to_csv(out_path, index=False)
    print(f'Codificação concluída: {len(final)} artigos classificados')
    return final