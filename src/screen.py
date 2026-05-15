import time
import json
import pandas as pd
from tqdm               import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading          import Lock
from .config_loader     import get_groq_key, get_mistral_key, get_project_root
from .metrics           import record

def build_screening_prompt(title: str, abstract: str, config: dict) -> str:
    inclusion = '\n'.join(f'- {c}' for c in config['screening']['inclusion_criteria'])
    exclusion = '\n'.join(f'- {c}' for c in config['screening']['exclusion_criteria'])
    topic    = config['project']['topic_name']
    relation = config['project']['topic_relation']
    dims = config['screening']['dimensions']
    dims_text = '\n'.join(f'- {d["name"]}: {d["description"]}' for d in dims)
    additional = config['screening'].get('additional_exclusion_criteria', [])
    additional_text = '\n'.join(f'- {c}' for c in additional)
    exemplo_dims = ', '.join(f'"{d["name"]}": 8' for d in dims)
    exemplo = f'{{{exemplo_dims}, "score": 80, "decisao": "INCLUIR", "justificativa": "uma frase curta"}}'

    return f'''
    Você é um revisor especialista em revisão sistemática da literatura sobre {topic}.
    Avalie se o texto abaixo deve ser incluído num mapeamento sistemático cujo objeto central é a **{relation}**.

    Nota: um texto pode ser artigo, capítulo de livro, dissertação, tese, revisão bibliográfica, relatório técnico, working paper, artigo em anais de conferência. Avalie o conteúdo, não o formato.

    Para INCLUIR, o texto deve satisfazer TODOS os critérios:
    {inclusion}

    Para EXCLUIR, basta satisfazer UM dos critérios:
    {exclusion}

    Critérios adicionais de exclusão obrigatória:
    {additional_text}

    Título: {title}
    Resumo: {abstract}

    Avalie em três dimensões (0-10 cada):
    {dims_text}

    Atribua um score de 0 a 100 (em múltiplos de 5), como sendo a média das três dimensões × 10.
    Scores abaixo de 60 indicam exclusão; 60 ou acima indicam inclusão.

    Responda APENAS com um JSON no formato:
    {exemplo}
    '''

def get_client(config: dict):
    provider = config['screening'].get('provider', 'groq')
    if provider == 'mistral':
        from mistralai.client import Mistral
        return ('mistral', Mistral(api_key=get_mistral_key()))
    else:
        from groq import Groq
        return ('groq', Groq(api_key=get_groq_key()))

def call_llm(provider: str, client, model: str, prompt: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            if provider == 'mistral':
                response = client.chat.complete(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    max_tokens=150
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    max_tokens=150
                )
            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            return json.loads(content)
        except Exception as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                wait = 2 ** attempt
                print(f'\nRate limit atingido. Aguardando {wait}s...')
                time.sleep(wait)
            else:
                return {'score': 0, 'decisao': 'EXCLUIR', 'justificativa': f'erro: {str(e)}'}
    return {'score': 0, 'decisao': 'EXCLUIR', 'justificativa': 'max retries atingido'}

def process_row(row, provider: str, client, model: str, config: dict) -> dict:
    title    = row.get('title') or ''
    abstract = row.get('abstract') or ''
    prompt   = build_screening_prompt(title, abstract, config)
    decision = call_llm(provider, client, model, prompt)
    threshold = config['screening']['threshold']
    score     = decision.get('score', 0)
    decisao   = 'INCLUIR' if score >= threshold else 'EXCLUIR'
    return {**row.to_dict(),
            'centralidade_pobreza':  decision.get('centralidade_pobreza', 0),
            'centralidade_migracao': decision.get('centralidade_migracao', 0),
            'relacao_direta':        decision.get('relacao_direta', 0),
            'score_triagem':         score,
            'decisao':               decisao,
            'justificativa_triagem': decision.get('justificativa', '')}

def screen_articles(df: pd.DataFrame, config: dict, max_workers: int = 1) -> pd.DataFrame:
    provider, client = get_client(config)
    model            = config['screening']['model']
    sleep_time       = config['screening'].get('sleep', 3)
    checkpoint_path  = get_project_root() / 'data' / 'processed' / 'screen_checkpoint.csv'
    lock             = Lock()

    print(f'Provider: {provider} | Modelo: {model} | Sleep: {sleep_time}s')

    if checkpoint_path.exists():
        processed    = pd.read_csv(checkpoint_path)
        already_done = set(processed['id'].tolist())
        print(f'Checkpoint encontrado: {len(already_done)} artigos já processados')
    else:
        already_done = set()

    pending = df[~df['id'].isin(already_done)]
    print(f'Artigos a processar: {len(pending)}')

    rows = [row for _, row in pending.iterrows()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, row, provider, client, model, config): row
                   for row in rows}

        with tqdm(total=len(futures), desc='Triagem') as pbar:
            for future in as_completed(futures):
                result    = future.result()
                result_df = pd.DataFrame([result])

                with lock:
                    result_df.to_csv(
                        checkpoint_path,
                        mode='a',
                        header=not checkpoint_path.exists(),
                        index=False
                    )
                pbar.update(1)
                time.sleep(sleep_time)

    final       = pd.read_csv(checkpoint_path)
    df_included = final[final['decisao'] == 'INCLUIR'].reset_index(drop=True)

    record('incluidos_apos_triagem', len(df_included),
           f'{len(final)} processados → {len(df_included)} incluídos')

    out_path = get_project_root() / 'data' / 'processed' / 'screened.csv'
    df_included.to_csv(out_path, index=False)
    print(f'\nTriagem: {len(final)} processados → {len(df_included)} incluídos')
    return df_included