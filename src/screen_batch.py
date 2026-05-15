import json
import time
import pandas as pd
from pathlib         import Path
from groq            import Groq
from .config_loader  import load_config, get_groq_key, get_project_root
from .screen         import build_screening_prompt
from .metrics        import record

def prepare_batch_file(df: pd.DataFrame, config: dict) -> Path:
    root     = get_project_root()
    out_path = root / 'data' / 'processed' / 'batch_input.jsonl'
    model    = config['screening']['model']

    with open(out_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            title    = row.get('title') or ''
            abstract = row.get('abstract') or ''
            prompt   = build_screening_prompt(title, abstract, config)

            request = {
                'custom_id': row['id'],
                'method':    'POST',
                'url':       '/v1/chat/completions',
                'body': {
                    'model':       model,
                    'messages':    [{'role': 'user', 'content': prompt}],
                    'temperature': 0.0,
                    'max_tokens':  150
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + '\n')

    print(f'Batch file criado: {out_path} ({len(df)} requisições)')
    return out_path

def upload_batch_file(client: Groq, file_path: Path) -> str:
    with open(file_path, 'rb') as f:
        response = client.files.create(file=f, purpose='batch')
    print(f'Arquivo enviado: {response.id}')
    return response.id

def create_batch_job(client: Groq, file_id: str, window: str = '7d') -> str:
    response = client.batches.create(
        completion_window=window,
        endpoint='/v1/chat/completions',
        input_file_id=file_id,
    )
    print(f'Job criado: {response.id} | status: {response.status}')
    return response.id

def poll_batch_status(client: Groq, batch_id: str, interval: int = 60) -> str:
    print(f'Monitorando batch {batch_id}...')
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f'  status: {batch.status} | '
              f'concluídos: {counts.completed}/{counts.total} | '
              f'falhos: {counts.failed}')

        if batch.status in ('completed', 'failed', 'expired', 'cancelled'):
            return batch.status

        time.sleep(interval)

def download_results(client: Groq, batch_id: str) -> Path:
    batch    = client.batches.retrieve(batch_id)
    root     = get_project_root()
    out_path = root / 'data' / 'processed' / 'batch_output.jsonl'

    if not batch.output_file_id:
        raise RuntimeError(f'Sem output_file_id. Status: {batch.status}')

    content = client.files.content(batch.output_file_id)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content.text)

    print(f'Resultados salvos: {out_path}')
    return out_path

def parse_results(output_path: Path, df: pd.DataFrame, config: dict) -> pd.DataFrame:
    threshold = config['screening']['threshold']
    results   = {}

    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            record_data = json.loads(line)
            custom_id   = record_data['custom_id']
            try:
                content = record_data['response']['body']['choices'][0]['message']['content']
                content = content.strip()
                if content.startswith('```'):
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:]
                    content = content.strip()
                parsed = json.loads(content)
            except Exception as e:
                parsed = {
                    'centralidade_pobreza':  0,
                    'centralidade_migracao': 0,
                    'relacao_direta':        0,
                    'score':                 0,
                    'justificativa':         f'erro: {str(e)}'
                }

            score   = parsed.get('score', 0)
            decisao = 'INCLUIR' if score >= threshold else 'EXCLUIR'
            results[custom_id] = {
                'centralidade_pobreza':  parsed.get('centralidade_pobreza', 0),
                'centralidade_migracao': parsed.get('centralidade_migracao', 0),
                'relacao_direta':        parsed.get('relacao_direta', 0),
                'score_triagem':         score,
                'decisao':               decisao,
                'justificativa_triagem': parsed.get('justificativa', ''),
            }

    df_out = df.copy()
    for col in ['centralidade_pobreza', 'centralidade_migracao',
                'relacao_direta', 'score_triagem', 'decisao', 'justificativa_triagem']:
        df_out[col] = df_out['id'].map(lambda x: results.get(x, {}).get(col, None))

    df_included = df_out[df_out['decisao'] == 'INCLUIR'].reset_index(drop=True)
    record('incluidos_apos_triagem', len(df_included),
           f'{len(df_out)} processados → {len(df_included)} incluídos')

    out_path = get_project_root() / 'data' / 'processed' / 'screened.csv'
    df_included.to_csv(out_path, index=False)
    print(f'Triagem concluída: {len(df_included)} incluídos')
    return df_included

def screen_batch(df: pd.DataFrame, config: dict, window: str = '7d') -> pd.DataFrame:
    client = Groq(api_key=get_groq_key())

    print('\n[1/4] Preparando arquivo batch...')
    batch_file = prepare_batch_file(df, config)

    print('\n[2/4] Enviando para o Groq...')
    file_id  = upload_batch_file(client, batch_file)
    batch_id = create_batch_job(client, file_id, window)

    print(f'\n[3/4] Aguardando conclusão (batch_id: {batch_id})...')
    print('      Salve o batch_id acima caso precise retomar depois.')
    status = poll_batch_status(client, batch_id)

    if status != 'completed':
        raise RuntimeError(f'Batch encerrou com status: {status}')

    print('\n[4/4] Baixando e parseando resultados...')
    output_path = download_results(client, batch_id)
    return parse_results(output_path, df, config)