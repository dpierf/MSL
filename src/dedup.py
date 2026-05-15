import pandas as pd
from rapidfuzz        import fuzz
from .config_loader   import get_project_root
from .metrics         import record

def dedup_by_doi(df: pd.DataFrame) -> pd.DataFrame:
    if 'doi' not in df.columns:
        return df
    # para duplicatas por DOI, mantém a de menor posição (mais relevante)
    df_with_doi    = df[df['doi'].notna() & (df['doi'] != '')]
    df_without_doi = df[df['doi'].isna() | (df['doi'] == '')]

    df_with_doi = (
        df_with_doi
        .sort_values('query_pos', ascending=True)
        .drop_duplicates(subset='doi', keep='first')
    )

    sem_doi = len(df) - len(df_with_doi) - len(df_without_doi)
    record('sem_doi', len(df_without_doi), 'sem DOI — não excluídos')

    return pd.concat([df_with_doi, df_without_doi], ignore_index=True)

def dedup_by_title(df: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
    # ordena por posição antes de deduplicar — mantém o mais relevante
    df = df.sort_values('query_pos', ascending=True).reset_index(drop=True)
    titles = df['title'].fillna('').tolist()
    to_drop = set()

    for i in range(len(titles)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(titles)):
            if j in to_drop:
                continue
            if fuzz.ratio(titles[i].lower(), titles[j].lower()) >= threshold:
                to_drop.add(j)

    return df.drop(index=list(to_drop)).reset_index(drop=True)

def filter_empty_fields(df: pd.DataFrame) -> pd.DataFrame:
    before       = len(df)
    sem_titulo   = df['title'].isna() | df['title'].str.strip().eq('')
    sem_abstract = df['abstract'].isna() | df['abstract'].str.strip().eq('')
    df_clean     = df[~sem_titulo & ~sem_abstract].reset_index(drop=True)
    removidos    = before - len(df_clean)
    record('removidos_sem_titulo_ou_abstract', removidos,
           f'sem título: {int(sem_titulo.sum())}, sem abstract: {int(sem_abstract.sum())}')
    return df_clean

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    record('total_buscados', len(df))

    before = len(df)
    df     = dedup_by_doi(df)
    df     = df.reset_index(drop=True)
    df     = dedup_by_title(df)
    after  = len(df)
    record('removidos_duplicados', before - after,
           f'{before} → {after} após deduplicação')

    df = filter_empty_fields(df)
    record('corpus_para_triagem', len(df))

    out_path = get_project_root() / 'data' / 'processed' / 'deduped.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_json(out_path, orient='records', force_ascii=False, indent=2)
    return df