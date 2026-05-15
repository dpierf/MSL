import pandas as pd
from pyalex           import Works
from itertools        import product
from .config_loader   import load_config, get_project_root
from .metrics         import record

def build_queries(config: dict) -> list[dict]:
    group_a = config['search']['term_groups']['group_a']
    group_b = config['search']['term_groups']['group_b']
    queries = []
    for i, (a, b) in enumerate(product(group_a, group_b), start=1):
        queries.append({
            'id':      f'T{i:02d}',
            'string':  f'{a} {b}',
            'group_a': a,
            'group_b': b,
        })
    return queries

def reconstruct_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ''
    positions = [
        (pos, word)
        for word, positions in inverted_index.items()
        for pos in positions
    ]
    return ' '.join(word for _, word in sorted(positions))

def search_openalex(config: dict) -> pd.DataFrame:
    queries    = build_queries(config)
    date_from  = config['search']['date_from']
    date_to    = config['search']['date_to']
    max_pages  = config['search']['max_pages']
    all_results = []

    for query in queries:
        print(f'Buscando {query["id"]}: {query["string"]}')
        works = (
            Works()
            .search(query['string'])
            .filter(
                publication_year=f'{date_from}-{date_to}',
                language=config['search']['language']
            )
            .select([
                'id', 'doi', 'title', 'abstract_inverted_index',
                'publication_year', 'primary_location', 'authorships',
                'keywords', 'topics', 'cited_by_count',
                'referenced_works', 'countries_distinct_count', 'open_access'
            ])
            .paginate(per_page=200)
        )

        position = 1
        page_num = 1
        for page in works:
            if page_num > max_pages:
                break
            for work in page:
                work['query_rank']    = f'{query["id"]}-{position:03d}'
                work['query_id']      = query['id']
                work['query_string']  = query['string']
                work['query_group_a'] = query['group_a']
                work['query_group_b'] = query['group_b']
                work['query_page']    = page_num
                work['query_pos']     = position
                all_results.append(work)
                position += 1
            page_num += 1

    df = pd.DataFrame(all_results)
    df['abstract'] = df['abstract_inverted_index'].apply(
        lambda x: reconstruct_abstract(x) if isinstance(x, dict) else ''
    )

    record('total_buscados_bruto', len(df),
           f'{len(queries)} queries × até {max_pages * 200} resultados')

    raw_path = get_project_root() / 'data' / 'raw' / 'search_results.json'
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_json(raw_path, orient='records', force_ascii=False, indent=2)
    print(f'{len(df)} registros salvos em {raw_path}')
    return df