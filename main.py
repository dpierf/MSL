from src.config_loader import load_config
from src.search        import search_openalex
from src.dedup         import deduplicate
from src.screen        import screen_articles
from src.coding        import code_articles
from src.flatten       import flatten_corpus
from src.analyze       import run_all
from src.keywords      import run_all as run_keywords
import pandas          as pd

def run(config_name: str = None):
    if config_name is None:
        config_name = input('Nome do arquivo de configuração (sem .yaml): ').strip()

    print(f'\n=== MSL Pipeline: {config_name} ===\n')

    config = load_config(config_name)
    print(f'Projeto: {config["project"]["name"]}')

    print('\n[1/7] Buscando no OpenAlex...')
    df_raw = search_openalex(config)

    print('\n[2/7] Deduplicando...')
    df_deduped = deduplicate(df_raw)

    print('\n[3/7] Triagem com LLM...')
    df_screened = screen_articles(df_deduped, config)

    print('\n[4/7] Codificação temática...')
    df_coded = code_articles(df_screened, config)

    print('\n[5/7] Achatando corpus...')
    flatten_corpus(config)

    print('\n[6/7] Análise geral...')
    run_all(config)

    print('\n[7/7] Avaliação de palavras-chave...')
    run_keywords(config)

    print(f'\n=== Concluído: {len(df_coded)} artigos no corpus final ===')
    return df_coded

if __name__ == '__main__':
    run()