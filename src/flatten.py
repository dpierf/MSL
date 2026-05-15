import ast
import time
import requests
import pandas as pd
from pathlib           import Path
from collections       import Counter
from .config_loader    import get_project_root

# --- constantes ---
ISO_COUNTRIES = {
    'AF': 'Afghanistan', 'AL': 'Albania', 'DZ': 'Algeria', 'AD': 'Andorra',
    'AO': 'Angola', 'AG': 'Antigua and Barbuda', 'AR': 'Argentina',
    'AM': 'Armenia', 'AU': 'Australia', 'AT': 'Austria', 'AZ': 'Azerbaijan',
    'BS': 'Bahamas', 'BH': 'Bahrain', 'BD': 'Bangladesh', 'BB': 'Barbados',
    'BY': 'Belarus', 'BE': 'Belgium', 'BZ': 'Belize', 'BJ': 'Benin',
    'BT': 'Bhutan', 'BO': 'Bolivia', 'BA': 'Bosnia and Herzegovina',
    'BW': 'Botswana', 'BR': 'Brazil', 'BN': 'Brunei', 'BG': 'Bulgaria',
    'BF': 'Burkina Faso', 'BI': 'Burundi', 'CV': 'Cabo Verde',
    'KH': 'Cambodia', 'CM': 'Cameroon', 'CA': 'Canada',
    'CF': 'Central African Republic', 'TD': 'Chad', 'CL': 'Chile',
    'CN': 'China', 'CO': 'Colombia', 'KM': 'Comoros', 'CG': 'Congo',
    'CD': 'Congo (DRC)', 'CR': 'Costa Rica', 'HR': 'Croatia', 'CU': 'Cuba',
    'CY': 'Cyprus', 'CZ': 'Czech Republic', 'DK': 'Denmark',
    'DJ': 'Djibouti', 'DM': 'Dominica', 'DO': 'Dominican Republic',
    'EC': 'Ecuador', 'EG': 'Egypt', 'SV': 'El Salvador',
    'GQ': 'Equatorial Guinea', 'ER': 'Eritrea', 'EE': 'Estonia',
    'SZ': 'Eswatini', 'ET': 'Ethiopia', 'FJ': 'Fiji', 'FI': 'Finland',
    'FR': 'France', 'GA': 'Gabon', 'GM': 'Gambia', 'GE': 'Georgia',
    'DE': 'Germany', 'GH': 'Ghana', 'GR': 'Greece', 'GD': 'Grenada',
    'GT': 'Guatemala', 'GN': 'Guinea', 'GW': 'Guinea-Bissau', 'GY': 'Guyana',
    'HT': 'Haiti', 'HN': 'Honduras', 'HU': 'Hungary', 'IS': 'Iceland',
    'IN': 'India', 'ID': 'Indonesia', 'IR': 'Iran', 'IQ': 'Iraq',
    'IE': 'Ireland', 'IL': 'Israel', 'IT': 'Italy', 'JM': 'Jamaica',
    'JP': 'Japan', 'JO': 'Jordan', 'KZ': 'Kazakhstan', 'KE': 'Kenya',
    'KI': 'Kiribati', 'KW': 'Kuwait', 'KG': 'Kyrgyzstan', 'LA': 'Laos',
    'LV': 'Latvia', 'LB': 'Lebanon', 'LS': 'Lesotho', 'LR': 'Liberia',
    'LY': 'Libya', 'LI': 'Liechtenstein', 'LT': 'Lithuania',
    'LU': 'Luxembourg', 'MG': 'Madagascar', 'MW': 'Malawi',
    'MY': 'Malaysia', 'MV': 'Maldives', 'ML': 'Mali', 'MT': 'Malta',
    'MH': 'Marshall Islands', 'MR': 'Mauritania', 'MU': 'Mauritius',
    'MX': 'Mexico', 'FM': 'Micronesia', 'MD': 'Moldova', 'MC': 'Monaco',
    'MN': 'Mongolia', 'ME': 'Montenegro', 'MA': 'Morocco', 'MZ': 'Mozambique',
    'MM': 'Myanmar', 'NA': 'Namibia', 'NR': 'Nauru', 'NP': 'Nepal',
    'NL': 'Netherlands', 'NZ': 'New Zealand', 'NI': 'Nicaragua',
    'NE': 'Niger', 'NG': 'Nigeria', 'NO': 'Norway', 'OM': 'Oman',
    'PK': 'Pakistan', 'PW': 'Palau', 'PA': 'Panama',
    'PG': 'Papua New Guinea', 'PY': 'Paraguay', 'PE': 'Peru',
    'PH': 'Philippines', 'PL': 'Poland', 'PT': 'Portugal', 'QA': 'Qatar',
    'RO': 'Romania', 'RU': 'Russia', 'RW': 'Rwanda',
    'KN': 'Saint Kitts and Nevis', 'LC': 'Saint Lucia',
    'VC': 'Saint Vincent and the Grenadines', 'WS': 'Samoa',
    'SM': 'San Marino', 'ST': 'Sao Tome and Principe', 'SA': 'Saudi Arabia',
    'SN': 'Senegal', 'RS': 'Serbia', 'SC': 'Seychelles',
    'SL': 'Sierra Leone', 'SG': 'Singapore', 'SK': 'Slovakia',
    'SI': 'Slovenia', 'SB': 'Solomon Islands', 'SO': 'Somalia',
    'ZA': 'South Africa', 'SS': 'South Sudan', 'ES': 'Spain',
    'LK': 'Sri Lanka', 'SD': 'Sudan', 'SR': 'Suriname', 'SE': 'Sweden',
    'CH': 'Switzerland', 'SY': 'Syria', 'TW': 'Taiwan', 'TJ': 'Tajikistan',
    'TZ': 'Tanzania', 'TH': 'Thailand', 'TL': 'Timor-Leste', 'TG': 'Togo',
    'TO': 'Tonga', 'TT': 'Trinidad and Tobago', 'TN': 'Tunisia',
    'TR': 'Turkey', 'TM': 'Turkmenistan', 'TV': 'Tuvalu', 'UG': 'Uganda',
    'UA': 'Ukraine', 'AE': 'United Arab Emirates', 'GB': 'United Kingdom',
    'US': 'United States', 'UY': 'Uruguay', 'UZ': 'Uzbekistan',
    'VU': 'Vanuatu', 'VE': 'Venezuela', 'VN': 'Vietnam', 'YE': 'Yemen',
    'ZM': 'Zambia', 'ZW': 'Zimbabwe',
}

# cache para evitar consultas repetidas ao CrossRef
_crossref_cache: dict = {}


# --- funções auxiliares ---

def safe_parse(val):
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, float):
        return None
    if isinstance(val, str) and val.strip() == '':
        return None
    try:
        return ast.literal_eval(str(val))
    except:
        return None


def is_institutional(author: dict, institutions: list) -> bool:
    '''Retorna True se o autor for institucional (sem ID OpenAlex e sem ORCID,
    e cujo nome coincide com uma das instituições listadas).'''
    name = (author.get('display_name') or '').strip()
    has_id    = bool(author.get('id'))
    has_orcid = bool(author.get('orcid'))
    if has_id or has_orcid:
        return False
    inst_names = {i.get('display_name', '') for i in institutions}
    return name in inst_names


def extract_first_author(authorships: list) -> str:
    if not authorships:
        return None
    # tenta encontrar o primeiro autor pessoa física
    for a in authorships:
        author = a.get('author', {}) or {}
        insts  = a.get('institutions', []) or []
        if not is_institutional(author, insts):
            name = author.get('display_name')
            if name:
                return name
    # fallback: retorna o primeiro independente de ser institucional
    try:
        return authorships[0]['author']['display_name']
    except:
        return None


def lookup_crossref_journal(doi: str) -> str | None:
    '''Consulta o CrossRef para obter o nome do periódico a partir do DOI.'''
    if not doi:
        return None
    doi_clean = doi.replace('https://doi.org/', '').strip()
    if doi_clean in _crossref_cache:
        return _crossref_cache[doi_clean]
    try:
        url = f'https://api.crossref.org/works/{doi_clean}'
        headers = {'User-Agent': 'ProjetoMSL/1.0 (mailto:pesquisa@exemplo.com)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            titles = data.get('message', {}).get('container-title', [])
            result = titles[0] if titles else None
            _crossref_cache[doi_clean] = result
            time.sleep(0.1)  # respeita rate limit do CrossRef
            return result
    except:
        pass
    _crossref_cache[doi_clean] = None
    return None


def extract_journal(primary_location: dict, doi: str = None, repositorios: set = None) -> str:
    try:
        source_name = primary_location['source']['display_name']
        if repositorios and source_name in repositorios and doi:
            crossref_name = lookup_crossref_journal(doi)
            return crossref_name if crossref_name else source_name
        return source_name
    except:
        return None


def extract_publisher(primary_location: dict) -> str:
    try:
        return primary_location['source']['host_organization_name']
    except:
        return None


def extract_source_type(primary_location: dict) -> str:
    try:
        return primary_location['source']['type']
    except:
        return None


def extract_is_oa(primary_location: dict) -> bool:
    try:
        return primary_location['source']['is_oa']
    except:
        return None


def extract_author_count(authorships: list) -> int:
    try:
        return len(authorships)
    except:
        return 0


def extract_countries(authorships: list) -> list:
    try:
        codes = []
        for a in authorships:
            codes.extend(a.get('countries', []))
        codes = list(set(codes))
        return [ISO_COUNTRIES.get(c, c) for c in codes]
    except:
        return []


def extract_institutions(authorships: list) -> list:
    try:
        insts = []
        for a in authorships:
            for inst in a.get('institutions', []):
                name = inst.get('display_name')
                if name:
                    insts.append(name)
        return list(set(insts))
    except:
        return []


def extract_primary_topic(topics: list) -> str:
    try:
        return topics[0]['display_name']
    except:
        return None


def extract_primary_field(topics: list) -> str:
    try:
        return topics[0]['field']['display_name']
    except:
        return None


def extract_primary_domain(topics: list) -> str:
    try:
        return topics[0]['domain']['display_name']
    except:
        return None


def extract_keyword_names(keywords: list) -> list:
    try:
        return [k['display_name'] for k in keywords]
    except:
        return []


# --- função principal ---

def flatten_corpus(config: dict) -> pd.DataFrame:
    repositorios = set(config['sources']['repositories'])
    root = get_project_root()

    df = pd.read_json(root / 'data' / 'processed' / 'deduped.json', orient='records')

    screened_path = root / 'data' / 'processed' / 'screened.csv'
    if screened_path.exists():
        screened = pd.read_csv(screened_path)[
            ['id', 'score_triagem', 'centralidade_pobreza',
             'centralidade_migracao', 'relacao_direta',
             'decisao', 'justificativa_triagem']
        ]
    else:
        screened = pd.DataFrame(columns=['id', 'score_triagem', 'decisao',
                                          'justificativa_triagem'])

    df = df.merge(screened, on='id', how='left')
    df['decisao']       = df['decisao'].fillna('PENDENTE')
    df['score_triagem'] = df['score_triagem'].fillna(0)

    for col in ['primary_location', 'authorships', 'keywords', 'topics', 'open_access']:
        df[col] = df[col].apply(safe_parse)

    print('Extraindo campos e consultando CrossRef para repositórios...')
    n_crossref = 0

    journals      = []
    publishers    = []
    source_types  = []
    is_oas        = []
    first_authors = []
    author_counts = []
    countries_    = []
    institutions_ = []
    topics_       = []
    fields_       = []
    domains_      = []
    keywords_     = []

    for _, row in df.iterrows():
        loc   = row['primary_location']
        auths = row['authorships']
        kws   = row['keywords']
        tops  = row['topics']
        doi   = row.get('doi')

        # verifica se vai consultar CrossRef
        journal_raw = None
        if isinstance(loc, dict):
            try:
                journal_raw = loc['source']['display_name']
            except:
                pass
        needs_crossref = journal_raw in repositorios and doi

        if needs_crossref:
            n_crossref += 1

        journals.append(extract_journal(loc, doi, repositorios) if isinstance(loc, dict) else None)
        publishers.append(extract_publisher(loc) if isinstance(loc, dict) else None)
        source_types.append(extract_source_type(loc) if isinstance(loc, dict) else None)
        is_oas.append(extract_is_oa(loc) if isinstance(loc, dict) else None)
        first_authors.append(extract_first_author(auths) if isinstance(auths, list) else None)
        author_counts.append(extract_author_count(auths) if isinstance(auths, list) else 0)
        countries_.append(extract_countries(auths) if isinstance(auths, list) else [])
        institutions_.append(extract_institutions(auths) if isinstance(auths, list) else [])
        topics_.append(extract_primary_topic(tops) if isinstance(tops, list) else None)
        fields_.append(extract_primary_field(tops) if isinstance(tops, list) else None)
        domains_.append(extract_primary_domain(tops) if isinstance(tops, list) else None)
        keywords_.append(extract_keyword_names(kws) if isinstance(kws, list) else [])

    df['journal']         = journals
    df['publisher']       = publishers
    df['source_type']     = source_types
    df['is_oa']           = is_oas
    df['first_author']    = first_authors
    df['author_count']    = author_counts
    df['countries']       = countries_
    df['institutions']    = institutions_
    df['primary_topic']   = topics_
    df['primary_field']   = fields_
    df['primary_domain']  = domains_
    df['keyword_names']   = keywords_

    out_path = root / 'data' / 'processed' / 'flat_corpus.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f'Corpus achatado: {len(df)} registros → {out_path}')
    print(f'Consultas CrossRef realizadas: {n_crossref}')
    return df