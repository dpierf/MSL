import re
import ast
import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt
import matplotlib.colors   as mcolors
from collections           import Counter
from nltk.corpus           import stopwords
from nltk.tokenize         import word_tokenize
from nltk.stem             import WordNetLemmatizer
from nltk.collocations     import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics          import BigramAssocMeasures, TrigramAssocMeasures
from wordcloud             import WordCloud
from .config_loader        import get_project_root

# --- constantes ---

FIG_W   = 6.30
FIG_H_S = 3.15
FIG_H_M = 4.72
FIG_H_L = 6.30

STYLE = {
    'color':      '#2C5F8A',
    'edgecolor':  '#1A3F5C',
    'gridcolor':  '#E5E5E5',
    'fontfamily': 'DejaVu Sans',
}

PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                '2005-2009','2010-2014','2015-2019','2020-2024']

# stopwords absolutas — removidas de unigramas E de bordas de bigramas/trigramas
STOP_UNIGRAMS = set(stopwords.words('english')) | {
    'study', 'paper', 'using', 'data', 'result', 'results', 'model',
    'based', 'used', 'found', 'also', 'may', 'two', 'one', 'new',
    'although', 'however', 'therefore', 'thus', 'hence', 'since',
    'while', 'whereas', 'despite', 'whether', 'either',
    'both', 'neither', 'each', 'every', 'much', 'many', 'several',
    'various', 'certain', 'particular', 'general', 'specific',
    'likely', 'possible', 'potential', 'similar',
    'main', 'major', 'key', 'primary',
    'show', 'shows', 'find', 'finds', 'use', 'uses',
    'include', 'includes', 'suggest', 'suggests', 'indicate',
    'indicates', 'provide', 'provides', 'examine', 'examines',
    'explore', 'explores', 'investigate', 'investigates', 'present',
    'presents', 'discuss', 'discusses', 'argue', 'argues', 'focus',
    'focuses', 'address', 'addresses', 'contribute', 'contributes',
    'abstract', 'introduction', 'conclusion', 'review',
    'article', 'research', 'survey', 'sample',
    'method', 'methods', 'approach', 'analysis',
    'ha', 'wa', 'searched', 'finding', 'findings', 'lmic',
    'associated', 'significant', 'importance', 'important',
    'higher', 'lower', 'greater', 'less', 'high', 'low',
    'large', 'small', 'first', 'second', 'third', 'recent',
    'different', 'various',
}

lemmatizer = WordNetLemmatizer()


# --- funções auxiliares ---

def setup_style():
    plt.rcParams.update({
        'font.family':       STYLE['fontfamily'],
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.color':        STYLE['gridcolor'],
        'grid.linewidth':    0.3,
        'grid.alpha':        0.5,
        'axes.axisbelow':    True,
    })


def save(fig, name: str):
    out = get_project_root() / 'data' / 'output' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  salvo: {name}.png')


def save_table(df: pd.DataFrame, name: str):
    out = get_project_root() / 'data' / 'output' / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f'{name}.csv', index=False)
    print(f'  tabela: {name}.csv')


def get_period(year: int) -> str:
    if year <= 1989:   return 'ate 1989'
    elif year <= 1994: return '1990-1994'
    elif year <= 1999: return '1995-1999'
    elif year <= 2004: return '2000-2004'
    elif year <= 2009: return '2005-2009'
    elif year <= 2014: return '2010-2014'
    elif year <= 2019: return '2015-2019'
    else:              return '2020-2024'


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^a-z\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list:
    return [w for w in word_tokenize(text) if len(w) > 2 and w.isalpha()]


def lemmatize(tokens: list) -> list:
    return [lemmatizer.lemmatize(t) for t in tokens]


def get_unigrams(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_DISPLAY_UNIGRAMS]


def get_bigrams(tokens: list) -> list:
    bigrams = []
    for a, b in zip(tokens, tokens[1:]):
        if len(a) < 3 or len(b) < 3:
            continue
        # nem início nem fim pode ser stopword
        if a in STOP_UNIGRAMS or b in STOP_UNIGRAMS:
            continue
        bg = f'{a} {b}'
        if bg in BANNED_NGRAMS:
            continue
        bigrams.append(bg)
    return bigrams


def get_trigrams(tokens: list) -> list:
    trigrams = []
    for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
        if len(a) < 3 or len(b) < 3 or len(c) < 3:
            continue
        # nem início nem fim pode ser stopword (meio pode)
        if a in STOP_UNIGRAMS or c in STOP_UNIGRAMS:
            continue
        tg = f'{a} {b} {c}'
        if tg in BANNED_NGRAMS:
            continue
        trigrams.append(tg)
    return trigrams


def process_texts(series: pd.Series) -> pd.Series:
    '''Tokeniza, lematiza e retorna tokens por documento.'''
    return series.fillna('').apply(
        lambda t: lemmatize(tokenize(clean_text(t))))


def load_flat() -> pd.DataFrame:
    path = get_project_root() / 'data' / 'processed' / 'flat_corpus.csv'
    df = pd.read_csv(path)
    df = df[df['decisao'] == 'INCLUIR'].reset_index(drop=True)
    df['period'] = df['publication_year'].apply(get_period)
    return df


# --- análises de títulos ---

def title_top_terms(df: pd.DataFrame, n: int = 30):
    tokens_series = process_texts(df['title'])

    uni  = Counter()
    bi   = Counter()
    tri  = Counter()
    for tokens in tokens_series:
        uni.update(get_unigrams(tokens))
        bi.update(get_bigrams(tokens))
        tri.update(get_trigrams(tokens))

    for label, counter, fname in [
        ('unigramas', uni,  'kw_title_unigrams'),
        ('bigramas',  bi,   'kw_title_bigrams'),
        ('trigramas', tri,  'kw_title_trigrams'),
    ]:
        top = pd.DataFrame(counter.most_common(n), columns=['termo', 'n'])
        save_table(top, fname)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
        ax.barh(top['termo'][::-1], top['n'][::-1],
                color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
        for i, (_, row) in enumerate(top[::-1].iterrows()):
            ax.text(row['n'] + 0.1, i, str(row['n']),
                    va='center', ha='left', fontsize=7)
        ax.set_xlabel('Frequencia')
        ax.set_title(f'Top {n} {label} — titulos')
        save(fig, fname)


def title_term_evolution(df: pd.DataFrame, n_terms: int = 10):
    '''Evolução dos n termos mais frequentes (bigramas) por período.'''
    tokens_series = process_texts(df['title'])
    df2 = df.copy()
    df2['tokens'] = tokens_series

    # identifica top bigramas globalmente
    all_bi = Counter()
    for tokens in tokens_series:
        all_bi.update(get_bigrams(tokens))
    top_terms = [t for t, _ in all_bi.most_common(n_terms)]

    rows = []
    for period, grp in df2.groupby('period'):
        total = len(grp)
        period_bi = Counter()
        for tokens in grp['tokens']:
            period_bi.update(get_bigrams(tokens))
        for term in top_terms:
            rows.append({
                'period': period,
                'term':   term,
                'pct':    period_bi[term] / total * 100
            })

    result = pd.DataFrame(rows)
    save_table(result, 'kw_title_bigram_evolution')

    pivot = result.pivot(index='period', columns='term', values='pct').reindex(PERIOD_ORDER)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    for col in pivot.columns:
        ax.plot(PERIOD_ORDER, pivot[col], linewidth=1.5,
                marker='o', markersize=3, label=col)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Ocorrencias por 100 titulos')
    ax.set_title(f'Evolucao dos top-{n_terms} bigramas nos titulos')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Bigrama', fontsize=6, title_fontsize=7, ncol=2)
    save(fig, 'kw_title_bigram_evolution')


def wordcloud_color_func(word, font_size, position, orientation,
                          random_state=None, **kwargs):
    if font_size >= 80:
        return '#0D2B45'   # navy escuro: palavras dominantes
    elif font_size >= 50:
        return '#1A3F5C'   # azul escuro
    elif font_size >= 30:
        return '#2C5F8A'   # azul médio
    elif font_size >= 15:
        return '#5B8DB8'   # azul claro
    else:
        return '#A8C4DC'   # azul muito claro: palavras raras


def title_wordcloud(df: pd.DataFrame):
    tokens_series = process_texts(df['title'])
    bi = Counter()
    uni = Counter()
    for tokens in tokens_series:
        uni.update(get_unigrams(tokens))
        bi.update(get_bigrams(tokens))

    # combina unigramas e bigramas
    freq = {k: v for k, v in uni.items()}
    freq.update({k.replace(' ', '_'): v for k, v in bi.items()})

    wc = WordCloud(width=1890, height=1190,
                   background_color='white',
                   color_func=wordcloud_color_func,
                   max_words=150,
                   collocations=False).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Nuvem de palavras — titulos')
    save(fig, 'kw_title_wordcloud')
    save_table(pd.DataFrame(freq.items(), columns=['termo','n']).sort_values('n', ascending=False),
               'kw_title_wordcloud_freq')


# --- análises de abstracts ---

def abstract_top_terms(df: pd.DataFrame, n: int = 30):
    tokens_series = process_texts(df['abstract'])

    uni = Counter()
    bi  = Counter()
    tri = Counter()
    for tokens in tokens_series:
        uni.update(get_unigrams(tokens))
        bi.update(get_bigrams(tokens))
        tri.update(get_trigrams(tokens))

    for label, counter, fname in [
        ('unigramas', uni,  'kw_abstract_unigrams'),
        ('bigramas',  bi,   'kw_abstract_bigrams'),
        ('trigramas', tri,  'kw_abstract_trigrams'),
    ]:
        top = pd.DataFrame(counter.most_common(n), columns=['termo', 'n'])
        save_table(top, fname)

        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
        ax.barh(top['termo'][::-1], top['n'][::-1],
                color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
        for i, (_, row) in enumerate(top[::-1].iterrows()):
            ax.text(row['n'] + 1, i, str(row['n']),
                    va='center', ha='left', fontsize=7)
        ax.set_xlabel('Frequencia')
        ax.set_title(f'Top {n} {label} — abstracts')
        save(fig, fname)


def abstract_emerging_declining(df: pd.DataFrame, min_freq: int = 5):
    '''Termos emergentes (cresceram) e em declínio (caíram) entre período inicial e final.'''
    tokens_series = process_texts(df['abstract'])
    df2 = df.copy()
    df2['tokens'] = tokens_series

    early = df2[df2['publication_year'] <= 2004]
    late  = df2[df2['publication_year'] >= 2015]

    def term_pct(grp, use_bi=True):
        counter = Counter()
        total = len(grp)
        for tokens in grp['tokens']:
            if use_bi:
                counter.update(get_bigrams(tokens))
            else:
                counter.update(get_unigrams(tokens))
        return {k: v / total * 100 for k, v in counter.items() if v >= min_freq}

    early_pct = term_pct(early)
    late_pct  = term_pct(late)

    all_terms = set(early_pct) | set(late_pct)
    rows = []
    for term in all_terms:
        e = early_pct.get(term, 0)
        l = late_pct.get(term, 0)
        if e > 0 or l > 0:
            rows.append({'termo': term, 'pct_early': e, 'pct_late': l,
                         'delta': l - e})

    result = pd.DataFrame(rows).sort_values('delta', ascending=False)
    save_table(result, 'kw_abstract_emerging_declining')

    # emergentes: maior delta positivo
    emerging = result.head(20)
    # em declínio: maior delta negativo
    declining = result.tail(20).sort_values('delta')

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 1.5, FIG_H_M))

    for ax, data, title, color in [
        (axes[0], emerging,  'Termos emergentes\n(ate 2004 → 2015+)', '#2A9E4F'),
        (axes[1], declining, 'Termos em declinio\n(ate 2004 → 2015+)', '#C0392B'),
    ]:
        ax.barh(data['termo'][::-1], data['delta'][::-1].abs(),
                color=color, alpha=0.8, edgecolor='white', linewidth=0.4)
        ax.set_xlabel('Variacao em p.p. por 100 abstracts')
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis='y', labelsize=7)

    plt.tight_layout()
    save(fig, 'kw_abstract_emerging_declining')


def abstract_wordcloud(df: pd.DataFrame):
    tokens_series = process_texts(df['abstract'])
    bi  = Counter()
    uni = Counter()
    for tokens in tokens_series:
        uni.update(get_unigrams(tokens))
        bi.update(get_bigrams(tokens))

    freq = {k: v for k, v in uni.items()}
    freq.update({k.replace(' ', '_'): v for k, v in bi.items()})

    wc = WordCloud(width=1890, height=1190,
                   background_color='white',
                   color_func=wordcloud_color_func,
                   max_words=150,
                   collocations=False).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Nuvem de palavras — abstracts')
    save(fig, 'kw_abstract_wordcloud')
    save_table(pd.DataFrame(freq.items(), columns=['termo','n']).sort_values('n', ascending=False),
               'kw_abstract_wordcloud_freq')


def abstract_bigram_evolution(df: pd.DataFrame, n_terms: int = 10):
    tokens_series = process_texts(df['abstract'])
    df2 = df.copy()
    df2['tokens'] = tokens_series

    all_bi = Counter()
    for tokens in tokens_series:
        all_bi.update(get_bigrams(tokens))
    top_terms = [t for t, _ in all_bi.most_common(n_terms)]

    rows = []
    for period, grp in df2.groupby('period'):
        total = len(grp)
        period_bi = Counter()
        for tokens in grp['tokens']:
            period_bi.update(get_bigrams(tokens))
        for term in top_terms:
            rows.append({
                'period': period,
                'term':   term,
                'pct':    period_bi[term] / total * 100
            })

    result = pd.DataFrame(rows)
    save_table(result, 'kw_abstract_bigram_evolution')

    pivot = result.pivot(index='period', columns='term', values='pct').reindex(PERIOD_ORDER)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    for col in pivot.columns:
        ax.plot(PERIOD_ORDER, pivot[col], linewidth=1.5,
                marker='o', markersize=3, label=col)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Ocorrencias por 100 abstracts')
    ax.set_title(f'Evolucao dos top-{n_terms} bigramas nos abstracts')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Bigrama', fontsize=6, title_fontsize=7, ncol=2)
    save(fig, 'kw_abstract_bigram_evolution')


def abstract_vocab_by_field(df: pd.DataFrame, n: int = 15):
    '''Top bigramas exclusivos ou dominantes por área de conhecimento.'''
    top_fields = df['primary_field'].value_counts().head(4).index
    tokens_series = process_texts(df['abstract'])
    df2 = df.copy()
    df2['tokens'] = tokens_series

    all_rows = []
    field_counters = {}
    for field in top_fields:
        grp = df2[df2['primary_field'] == field]
        counter = Counter()
        for tokens in grp['tokens']:
            counter.update(get_bigrams(tokens))
        field_counters[field] = counter

    # normaliza por total de documentos por área
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W * 1.5, FIG_H_L))
    axes = axes.flatten()

    for ax, field in zip(axes, top_fields):
        counter = field_counters[field]
        total   = len(df2[df2['primary_field'] == field])
        top = pd.DataFrame(
            [(k, v / total * 100) for k, v in counter.most_common(n)],
            columns=['termo', 'pct'])
        all_rows.append(top.assign(area=field))
        ax.barh(top['termo'][::-1], top['pct'][::-1],
                color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
        ax.set_title(field, fontsize=8)
        ax.set_xlabel('Ocorrencias por 100 abstracts', fontsize=7)
        ax.tick_params(axis='y', labelsize=6.5)

    plt.suptitle('Top bigramas por area de conhecimento', fontsize=10)
    plt.tight_layout()
    save(fig, 'kw_abstract_vocab_by_field')
    save_table(pd.concat(all_rows), 'kw_abstract_vocab_by_field')


# --- orquestrador ---

def run_all(config: dict):
    global STOP_DISPLAY_UNIGRAMS, BANNED_NGRAMS

    stop_display_extra = set(config['keywords'].get('stop_display_unigrams', []))
    stop_artifact      = set(config['keywords'].get('stop_artifact', []))
    STOP_UNIGRAMS.update(stop_artifact)
    STOP_DISPLAY_UNIGRAMS = STOP_UNIGRAMS | stop_display_extra

    BANNED_NGRAMS = set(config['keywords'].get('banned_ngrams', []))

    setup_style()
    print('\n=== Analise de palavras-chave ===\n')

    df = load_flat()

    print('[T1] Top termos — titulos')
    title_top_terms(df)

    print('[T2] Evolucao de bigramas — titulos')
    title_term_evolution(df)

    print('[T3] Wordcloud — titulos')
    title_wordcloud(df)

    print('[A1] Top termos — abstracts')
    abstract_top_terms(df)

    print('[A2] Evolucao de bigramas — abstracts')
    abstract_bigram_evolution(df)

    print('[A3] Termos emergentes e em declinio — abstracts')
    abstract_emerging_declining(df)

    print('[A4] Wordcloud — abstracts')
    abstract_wordcloud(df)

    print('[A5] Vocabulario por area — abstracts')
    abstract_vocab_by_field(df)

    print('\nConcluido.')