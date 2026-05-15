import ast
import numpy              as np
import pandas             as pd
import matplotlib.pyplot  as plt
import matplotlib.ticker  as ticker
from collections          import Counter
from .config_loader       import get_project_root

# --- layout A4 ---
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


AMERICAS = {
    'United States', 'Canada', 'Mexico', 'Brazil', 'Argentina', 'Colombia',
    'Chile', 'Peru', 'Venezuela', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay',
    'Guyana', 'Suriname', 'Trinidad and Tobago', 'Jamaica', 'Cuba', 'Haiti',
    'Dominican Republic', 'Guatemala', 'Honduras', 'El Salvador', 'Nicaragua',
    'Costa Rica', 'Panama', 'Belize', 'Barbados', 'Bahamas', 'Grenada',
    'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines',
    'Antigua and Barbuda', 'Dominica',
}

EUROPE = {
    'United Kingdom', 'Germany', 'France', 'Netherlands', 'Sweden', 'Spain',
    'Italy', 'Switzerland', 'Norway', 'Belgium', 'Austria', 'Denmark',
    'Finland', 'Ireland', 'Portugal', 'Poland', 'Czech Republic', 'Hungary',
    'Romania', 'Bulgaria', 'Greece', 'Croatia', 'Slovakia', 'Slovenia',
    'Estonia', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Cyprus',
    'Iceland', 'Liechtenstein', 'Monaco', 'Andorra', 'Serbia', 'Montenegro',
    'Bosnia and Herzegovina', 'Albania', 'North Macedonia', 'Moldova',
    'Ukraine', 'Belarus', 'Russia', 'Turkey',
}

ASIA_OCEANIA_AFRICA = {
    'China', 'India', 'Japan', 'South Korea', 'Indonesia', 'Pakistan',
    'Bangladesh', 'Vietnam', 'Philippines', 'Thailand', 'Malaysia',
    'Singapore', 'Sri Lanka', 'Nepal', 'Cambodia', 'Myanmar', 'Laos',
    'Mongolia', 'Kazakhstan', 'Uzbekistan', 'Kyrgyzstan', 'Tajikistan',
    'Turkmenistan', 'Azerbaijan', 'Georgia', 'Armenia', 'Iran', 'Iraq',
    'Saudi Arabia', 'United Arab Emirates', 'Kuwait', 'Qatar', 'Bahrain',
    'Oman', 'Jordan', 'Lebanon', 'Israel', 'Syria', 'Yemen', 'Afghanistan',
    'Taiwan', 'Maldives', 'Bhutan', 'Brunei', 'Timor-Leste',
    'Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands',
    'Vanuatu', 'Samoa', 'Tonga', 'Kiribati', 'Micronesia', 'Palau', 'Nauru',
    'Tuvalu', 'Marshall Islands',
    'Nigeria', 'South Africa', 'Ethiopia', 'Egypt', 'Kenya', 'Ghana',
    'Tanzania', 'Uganda', 'Mozambique', 'Zambia', 'Zimbabwe', 'Malawi',
    'Senegal', 'Mali', 'Burkina Faso', 'Niger', 'Chad', 'Sudan', 'South Sudan',
    'Somalia', 'Eritrea', 'Djibouti', 'Rwanda', 'Burundi', 'Congo (DRC)',
    'Congo', 'Cameroon', 'Gabon', 'Central African Republic', 'Angola',
    'Namibia', 'Botswana', 'Lesotho', 'Eswatini', 'Madagascar', 'Mauritius',
    'Seychelles', 'Cabo Verde', 'Guinea', 'Guinea-Bissau', 'Sierra Leone',
    'Liberia', 'Togo', 'Benin', 'Equatorial Guinea', 'Sao Tome and Principe',
    'Comoros', 'Libya', 'Tunisia', 'Algeria', 'Morocco', 'Mauritania',
}


def get_region(country: str, top5: set) -> str | None:
    if country in top5:
        return country
    if country in AMERICAS:
        return 'Demais Americas'
    if country in EUROPE:
        return 'Demais Europa'
    if country in ASIA_OCEANIA_AFRICA:
        return 'Asia, Oceania e Africa'
    return None


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


def load_flat(only_included: bool = True) -> pd.DataFrame:
    path = get_project_root() / 'data' / 'processed' / 'flat_corpus.csv'
    df = pd.read_csv(path)
    if only_included:
        df = df[df['decisao'] == 'INCLUIR'].reset_index(drop=True)
    return df


def parse_list_col(series: pd.Series) -> list:
    result = []
    for val in series.dropna():
        try:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(parsed, list):
                result.extend(parsed)
        except:
            pass
    return result


def temporal_distribution(df: pd.DataFrame):
    counts = df.groupby('publication_year').size().reset_index(name='n')
    save_table(counts, '01_temporal_distribution')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_S))
    ax.bar(counts['publication_year'], counts['n'],
           color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xlabel('Ano de publicacao')
    ax.set_ylabel('Numero de artigos')
    ax.set_title('Distribuicao temporal das publicacoes')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    save(fig, '01_temporal_distribution')


def top_journals(df, repositorios, n=20):
    df_j = df[~df['journal'].isin(repositorios)]
    counts = (df_j['journal'].dropna()
              .value_counts().head(n)
              .reset_index()
              .rename(columns={'journal': 'periodico', 'count': 'n'}))
    save_table(counts, '02_top_journals')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    bars = ax.barh(counts['periodico'][::-1], counts['n'][::-1],
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xlabel('Numero de artigos')
    ax.set_title(f'Top {n} periodicos')
    for bar, val in zip(bars, counts['n'][::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=7)
    save(fig, '02_top_journals')


def top_authors(df, autores_institucionais, n=20):
    df_a = df[~df['first_author'].isin(autores_institucionais)]
    counts = (df_a['first_author'].dropna()
              .value_counts().head(n)
              .reset_index()
              .rename(columns={'first_author': 'autor', 'count': 'n'}))
    save_table(counts, '03_top_authors')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    bars = ax.barh(counts['autor'][::-1], counts['n'][::-1],
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xlabel('Numero de artigos')
    ax.set_title(f'Top {n} primeiros autores')
    for bar, val in zip(bars, counts['n'][::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=7)
    save(fig, '03_top_authors')


def top_fields(df: pd.DataFrame):
    counts = (df['primary_field'].dropna()
              .value_counts()
              .reset_index()
              .rename(columns={'primary_field': 'area', 'count': 'n'}))
    save_table(counts, '04_top_fields')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    bars = ax.barh(counts['area'][::-1], counts['n'][::-1],
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xscale('log')
    ax.set_xlabel('Numero de artigos (escala log)')
    ax.set_title('Distribuicao por area de conhecimento')
    for bar, val in zip(bars, counts['n'][::-1]):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=7)
    save(fig, '04_top_fields')


def top_countries(df: pd.DataFrame, n: int = 20):
    all_countries = parse_list_col(df['countries'])
    counts = (pd.Series(Counter(all_countries))
              .sort_values(ascending=False).head(n)
              .reset_index())
    counts.columns = ['pais', 'n']
    save_table(counts, '05_top_countries')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    bars = ax.barh(counts['pais'][::-1], counts['n'][::-1],
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xscale('log')
    ax.set_xlabel('Numero de artigos (escala log)')
    ax.set_title(f'Top {n} paises de afiliacao')
    for bar, val in zip(bars, counts['n'][::-1]):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=7)
    save(fig, '05_top_countries')


def top_institutions(df: pd.DataFrame, n: int = 20):
    all_insts = parse_list_col(df['institutions'])
    counts = (pd.Series(Counter(all_insts))
              .sort_values(ascending=False).head(n)
              .reset_index())
    counts.columns = ['instituicao', 'n']
    save_table(counts, '06_top_institutions')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    bars = ax.barh(counts['instituicao'][::-1], counts['n'][::-1],
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xlabel('Numero de artigos')
    ax.set_title(f'Top {n} instituicoes')
    for bar, val in zip(bars, counts['n'][::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=7)
    save(fig, '06_top_institutions')


def citation_distribution(df: pd.DataFrame):
    df_cit = df[df['cited_by_count'] >= 0].copy()

    bins = [0, 1, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
            600, 700, 800, 900, 1000, float('inf')]
    labels = ['0', '1-9', '10-24', '25-49', '50-99',
              '100-149', '150-199', '200-249', '250-299',
              '300-349', '350-399', '400-449', '450-499',
              '500-599', '600-699', '700-799', '800-899',
              '900-999', '1000+']

    df_cit['faixa'] = pd.cut(df_cit['cited_by_count'],
                              bins=bins, labels=labels, right=False)
    counts = df_cit['faixa'].value_counts().sort_index().reset_index()
    counts.columns = ['faixa', 'n']
    save_table(counts, '07_citations')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    ax.bar(range(len(counts)), counts['n'],
           color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_yscale('log')
    ax.set_ylabel('Numero de artigos (escala log)')
    ax.set_xlabel('Faixa de citacoes')
    ax.set_title('Distribuicao de citacoes')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts['faixa'], rotation=55, ha='right', fontsize=6.5)
    save(fig, '07_citation_distribution')


def open_access_share(df: pd.DataFrame):
    counts = df['is_oa'].map(
        {True: 'Aberto', False: 'Fechado', 'True': 'Aberto', 'False': 'Fechado'}
    ).value_counts()
    save_table(counts.reset_index().rename(columns={'is_oa': 'tipo', 'count': 'n'}),
               '08_open_access')

    colors = ['#2C5F8A', '#A8C4DC']
    fig, ax = plt.subplots(figsize=(FIG_W * 0.7, FIG_W * 0.7))
    ax.pie(counts.values, labels=counts.index, colors=colors,
           autopct='%1.1f%%', startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.set_title('Acesso aberto vs. fechado')
    save(fig, '08_open_access')


def score_distribution(df_all: pd.DataFrame):
    df_scored = df_all[df_all['score_triagem'] > 0].copy()
    bins   = list(range(60, 101, 5)) + [101]
    labels = ['60-64','65-69','70-74','75-79','80-84',
              '85-89','90-94','95-99','100']
    df_inc = df_scored[df_scored['decisao'] == 'INCLUIR'].copy()
    df_inc['faixa'] = pd.cut(df_inc['score_triagem'],
                              bins=bins, labels=labels, right=False)
    counts = df_inc['faixa'].value_counts().sort_index().reset_index()
    counts.columns = ['faixa', 'n']
    save_table(counts, '09_score_distribution')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_S))
    bars = ax.bar(counts['faixa'], counts['n'],
                  color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xlabel('Faixa de score de triagem')
    ax.set_ylabel('Numero de artigos')
    ax.set_title('Distribuicao de scores de triagem (incluidos)')
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, counts['n']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=7)
    save(fig, '09_score_distribution')


def temporal_by_field(df: pd.DataFrame):
    def get_period(year: int) -> str:
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    df = df.copy()
    df['period'] = df['publication_year'].apply(get_period)

    top_fields = df['primary_field'].value_counts().head(5).index
    df_f = df[df['primary_field'].isin(top_fields)]
    pivot = (df_f.groupby(['period', 'primary_field'])
             .size().unstack(fill_value=0)
             .reindex(PERIOD_ORDER).fillna(0))

    save_table(pivot.reset_index(), '10_temporal_by_field')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    for col in pivot.columns:
        ax.plot(PERIOD_ORDER, pivot[col], linewidth=1.5,
                marker='o', markersize=3, label=col)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Numero de artigos')
    ax.set_title('Evolucao temporal por area de conhecimento')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Area', fontsize=7, title_fontsize=8)
    save(fig, '10_temporal_by_field')


def top_topics(df: pd.DataFrame, n: int = 30):
    all_kws = parse_list_col(df['keyword_names'])
    counts = (pd.Series(Counter(all_kws))
              .sort_values(ascending=False).head(n)
              .reset_index())
    counts.columns = ['topico', 'n']
    save_table(counts, '11_top_topics')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_L))
    bars = ax.barh(counts['topico'][::-1], counts['n'][::-1],
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xlabel('Frequencia')
    ax.set_title(f'Top {n} topicos (gerados pelo OpenAlex)')
    for bar, val in zip(bars, counts['n'][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=7)
    save(fig, '11_top_topics')


def top_cited_articles(df: pd.DataFrame, n: int = 20):
    df_cit = (df[df['cited_by_count'] > 0]
              [['title', 'first_author', 'publication_year',
                'cited_by_count', 'journal', 'doi']]
              .sort_values('cited_by_count', ascending=False)
              .head(n)
              .reset_index(drop=True))
    save_table(df_cit, '12_top_cited')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    labels = [f'{row.first_author or "?"} ({int(row.publication_year)})'
              for row in df_cit.itertuples()]
    bars = ax.barh(labels[::-1], df_cit['cited_by_count'][::-1].values,
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xlabel('Numero de citacoes')
    ax.set_title(f'Top {n} artigos mais citados')
    for bar, val in zip(bars, df_cit['cited_by_count'][::-1].values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va='center', ha='left', fontsize=7)
    plt.tight_layout()
    save(fig, '12_top_cited')


def temporal_by_country_regions(df: pd.DataFrame, n_top: int = 5):
    all_countries = parse_list_col(df['countries'])
    top5 = {c for c, _ in Counter(all_countries).most_common(n_top)}

    def get_period(year: int) -> str:
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    rows = []
    for _, row in df.iterrows():
        try:
            countries = ast.literal_eval(row['countries']) if isinstance(row['countries'], str) else row['countries']
            seen_regions = set()
            for c in (countries or []):
                region = get_region(c, top5)
                if region and region not in seen_regions:
                    rows.append({
                        'period': get_period(int(row['publication_year'])),
                        'region': region
                    })
                    seen_regions.add(region)
        except:
            pass

    if not rows:
        return

    df_r  = pd.DataFrame(rows)
    pivot = df_r.groupby(['period', 'region']).size().unstack(fill_value=0)
    pivot = pivot.reindex(PERIOD_ORDER).fillna(0)

    top5_sorted = [c for c, _ in Counter(all_countries).most_common(n_top) if c in pivot.columns]
    regional    = [c for c in ['Demais Americas', 'Demais Europa', 'Asia, Oceania e Africa']
                   if c in pivot.columns]
    pivot = pivot[top5_sorted + regional]

    save_table(pivot.reset_index(), '13_temporal_by_country_regions')


    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    for col in top5_sorted:
        ax.plot(PERIOD_ORDER, pivot[col], linewidth=1.5, marker='o', markersize=3, label=col)
    for col in regional:
        ax.plot(PERIOD_ORDER, pivot[col], linewidth=1.2, linestyle='--',
                marker='s', markersize=3, label=col)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Numero de artigos')
    ax.set_title('Evolucao temporal - top-5 paises e demais regioes')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Pais / Regiao', fontsize=6, title_fontsize=7)
    save(fig, '13_temporal_by_country_regions')

def citations_by_field(df: pd.DataFrame):
    df_c = df[df['cited_by_count'] > 0].copy()
    stats = (df_c.groupby('primary_field')['cited_by_count']
             .agg(['median', 'mean', 'sum', 'count'])
             .sort_values('sum', ascending=False)
             .reset_index()
             .rename(columns={'primary_field': 'area', 'median': 'mediana',
                               'mean': 'media', 'sum': 'total', 'count': 'n'}))
    save_table(stats, '14_citations_by_field')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    bars = ax.barh(stats['area'][::-1], stats['mediana'][::-1],
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    ax.set_xlabel('Mediana de citacoes')
    ax.set_title('Mediana de citacoes por area de conhecimento')
    for bar, val in zip(bars, stats['mediana'][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}', va='center', ha='left', fontsize=7)
    save(fig, '14_citations_by_field')


def dimensional_profile_by_field(df: pd.DataFrame):
    import numpy as np

    dims   = ['centralidade_pobreza', 'centralidade_migracao', 'relacao_direta']
    labels = ['Pobreza', 'Migracao', 'Relacao direta']
    colors = ['#2C5F8A', '#E07B2A', '#2A9E4F']

    for col in dims:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def get_period(year):
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    df = df.copy()
    df['period'] = df['publication_year'].apply(get_period)
    df['group']  = df['primary_field'].apply(
        lambda x: 'Social Sciences' if x == 'Social Sciences' else 'Demais areas')

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 1.5, FIG_H_M), sharey=True)

    all_tables = []

    for ax, group in zip(axes, ['Social Sciences', 'Demais areas']):
        df_g = df[df['group'] == group]
        for dim, label, color in zip(dims, labels, colors):
            stats = df_g.groupby('period')[dim].agg(['mean','std']).reindex(PERIOD_ORDER)
            mean  = stats['mean'].values
            std   = stats['std'].fillna(0).values
            x     = range(len(PERIOD_ORDER))

            ax.plot(PERIOD_ORDER, mean, color=color, linewidth=1.8,
                    marker='o', markersize=3, label=label)
            ax.fill_between(PERIOD_ORDER,
                            mean - std, mean + std,
                            color=color, alpha=0.15)

            for p, m, s in zip(PERIOD_ORDER, mean, std):
                all_tables.append({'grupo': group, 'periodo': p,
                                   'dimensao': label, 'media': m, 'dp': s})

        ax.set_title(group, fontsize=9)
        ax.set_xlabel('Periodo')
        ax.set_ylim(0, 10.5)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Score medio (0-10)')
    fig.suptitle('Perfil dimensional por periodo', fontsize=10, y=1.01)
    plt.tight_layout()

    save_table(pd.DataFrame(all_tables), '15_dimensional_profile_by_field')
    save(fig, '15_dimensional_profile_by_field')


def disciplinary_diversity(df: pd.DataFrame):
    def get_period(year):
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    df = df.copy()
    df['period'] = df['publication_year'].apply(get_period)
    pivot = (df.groupby(['period', 'primary_field']).size()
               .unstack(fill_value=0).reindex(PERIOD_ORDER).fillna(0))

    def shannon(row):
        total = row.sum()
        if total == 0: return 0
        p = row / total
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    entropy = pivot.apply(shannon, axis=1).reset_index()
    entropy.columns = ['periodo', 'entropia']
    save_table(entropy, '16_disciplinary_diversity')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_S))
    ax.plot(entropy['periodo'], entropy['entropia'],
            color=STYLE['color'], linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Entropia de Shannon')
    ax.set_title('Diversidade disciplinar por periodo')
    plt.xticks(rotation=45, ha='right')
    save(fig, '16_disciplinary_diversity')


def oa_vs_citations(df: pd.DataFrame):
    df_c = df[df['cited_by_count'] > 0].copy()
    df_c['oa_label'] = df_c['is_oa'].map(
        {True: 'Aberto', False: 'Fechado', 'True': 'Aberto', 'False': 'Fechado'})
    stats = df_c.groupby('oa_label')['cited_by_count'].agg(['median','mean','count']).reset_index()
    stats.columns = ['tipo', 'mediana', 'media', 'n']
    save_table(stats, '17_oa_vs_citations')

    fig, ax = plt.subplots(figsize=(FIG_W * 0.6, FIG_H_S))
    bars = ax.bar(stats['tipo'], stats['mediana'],
                  color=[STYLE['color'], '#A8C4DC'],
                  edgecolor=STYLE['edgecolor'], linewidth=0.4)
    for bar, val in zip(bars, stats['mediana']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Mediana de citacoes')
    ax.set_title('Citacoes medianas: acesso aberto vs. fechado')
    save(fig, '17_oa_vs_citations')


def citation_longevity(df: pd.DataFrame):
    df_c = df[df['cited_by_count'] > 0].copy()
    df_c['age'] = 2024 - df_c['publication_year']
    df_c = df_c[df_c['age'] > 0]
    df_c['cit_per_year'] = df_c['cited_by_count'] / df_c['age']

    top = df_c.nlargest(20, 'cit_per_year')[
        ['title', 'first_author', 'publication_year', 'cited_by_count', 'cit_per_year']]
    save_table(top, '18_citation_longevity')

    # scatter: ano × citações/ano, tamanho = total citações
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    sc = ax.scatter(df_c['publication_year'], df_c['cit_per_year'],
                    s=df_c['cited_by_count'] / 10,
                    alpha=0.4, color=STYLE['color'], edgecolors='none')
    ax.set_xlabel('Ano de publicacao')
    ax.set_ylabel('Citacoes por ano de vida')
    ax.set_title('Longevidade da citacao\n(tamanho = total de citacoes)')
    save(fig, '18_citation_longevity')


def title_term_evolution(df: pd.DataFrame, terms: list = None):
    if terms is None:
        terms = ['remittance', 'displacement', 'refugee', 'climate',
                 'inequality', 'urbanization', 'rural', 'internal']

    def get_period(year):
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    df = df.copy()
    df['period'] = df['publication_year'].apply(get_period)
    df['title_lower'] = df['title'].fillna('').str.lower()

    rows = []
    for period, grp in df.groupby('period'):
        total = len(grp)
        for term in terms:
            n = grp['title_lower'].str.contains(term, na=False).sum()
            rows.append({'period': period, 'term': term, 'pct': n / total * 100})

    result = pd.DataFrame(rows)
    save_table(result, '19_title_term_evolution')

    pivot = result.pivot(index='period', columns='term', values='pct').reindex(PERIOD_ORDER)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    for col in pivot.columns:
        ax.plot(PERIOD_ORDER, pivot[col], linewidth=1.5,
                marker='o', markersize=3, label=col)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('% dos titulos contendo o termo')
    ax.set_title('Evolucao de termos nos titulos')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Termo', fontsize=6, title_fontsize=7, ncol=2)
    save(fig, '19_title_term_evolution')


def international_collaboration(df: pd.DataFrame):
    def get_period(year):
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    df = df.copy()
    df['period'] = df['publication_year'].apply(get_period)
    df['intl'] = pd.to_numeric(df['countries_distinct_count'], errors='coerce') > 1

    stats = df.groupby('period')['intl'].agg(['sum','count']).reindex(PERIOD_ORDER).fillna(0)
    stats['pct'] = stats['sum'] / stats['count'] * 100
    stats = stats.reset_index()
    stats.columns = ['periodo', 'n_intl', 'total', 'pct_intl']
    save_table(stats, '20_international_collaboration')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_S))
    ax.bar(stats['periodo'], stats['pct_intl'],
           color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    for i, val in enumerate(stats['pct_intl']):
        ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('% de artigos com coautoria internacional')
    ax.set_title('Colaboracao internacional por periodo')
    plt.xticks(rotation=45, ha='right')
    save(fig, '20_international_collaboration')


def authors_per_article(df: pd.DataFrame):
    def get_period(year):
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    df = df.copy()
    df['period']       = df['publication_year'].apply(get_period)
    df['author_count'] = pd.to_numeric(df['author_count'], errors='coerce')
    df = df[df['author_count'] > 0]

    data_by_period = [df[df['period'] == p]['author_count'].dropna().tolist()
                      for p in PERIOD_ORDER]

    stats = df.groupby('period')['author_count'].agg(
        ['median','mean','count']).reindex(PERIOD_ORDER).reset_index()
    stats.columns = ['periodo','mediana','media','n']
    save_table(stats, '21_authors_per_article')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_M))
    ax.boxplot(data_by_period,
               labels=PERIOD_ORDER,
               patch_artist=True,
               boxprops=dict(facecolor=STYLE['color'], alpha=0.7),
               medianprops=dict(color='white', linewidth=1.5),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8),
               flierprops=dict(marker='o', markersize=2,
                               alpha=0.3, color=STYLE['color']),
               showfliers=True)
    ax.set_yscale('log')
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Numero de autores por artigo (escala log)')
    ax.set_title('Numero de autores por artigo por periodo')
    plt.xticks(rotation=45, ha='right')
    save(fig, '21_authors_per_article')


def source_type_distribution(df: pd.DataFrame):
    def get_period(year):
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    counts = (df['source_type'].dropna()
              .value_counts()
              .reset_index()
              .rename(columns={'source_type': 'tipo', 'count': 'n'}))
    save_table(counts, '22_source_type')

    fig, ax = plt.subplots(figsize=(FIG_W * 0.8, FIG_H_S))
    bars = ax.barh(counts['tipo'][::-1], counts['n'][::-1],
                   color=STYLE['color'], edgecolor=STYLE['edgecolor'], linewidth=0.4)
    for bar, val in zip(bars, counts['n'][::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', ha='left', fontsize=7)
    ax.set_xlabel('Numero de artigos')
    ax.set_title('Distribuicao por tipo de publicacao')
    save(fig, '22_source_type')

    # evolucao temporal por tipo
    df2 = df.copy()
    df2['period'] = df2['publication_year'].apply(get_period)
    top_types = df2['source_type'].value_counts().head(5).index
    df2 = df2[df2['source_type'].isin(top_types)]
    pivot = (df2.groupby(['period', 'source_type']).size()
               .unstack(fill_value=0).reindex(PERIOD_ORDER).fillna(0))
    save_table(pivot.reset_index(), '22b_source_type_temporal')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_S))
    for col in pivot.columns:
        ax.plot(PERIOD_ORDER, pivot[col], linewidth=1.5,
                marker='o', markersize=3, label=col)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Numero de artigos')
    ax.set_title('Evolucao temporal por tipo de publicacao')
    plt.xticks(rotation=45, ha='right')
    ax.legend(fontsize=7)
    save(fig, '22b_source_type_temporal')


def oa_by_period(df: pd.DataFrame):
    def get_period(year):
        if year <= 1989:   return 'ate 1989'
        elif year <= 1994: return '1990-1994'
        elif year <= 1999: return '1995-1999'
        elif year <= 2004: return '2000-2004'
        elif year <= 2009: return '2005-2009'
        elif year <= 2014: return '2010-2014'
        elif year <= 2019: return '2015-2019'
        else:              return '2020-2024'

    PERIOD_ORDER = ['ate 1989','1990-1994','1995-1999','2000-2004',
                    '2005-2009','2010-2014','2015-2019','2020-2024']

    df2 = df.copy()
    df2['period']    = df2['publication_year'].apply(get_period)
    df2['is_oa_bool'] = df2['is_oa'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0)

    stats = df2.groupby('period')['is_oa_bool'].agg(
        ['sum', 'count']).reindex(PERIOD_ORDER).fillna(0)
    stats['pct_oa'] = stats['sum'] / stats['count'] * 100
    stats = stats.reset_index()
    stats.columns = ['periodo', 'n_oa', 'total', 'pct_oa']
    save_table(stats, '23_oa_by_period')

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_S))
    ax.plot(stats['periodo'], stats['pct_oa'],
            color=STYLE['color'], linewidth=2, marker='o', markersize=4)
    for i, val in enumerate(stats['pct_oa']):
        ax.text(i, val + 0.8, f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xlabel('Periodo')
    ax.set_ylabel('% de artigos open access')
    ax.set_title('Evolucao do acesso aberto por periodo')
    ax.set_ylim(0, max(stats['pct_oa']) * 1.2)
    plt.xticks(rotation=45, ha='right')
    save(fig, '23_oa_by_period')


# -- Chamando para execução e salvamento
def run_all(config: dict):
    setup_style()
    repositorios           = set(config['sources']['repositories'])
    autores_institucionais = set(config['sources']['institutional_authors'])

    print('\n=== Analise do corpus ===\n')

    df_all      = load_flat(only_included=False)
    df_included = load_flat(only_included=True)

    print('[01] Distribuicao temporal')
    temporal_distribution(df_included)

    print('[02] Top periodicos')
    top_journals(df_included, repositorios)

    print('[03] Top autores')
    top_authors(df_included, autores_institucionais)

    print('[04] Areas de conhecimento')
    top_fields(df_included)

    print('[05] Paises')
    top_countries(df_included)

    print('[06] Instituicoes')
    top_institutions(df_included)

    print('[07] Citacoes')
    citation_distribution(df_included)

    print('[08] Acesso aberto')
    open_access_share(df_included)

    print('[09] Scores de triagem')
    score_distribution(df_all)

    print('[10] Temporal por area')
    temporal_by_field(df_included)

    print('[11] Top topicos')
    top_topics(df_included)

    print('[12] Top citados')
    top_cited_articles(df_included)

    print('[13] Temporal por pais e regioes')
    temporal_by_country_regions(df_included)

    print('[14] Citacoes por area')
    citations_by_field(df_included)

    print('[15] Perfil dimensional por area')
    dimensional_profile_by_field(df_included)

    print('[16] Diversidade disciplinar')
    disciplinary_diversity(df_included)

    print('[17] OA vs citacoes')
    oa_vs_citations(df_included)

    print('[18] Longevidade da citacao')
    citation_longevity(df_included)

    print('[19] Evolucao de termos nos titulos')
    title_term_evolution(df_included)

    print('[20] Colaboracao internacional')
    international_collaboration(df_included)

    print('[21] Autores por artigo')
    authors_per_article(df_included)

    print('[22] Tipo de publicacao')
    source_type_distribution(df_included)

    print('[23] OA por periodo')
    oa_by_period(df_included)

    print('\nConcluido.')