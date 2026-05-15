"""
MSL Dashboard — Pobreza e Migração
Mapeamento Sistemático da Literatura
"""

import re
import ast
import json
import nltk
import yaml
import numpy                as np
import pandas               as pd
import streamlit            as st
import plotly.io            as pio
import plotly.express       as px
import plotly.graph_objects as go

from io                     import BytesIO
from pathlib                import Path
from collections            import Counter
from wordcloud              import WordCloud
from nltk.corpus            import stopwords
from nltk.tokenize          import word_tokenize
from nltk.stem              import WordNetLemmatizer

# --- NLTK setup ---
for resource in ['stopwords', 'punkt', 'punkt_tab', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

BASE = Path(__file__).parent

PRIMARY  = '#2C5F8A'
DARK     = '#1A3F5C'
LIGHT    = '#A8C4DC'
ACCENT   = '#E07B2A'
MUTED    = '#6B8FAD'

PERIOD_ORDER = [
    'até 1989','1990–1994','1995–1999','2000–2004',
    '2005–2009','2010–2014','2015–2019','2020–2024'
]

AMERICAS = {
    'United States','Canada','Mexico','Brazil','Argentina','Colombia',
    'Chile','Peru','Venezuela','Ecuador','Bolivia','Paraguay','Uruguay',
    'Guyana','Suriname','Trinidad and Tobago','Jamaica','Cuba','Haiti',
    'Dominican Republic','Guatemala','Honduras','El Salvador','Nicaragua',
    'Costa Rica','Panama','Belize','Barbados','Bahamas','Grenada',
    'Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines',
    'Antigua and Barbuda','Dominica',
}
EUROPE = {
    'United Kingdom','Germany','France','Netherlands','Sweden','Spain',
    'Italy','Switzerland','Norway','Belgium','Austria','Denmark',
    'Finland','Ireland','Portugal','Poland','Czech Republic','Hungary',
    'Romania','Bulgaria','Greece','Croatia','Slovakia','Slovenia',
    'Estonia','Latvia','Lithuania','Luxembourg','Malta','Cyprus',
    'Iceland','Liechtenstein','Monaco','Andorra','Serbia','Montenegro',
    'Bosnia and Herzegovina','Albania','North Macedonia','Moldova',
    'Ukraine','Belarus','Russia','Turkey',
}
AFRICA = {
    'Nigeria','South Africa','Ethiopia','Egypt','Kenya','Ghana',
    'Tanzania','Uganda','Mozambique','Zambia','Zimbabwe','Malawi',
    'Senegal','Mali','Niger','Chad','Sudan','Somalia','Rwanda',
    'Burundi','Congo (DRC)','Congo','Cameroon','Angola','Namibia',
    'Botswana','Libya','Tunisia','Algeria','Morocco','Mauritania',
    'Djibouti','Eritrea','Eswatini','Lesotho','Madagascar',
    'Mauritius','Seychelles','Cabo Verde','Guinea','Guinea-Bissau',
    'Sierra Leone','Liberia','Togo','Benin','Equatorial Guinea',
    'Sao Tome and Principe','Comoros','South Sudan','Burkina Faso',
    'Central African Republic','Gabon',
}

ASIA_OCEANIA = {
    'China','India','Japan','South Korea','Indonesia','Pakistan',
    'Bangladesh','Vietnam','Philippines','Thailand','Malaysia',
    'Singapore','Sri Lanka','Nepal','Cambodia','Myanmar','Laos',
    'Kazakhstan','Uzbekistan','Azerbaijan','Georgia','Armenia',
    'Iran','Iraq','Saudi Arabia','United Arab Emirates','Kuwait',
    'Qatar','Bahrain','Oman','Jordan','Lebanon','Israel','Syria',
    'Yemen','Afghanistan','Taiwan','Maldives','Bhutan','Brunei',
    'Timor-Leste','Mongolia','Kyrgyzstan','Tajikistan','Turkmenistan',
    'Australia','New Zealand','Papua New Guinea','Fiji','Solomon Islands',
    'Vanuatu','Samoa','Tonga','Kiribati','Micronesia','Palau',
    'Nauru','Tuvalu','Marshall Islands',
}

def country_to_region(c):
    if c in AMERICAS:     return 'Américas'
    if c in EUROPE:       return 'Europa'
    if c in AFRICA:       return 'África'
    if c in ASIA_OCEANIA: return 'Ásia e Oceania'
    return 'Outros'


HEALTH = {
    'Health Professions','Medicine','Nursing',
    'Psychology','Immunology and Microbiology',
    'Pharmacology, Toxicology and Pharmaceutics',
    'Biochemistry, Genetics and Molecular Biology',
}
AGRI_ENV = {
    'Agricultural and Biological Sciences',
    'Environmental Science','Earth and Planetary Sciences',
}

def reclassify_field(f):
    if pd.isna(f):                                  return 'Others'
    if f == 'Social Sciences':                      return 'Social Sciences'
    if f == 'Economics, Econometrics and Finance':  return 'Economics Sciences'
    if f in HEALTH:                                 return 'Health, Psychology & Medicine'
    if f in AGRI_ENV:                               return 'Agriculture & Environment'
    return 'Others'


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_corpus():
    path = BASE / 'data' / 'processed' / 'flat_corpus.csv'
    df = pd.read_csv(path)
    df = df[df['decisao'] == 'INCLUIR'].reset_index(drop=True)
    df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
    df['cited_by_count']   = pd.to_numeric(df['cited_by_count'],   errors='coerce').fillna(0)
    df['period'] = df['publication_year'].apply(get_period)
    for col in ['countries', 'institutions', 'keyword_names']:
        df[col] = df[col].apply(safe_parse_list)
    df['region'] = df['countries'].apply(
        lambda lst: list({country_to_region(c) for c in lst}) if lst else []
    )
    return df


@st.cache_data
def load_metrics():
    path = BASE / 'data' / 'metrics.json'
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_config():
    path = BASE / 'config' / 'poverty_migration.yaml'
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_period(year):
    try:
        y = int(year)
    except Exception:
        return 'Desconhecido'
    if y <= 1989:   return 'até 1989'
    elif y <= 1994: return '1990–1994'
    elif y <= 1999: return '1995–1999'
    elif y <= 2004: return '2000–2004'
    elif y <= 2009: return '2005–2009'
    elif y <= 2014: return '2010–2014'
    elif y <= 2019: return '2015–2019'
    else:           return '2020–2024'


def safe_parse_list(val):
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or val.strip() in ('', 'nan', '[]'):
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def explode_list_col(df, col):
    return (
        df[df[col].map(len) > 0]
        .explode(col)
        .dropna(subset=[col])
    )


def apply_filters(df, periods=None, regions=None, countries=None,
                  fields=None, oa=None, institutions=None, journals=None):
    mask = pd.Series([True] * len(df), index=df.index)
    if periods:
        mask &= df['period'].isin(periods)
    if regions:
        mask &= df['region'].apply(lambda lst: any(r in lst for r in regions))
    if countries:
        mask &= df['countries'].apply(lambda lst: any(c in lst for c in countries))
    if institutions:
        mask &= df['institutions'].apply(lambda lst: any(i in lst for i in institutions))
    if fields:
        mask &= df['primary_field'].isin(fields)
    if oa is not None:
        oa_bool = df['is_oa'].map(
            {True: True, False: False, 'True': True, 'False': False}
        ).fillna(False)
        mask &= (oa_bool == oa)
    if journals:
        mask &= df['journal'].isin(journals)
    return df[mask].reset_index(drop=True)


# ─────────────────────────────────────────────
# NLP
# ─────────────────────────────────────────────

STOP_BASE = set(stopwords.words('english')) | {
    'study','paper','using','data','result','results','model',
    'based','used','found','also','may','two','one','new',
    'although','however','therefore','thus','hence','since',
    'while','whereas','despite','whether','either',
    'both','neither','each','every','much','many','several',
    'various','certain','particular','general','specific',
    'likely','possible','potential','similar',
    'main','major','key','primary',
    'show','shows','find','finds','use','uses',
    'include','includes','suggest','suggests','indicate',
    'indicates','provide','provides','examine','examines',
    'explore','explores','investigate','investigates','present',
    'presents','discuss','discusses','argue','argues','focus',
    'focuses','address','addresses','contribute','contributes',
    'abstract','introduction','conclusion','review',
    'article','research','survey','sample',
    'method','methods','approach','analysis',
    'ha','wa','searched','finding','findings','lmic',
    'associated','significant','importance','important',
    'higher','lower','greater','less','high','low',
    'large','small','first','second','third','recent',
    'different','various',
}

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_lemmatize(text):
    return [lemmatizer.lemmatize(w)
            for w in word_tokenize(clean_text(text))
            if len(w) > 2 and w.isalpha()]

def get_bigrams(tokens, banned, stop):
    result = []
    for a, b in zip(tokens, tokens[1:]):
        if len(a) < 3 or len(b) < 3: continue
        if a in stop or b in stop: continue
        bg = f'{a} {b}'
        if bg not in banned:
            result.append(bg)
    return result

def get_unigrams(tokens, stop_display):
    return [t for t in tokens if t not in stop_display]

@st.cache_data(show_spinner=False)
def compute_nlp(texts_tuple, banned_tuple, stop_display_tuple):
    banned       = set(banned_tuple)
    stop_display = set(stop_display_tuple)
    uni_all, bi_all = [], []
    for text in texts_tuple:
        tokens = tokenize_lemmatize(text)
        uni_all.extend(get_unigrams(tokens, stop_display))
        bi_all.extend(get_bigrams(tokens, banned, STOP_BASE))
    return Counter(uni_all), Counter(bi_all)

@st.cache_data(show_spinner=False)
def compute_wordcloud_img(texts_tuple, banned_tuple, stop_display_tuple):
    _, bi = compute_nlp(texts_tuple, banned_tuple, stop_display_tuple)
    uni_c, _ = compute_nlp(texts_tuple, banned_tuple, stop_display_tuple)
    freq = dict(uni_c)
    freq.update({k.replace(' ', '_'): v for k, v in bi.items()})
    if not freq:
        return None
    def color_func(word, font_size, **kwargs):
        if font_size >= 80: return '#0D2B45'
        elif font_size >= 50: return '#1A3F5C'
        elif font_size >= 30: return '#2C5F8A'
        elif font_size >= 15: return '#5B8DB8'
        else: return '#A8C4DC'
    wc = WordCloud(width=1400, height=700, background_color='white',
                   max_words=120, collocations=False,
                   color_func=color_func).generate_from_frequencies(freq)
    buf = BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    return buf

@st.cache_data(show_spinner=False)
def compute_emerging_declining(df_early_texts, df_late_texts, banned_tuple, stop_tuple, min_freq=3):
    banned = set(banned_tuple)
    stop   = set(stop_tuple)

    def pct_dict(texts):
        total = max(len(texts), 1)
        c = Counter()
        for t in texts:
            tokens = tokenize_lemmatize(t)
            c.update(get_bigrams(tokens, banned, stop))
        return {k: v/total*100 for k, v in c.items() if v >= min_freq}

    early = pct_dict(df_early_texts)
    late  = pct_dict(df_late_texts)
    rows = []
    for term in set(early) | set(late):
        e, l = early.get(term, 0), late.get(term, 0)
        rows.append({'termo': term, 'pct_early': e, 'pct_late': l, 'delta': l - e})
    return pd.DataFrame(rows).sort_values('delta', ascending=False)


# ─────────────────────────────────────────────
# PAGE CONFIG & STYLE
# ─────────────────────────────────────────────

pio.templates['msl'] = go.layout.Template(
    layout=dict(
        font=dict(family='Source Sans 3', size=12),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
)
pio.templates.default = 'msl'

st.set_page_config(
    page_title='MSL Pobreza & Migração',
    page_icon='📚',
    layout='wide',
    initial_sidebar_state='collapsed',
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif;
}}

.main-header {{
    background: linear-gradient(135deg, {DARK} 0%, {PRIMARY} 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 0 0 16px 16px;
    margin-bottom: 2rem;
}}
.main-header h1 {{
    font-family: 'Libre Baskerville', serif;
    color: white;
    font-size: 2rem;
    margin: 0;
    letter-spacing: -0.5px;
}}
.main-header p {{
    color: {LIGHT};
    margin: 0.4rem 0 0 0;
    font-size: 0.95rem;
    font-weight: 300;
}}

.metric-card {{
    background: white;
    border: 1px solid #E8EDF2;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border-left: 4px solid {PRIMARY};
}}
.metric-card .label {{
    font-size: 0.78rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}}
.metric-card .value {{
    font-size: 2rem;
    font-weight: 700;
    color: {DARK};
    line-height: 1.2;
}}
.metric-card .sub {{
    font-size: 0.8rem;
    color: #888;
    margin-top: 2px;
}}

.section-title {{
    font-family: 'Libre Baskerville', serif;
    color: {DARK};
    font-size: 1.15rem;
    border-bottom: 2px solid {LIGHT};
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}}

.tag {{
    display: inline-block;
    background: {LIGHT};
    color: {DARK};
    border-radius: 20px;
    padding: 3px 12px;
    margin: 3px;
    font-size: 0.82rem;
    font-weight: 600;
}}
.tag-b {{
    background: {PRIMARY};
    color: white;
}}

div[data-testid="stTabs"] button {{
    font-family: 'Source Sans 3', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

df_full  = load_corpus()
metrics  = load_metrics()
config   = load_config()

banned_ngrams   = tuple(config.get('keywords', {}).get('banned_ngrams', []))
stop_artifact   = set(config.get('keywords', {}).get('stop_artifact', []))
stop_display_ex = set(config.get('keywords', {}).get('stop_display_unigrams', []))
STOP_DISPLAY    = STOP_BASE | stop_artifact | stop_display_ex
STOP_BASE.update(stop_artifact)

repositories = set(config.get('sources', {}).get('repositories', []))
inst_autores = set(config.get('sources', {}).get('institutional_authors', []))

# derived option lists
all_periods   = [p for p in PERIOD_ORDER if p in df_full['period'].unique()]
all_regions   = sorted({'Américas','Europa','Ásia e Oceania','África'})
all_countries = sorted({c for lst in df_full['countries'] for c in lst})
all_fields    = sorted(df_full['primary_field'].dropna().unique())
all_journals  = sorted(df_full[~df_full['journal'].isin(repositories)]['journal'].dropna().unique())
all_insts     = sorted({i for lst in df_full['institutions'] for i in lst})


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

proj = config.get('project', {})
st.markdown(f"""
<div class="main-header">
  <h1>📚 {proj.get('name', 'Mapeamento Sistemático da Literatura')}</h1>
  <p>{proj.get('description', '')}</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab0, tab1, tab2, tab3 = st.tabs([
    '🔬 Metodologia',
    '📊 Indicadores Gerais',
    '📅 Evolução Temporal',
    '🔤 Conteúdo & Tópicos',
])


# ═══════════════════════════════════════════
# ABA 0 — METODOLOGIA
# ═══════════════════════════════════════════
with tab0:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="section-title">Funil de pesquisa</div>', unsafe_allow_html=True)

        steps = [
            ('Artigos buscados (OpenAlex)', metrics.get('total_buscados', {}).get('n', 0)),
            ('Após deduplicação', metrics.get('total_buscados', {}).get('n', 0)
             - metrics.get('removidos_duplicados', {}).get('n', 0)),
            ('Com título e abstract', metrics.get('corpus_para_triagem', {}).get('n', 0)),
            ('Incluídos após triagem', metrics.get('incluidos_apos_triagem', {}).get('n', 0)),
        ]

        fig_funnel = go.Figure(go.Funnel(
            y=[s[0] for s in steps],
            x=[s[1] for s in steps],
            textinfo='value+percent initial',
            marker=dict(color=[DARK, PRIMARY, MUTED, ACCENT]),
            connector=dict(line=dict(color='#E8EDF2', width=2)),
        ))
        fig_funnel.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=480,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='Source Sans 3'),
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Métricas rápidas
        m1, m2, m3, m4 = st.columns(4)
        for col, label, key, sub in [
            (m1, 'Buscados',  'total_buscados',         'OpenAlex'),
            (m2, 'Duplicatas', 'removidos_duplicados',   'removidos'),
            (m3, 'Sem abstract','removidos_sem_titulo_ou_abstract', 'removidos'),
            (m4, 'Incluídos', 'incluidos_apos_triagem',  'corpus final'),
        ]:
            val = metrics.get(key, {}).get('n', '—')
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{val:,}</div>
              <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-title">Protocolo de busca</div>', unsafe_allow_html=True)

        search = config.get('search', {})
        st.markdown(f"""
        **Base de dados:** OpenAlex  
        **Idioma:** {search.get('language','en').upper()}  
        **Período:** {search.get('date_from','?')} – {search.get('date_to','?')}  
        **Páginas por query:** {search.get('max_pages','?')}
        """)

        tg = search.get('term_groups', {})
        for grp, terms in tg.items():
            st.markdown(f'**{grp.upper()}**')
            st.markdown(' '.join(f'<span class="tag">{t}</span>' for t in terms),
                        unsafe_allow_html=True)

        st.markdown('<div class="section-title">Modelo de triagem</div>', unsafe_allow_html=True)
        scr = config.get('screening', {})
        st.markdown(f"""
        **Provedor:** {scr.get('provider','—')}  
        **Modelo:** `{scr.get('model','—')}`  
        **Threshold:** {scr.get('threshold','—')} pontos  
        **Sleep:** {scr.get('sleep','—')} s entre requisições
        """)

        dims = scr.get('dimensions', [])
        if dims:
            st.markdown('**Dimensões avaliadas:**')
            for d in dims:
                st.markdown(f'- `{d["name"]}`: {d["description"]}')

        # Download do corpus completo
        st.markdown('<div class="section-title">Download</div>', unsafe_allow_html=True)
        csv_bytes = df_full.to_csv(index=False).encode('utf-8')
        st.download_button(
            '⬇ Baixar corpus completo (CSV)',
            data=csv_bytes,
            file_name='corpus_msl.csv',
            mime='text/csv',
        )


# ═══════════════════════════════════════════
# ABA 1 — INDICADORES GERAIS
# ═══════════════════════════════════════════
with tab1:
    # Filtros
    with st.expander('🎛 Filtros', expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        f1_periods  = fc1.multiselect('Período',                all_periods,    key='t1_period')
        f1_regions  = fc2.multiselect('Região',                 all_regions,    key='t1_region')
        f1_countries= fc3.multiselect('País',                   all_countries,  key='t1_country')
        f1_insts    = fc4.multiselect('Instituição',            all_insts,      key='t1_inst')
        fc5, fc6, fc7, _ = st.columns(4)
        f1_fields   = fc5.multiselect('Área do conhecimento',   all_fields,     key='t1_field')
        f1_journals = fc6.multiselect('Periódico',              all_journals,   key='t1_journal')
        f1_oa_sel   = fc7.selectbox( 'Acesso', ['Todos','Aberto','Fechado'],    key='t1_oa')

    f1_oa = True if f1_oa_sel == 'Aberto' else (False if f1_oa_sel == 'Fechado' else None)
    df1 = apply_filters(df_full, f1_periods, f1_regions, f1_countries,
                        f1_fields, f1_oa, f1_insts, f1_journals)

    st.caption(f'**{len(df1):,}** artigos após filtros')

    # Download filtrado
    st.download_button('⬇ Baixar seleção (CSV)',
                       data=df1.to_csv(index=False).encode('utf-8'),
                       file_name='corpus_filtrado.csv', mime='text/csv')

    # Linha 1: temporal + periódicos
    r1c1, r1c2 = st.columns([3, 2])
    with r1c1:
        st.markdown('<div class="section-title">Evolução temporal</div>', unsafe_allow_html=True)
        yr = df1.groupby('publication_year').size().reset_index(name='n')
        fig_yr = px.bar(yr, x='publication_year', y='n',
                        color_discrete_sequence=[PRIMARY])
        fig_yr.update_layout(xaxis_title='Ano', yaxis_title='Artigos',
                             margin=dict(t=10,b=10,l=10,r=10), height=480,
                             plot_bgcolor='white', paper_bgcolor='white')
        fig_yr.update_xaxes(showgrid=False)
        fig_yr.update_yaxes(showgrid=True, gridcolor='#EEF1F4')
        st.plotly_chart(fig_yr, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-title">Top periódicos</div>', unsafe_allow_html=True)
        journals_clean = df1[~df1['journal'].isin(repositories)]['journal'].dropna()
        top_j = journals_clean.value_counts().head(12).reset_index()
        top_j.columns = ['periódico', 'n']
        fig_j = px.bar(top_j, x='n', y='periódico', orientation='h',
                       color_discrete_sequence=[PRIMARY])
        fig_j.update_layout(yaxis={'categoryorder':'total ascending'},
                            xaxis_title='Artigos', yaxis_title='',
                            margin=dict(t=10,b=10,l=10,r=10), height=480,
                            plot_bgcolor='white', paper_bgcolor='white')
        fig_j.update_xaxes(showgrid=True, gridcolor='#EEF1F4')
        fig_j.update_yaxes(showgrid=False, tickfont=dict(size=12))
        st.plotly_chart(fig_j, use_container_width=True)

    # Linha 2: sunburst + áreas
    r2c1, r2c2 = st.columns([2, 3])
    with r2c1:
        st.markdown('<div class="section-title">País → Instituição → Autor</div>', unsafe_allow_html=True)
        sun_rows = []
        for _, row in df1.iterrows():
            countries_r = row['countries'] or []
            insts_r     = row['institutions'] or []
            author      = row.get('first_author') or 'Desconhecido'
            if author in inst_autores:
                author = 'Institucional'
            for c in countries_r[:1]:  # primeiro país
                for i in insts_r[:2]:  # até 2 instituições
                    sun_rows.append({'país': c, 'instituição': i, 'autor': author})
        if sun_rows:
            df_sun = pd.DataFrame(sun_rows)
            df_sun = df_sun.groupby(['país','instituição','autor']).size().reset_index(name='n')
            # top países
            top_countries_sun = df_sun.groupby('país')['n'].sum().nlargest(10).index
            df_sun = df_sun[df_sun['país'].isin(top_countries_sun)]
            fig_sun = px.sunburst(df_sun, path=['país','instituição','autor'], values='n',
                                  color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_sun.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=480)
            st.plotly_chart(fig_sun, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-title">Áreas de conhecimento</div>', unsafe_allow_html=True)
        fields_cnt = df1['primary_field'].dropna().value_counts().reset_index()
        fields_cnt.columns = ['área', 'n']
        fig_f = px.bar(fields_cnt, x='n', y='área', orientation='h',
                       color_discrete_sequence=[PRIMARY], log_x=True)
        fig_f.update_layout(yaxis={'categoryorder':'total ascending'},
                            xaxis_title='Artigos (escala log)', yaxis_title='',
                            margin=dict(t=10,b=10,l=10,r=10), height=480,
                            plot_bgcolor='white', paper_bgcolor='white')
        fig_f.update_xaxes(showgrid=True, gridcolor='#EEF1F4')
        fig_f.update_yaxes(tickfont=dict(size=12))
        st.plotly_chart(fig_f, use_container_width=True)

    # Linha 3: países + citações + OA + top-10
    r3c1, r3c2, r3c3 = st.columns([2, 2, 3])
    with r3c1:
        st.markdown('<div class="section-title">Países de afiliação</div>', unsafe_allow_html=True)
        all_c = [c for lst in df1['countries'] for c in lst]
        top_c = pd.Series(Counter(all_c)).sort_values(ascending=False).head(15).reset_index()
        top_c.columns = ['país', 'n']
        fig_c = px.bar(top_c, x='n', y='país', orientation='h',
                       color_discrete_sequence=[PRIMARY], log_x=True)
        fig_c.update_layout(yaxis={'categoryorder':'total ascending'},
                            xaxis_title='Artigos (log)', yaxis_title='',
                            margin=dict(t=10,b=10,l=10,r=10), height=480,
                            plot_bgcolor='white', paper_bgcolor='white')
        fig_c.update_xaxes(showgrid=True, gridcolor='#EEF1F4')
        fig_c.update_yaxes(tickfont=dict(size=12))
        st.plotly_chart(fig_c, use_container_width=True)

    with r3c2:
        st.markdown('<div class="section-title">Acesso aberto</div>', unsafe_allow_html=True)
        oa_cnt = df1['is_oa'].map(
            {True:'Aberto', False:'Fechado', 'True':'Aberto', 'False':'Fechado'}
        ).value_counts().reset_index()
        oa_cnt.columns = ['tipo','n']
        total = oa_cnt['n'].sum()
        oa_cnt['pct'] = oa_cnt['n'] / total * 100

        fig_oa = go.Figure()
        for _, row in oa_cnt.iterrows():
            color = PRIMARY if row['tipo'] == 'Aberto' else LIGHT
            fig_oa.add_trace(go.Bar(
                x=[row['pct']], y=[''],
                orientation='h',
                name=row['tipo'],
                marker_color=color,
                text=f"{row['tipo']}<br>{row['pct']:.1f}%",
                textposition='inside',
                insidetextanchor='middle',
            ))
        fig_oa.update_layout(
            barmode='stack',
            xaxis=dict(range=[0,100], showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False),
            margin=dict(t=10,b=10,l=10,r=10),
            height=100,
            showlegend=False,
            plot_bgcolor='white', paper_bgcolor='white',
        )
        st.plotly_chart(fig_oa, use_container_width=True)

        st.markdown('<div class="section-title">Citações</div>', unsafe_allow_html=True)
        bins   = [0,1,10,25,50,100,200,500,700,1000,9999]
        labels = ['0','1–9','10–24','25–49','50–99','100–199','200–499','500–699','700–999','1000+']
        df1_cit = df1.copy()
        df1_cit['faixa'] = pd.cut(df1_cit['cited_by_count'].clip(upper=3000),
                                   bins=bins, labels=labels, right=False)
        cit_cnt = df1_cit['faixa'].value_counts().sort_index().reset_index()
        cit_cnt.columns = ['faixa','n']
        fig_cit = px.bar(cit_cnt, x='faixa', y='n',
                         color_discrete_sequence=[PRIMARY])
        fig_cit.update_layout(xaxis_title='Citações', yaxis_title='Artigos',
                              yaxis_type='log',
                              margin=dict(t=10,b=10,l=10,r=10), height=280,
                              plot_bgcolor='white', paper_bgcolor='white')
        fig_cit.update_xaxes(showgrid=False, tickfont=dict(size=12))
        fig_cit.update_yaxes(showgrid=True, gridcolor='#EEF1F4')
        st.plotly_chart(fig_cit, use_container_width=True)

    with r3c3:
        st.markdown('<div class="section-title">Diversidade disciplinar (Entropia de Shannon)</div>',
                    unsafe_allow_html=True)
        entropy_rows = []
        for period in PERIOD_ORDER:
            df_p = df1[df1['period'] == period]['primary_field'].dropna()
            counts = df_p.value_counts().values
            total = counts.sum()
            if total == 0:
                entropy_rows.append({'period': period, 'entropia': 0})
                continue
            probs = counts / total
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log(probs))
            entropy_rows.append({'period': period, 'entropia': round(h, 3)})
        df_ent = pd.DataFrame(entropy_rows)
        fig_ent = px.line(df_ent, x='period', y='entropia', markers=True,
                        color_discrete_sequence=[ACCENT])
        fig_ent.update_traces(line_width=2.5, marker_size=7)
        fig_ent.update_layout(xaxis_title='Período', yaxis_title='Entropia de Shannon',
                            margin=dict(t=10,b=10,l=10,r=10), height=480,
                            plot_bgcolor='white', paper_bgcolor='white')
        fig_ent.update_xaxes(tickangle=45, showgrid=False, tickfont=dict(size=12))
        fig_ent.update_yaxes(showgrid=True, gridcolor='#EEF1F4')
        st.plotly_chart(fig_ent, use_container_width=True)

    st.markdown('<div class="section-title">Top 10 artigos mais citados</div>',
                unsafe_allow_html=True)
    top10 = (df1[df1['cited_by_count'] > 0]
            .sort_values('cited_by_count', ascending=False)
            .head(10)
            [['first_author','publication_year','title','cited_by_count','doi','journal']]
            .reset_index(drop=True))
    top10['Ano']       = top10['publication_year'].astype(int)
    top10['Citações']  = top10['cited_by_count'].astype(int)
    top10['Autor (Ano)'] = top10.apply(
        lambda r: f"{r['first_author'] or '?'} ({r['Ano']})", axis=1)
    top10['DOI'] = top10['doi'].apply(
        lambda d: f'[🔗]({d})' if pd.notna(d) and d else '—')
    st.dataframe(
        top10[['Autor (Ano)','title','Citações','journal','DOI']]
        .rename(columns={'title':'Título','journal':'Periódico'}),
        use_container_width=True, height=380,
        hide_index=True,
    )

# ═══════════════════════════════════════════
# ABA 2 — EVOLUÇÃO TEMPORAL
# ═══════════════════════════════════════════
with tab2:
    with st.expander('🎛 Filtros', expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        f2_periods  = fc1.multiselect('Período',                all_periods,    key='t2_period')
        f2_regions  = fc2.multiselect('Região',                 all_regions,    key='t2_region')
        f2_countries= fc3.multiselect('País',                   all_countries,  key='t2_country')
        f2_insts    = fc4.multiselect('Instituição',            all_insts,      key='t2_inst')
        fc5, fc6, fc7, _ = st.columns(4)
        f2_fields   = fc5.multiselect('Área do conhecimento',   all_fields,     key='t2_field')
        f2_journals = fc6.multiselect('Periódico',              all_journals,   key='t2_journal')
        f2_oa_sel   = fc7.selectbox( 'Acesso', ['Todos','Aberto','Fechado'],    key='t2_oa')

    f2_oa = True if f2_oa_sel == 'Aberto' else (False if f2_oa_sel == 'Fechado' else None)
    df2 = apply_filters(df_full, f2_periods, f2_regions, f2_countries,
                        f2_fields, f2_oa, f2_insts, f2_journals)
    st.caption(f'**{len(df2):,}** artigos após filtros')

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown('<div class="section-title">Áreas por período</div>', unsafe_allow_html=True)
        top5f = df2['primary_field'].value_counts().head(5).index
        df2f  = df2[df2['primary_field'].isin(top5f)]
        df2f = df2.copy()
        df2f['field_group'] = df2f['primary_field'].apply(reclassify_field)
        piv_f = (df2f.groupby(['period','field_group']).size()
                .unstack(fill_value=0).reindex(PERIOD_ORDER).fillna(0))
        piv_f = piv_f.replace(0, np.nan)
        fig_af = px.line(piv_f, markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
        fig_af.update_traces(connectgaps=True)
        fig_af.update_layout(xaxis_title='Período', yaxis_title='Artigos (escala log)', yaxis_type='log', 
                             legend_title='Área', margin=dict(t=10,b=10,l=10,r=10),
                             height=480, plot_bgcolor='white', paper_bgcolor='white')
        fig_af.update_xaxes(tickangle=45, tickfont=dict(size=12))
        st.plotly_chart(fig_af, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-title">Publicações por região por período</div>',
                    unsafe_allow_html=True)
        reg_rows = []
        for _, row in df2.iterrows():
            for r in (row['region'] or []):
                reg_rows.append({'period': row['period'], 'region': r})
        if reg_rows:
            df_reg = pd.DataFrame(reg_rows)
            piv_reg = (df_reg.groupby(['period','region']).size()
                       .unstack(fill_value=0).reindex(PERIOD_ORDER).fillna(0))
            fig_reg = px.bar(piv_reg.reset_index(), x='period',
                            y=piv_reg.columns.tolist(),
                            barmode='stack',
                            color_discrete_sequence=px.colors.qualitative.Set2)
            fig_reg.update_layout(xaxis_title='Período', yaxis_title='Artigos',
                                legend_title='Região',
                                margin=dict(t=10,b=10,l=10,r=10), height=480,
                                plot_bgcolor='white', paper_bgcolor='white')
            fig_reg.update_xaxes(tickangle=45, showgrid=False)
            fig_reg.update_yaxes(showgrid=True, gridcolor='#EEF1F4')
            st.plotly_chart(fig_reg, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown('<div class="section-title">Colaboração internacional e acesso aberto por período</div>',
                    unsafe_allow_html=True)

        df2['intl'] = df2['countries_distinct_count'].apply(
            lambda x: pd.to_numeric(x, errors='coerce') > 1)
        df2['is_oa_bool'] = df2['is_oa'].map(
            {True:1, False:0, 'True':1, 'False':0}).fillna(0)

        stats = df2.groupby('period').agg(
            n_intl=('intl','sum'),
            total=('intl','count'),
            n_oa=('is_oa_bool','sum'),
        ).reindex(PERIOD_ORDER).fillna(0).reset_index()
        stats['pct_intl'] = stats['n_intl'] / stats['total'].replace(0, np.nan) * 100
        stats['pct_oa']   = stats['n_oa']   / stats['total'].replace(0, np.nan) * 100

        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(
            x=stats['period'], y=stats['pct_intl'],
            mode='lines+markers', name='Coautoria internacional',
            line=dict(color=PRIMARY, width=2.5),
            marker=dict(size=7),
        ))
        fig_dual.add_trace(go.Scatter(
            x=stats['period'], y=stats['pct_oa'],
            mode='lines+markers', name='Acesso aberto',
            line=dict(color=ACCENT, width=2.5, dash='dash'),
            marker=dict(size=7),
        ))
        fig_dual.update_layout(
            xaxis_title='Período', yaxis_title='%',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(t=30,b=10,l=10,r=10), height=480,
            plot_bgcolor='white', paper_bgcolor='white',
        )
        fig_dual.update_xaxes(tickangle=45, showgrid=False)
        fig_dual.update_yaxes(showgrid=True, gridcolor='#EEF1F4')
        st.plotly_chart(fig_dual, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-title">Top tópicos por período</div>', unsafe_allow_html=True)
        topic_rows = []
        for _, row in df2.iterrows():
            for kw in (row['keyword_names'] or []):
                topic_rows.append({'period': row['period'], 'topico': kw})
        if topic_rows:
            df_topics = pd.DataFrame(topic_rows)
            top7 = df_topics['topico'].value_counts().head(7).index
            piv_t = (df_topics[df_topics['topico'].isin(top7)]
                     .groupby(['period','topico']).size()
                     .unstack(fill_value=0).reindex(PERIOD_ORDER).fillna(0))
            fig_top = px.line(piv_t, markers=True,
                              color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_top.update_layout(xaxis_title='Período', yaxis_title='Menções',
                                  legend_title='Tópico',
                                  margin=dict(t=10,b=10,l=10,r=10), height=480,
                                  plot_bgcolor='white', paper_bgcolor='white')
            fig_top.update_xaxes(tickangle=45,tickfont=dict(size=12))
            st.plotly_chart(fig_top, use_container_width=True)

    st.markdown('<div class="section-title">Termos nos títulos por período</div>',
                unsafe_allow_html=True)
    default_terms = ['migration','poverty','remittance','displacement',
                        'refugee','climate','inequality','health']
    terms_input = st.text_input('Termos (separados por vírgula)',
                                ', '.join(default_terms), key='t2_terms')
    terms = [t.strip().lower() for t in terms_input.split(',') if t.strip()]
    if terms:
        title_rows = []
        for _, row in df2.iterrows():
            title_lower = str(row.get('title','')).lower()
            for term in terms:
                if term in title_lower:
                    title_rows.append({'period': row['period'], 'term': term})
        if title_rows:
            df_title = pd.DataFrame(title_rows)
            total_per_period = df2.groupby('period').size().reindex(PERIOD_ORDER).fillna(1)
            piv_title = (df_title.groupby(['period','term']).size()
                            .unstack(fill_value=0).reindex(PERIOD_ORDER).fillna(0))
            for col in piv_title.columns:
                piv_title[col] = piv_title[col] / total_per_period * 100
            fig_title = px.line(piv_title, markers=True,
                                color_discrete_sequence=px.colors.qualitative.Set1)
            fig_title.update_layout(xaxis_title='Período', yaxis_title='% dos títulos',
                                    legend_title='Termo',
                                    margin=dict(t=10,b=10,l=10,r=10), height=480,
                                    plot_bgcolor='white', paper_bgcolor='white')
            fig_title.update_xaxes(tickangle=45, tickfont=dict(size=12))
            st.plotly_chart(fig_title, use_container_width=True)


# ═══════════════════════════════════════════
# ABA 3 — CONTEÚDO & TÓPICOS
# ═══════════════════════════════════════════
with tab3:
    with st.expander('🎛 Filtros', expanded=False):
        xc1, xc2, xc3, xc4 = st.columns(4)
        f3_periods  = xc1.multiselect('Período',                all_periods,    key='t3_period')
        f3_regions  = xc2.multiselect('Região',                 all_regions,    key='t3_region')
        f3_countries= xc3.multiselect('País',                   all_countries,  key='t3_country')
        f3_insts    = xc4.multiselect('Instituição',            all_insts,      key='t3_inst')
        xc5, xc6, xc7, _ = st.columns(4)
        f3_fields   = xc5.multiselect('Área do conhecimento',   all_fields,     key='t3_field')
        f3_journals = xc6.multiselect('Periódico',              all_journals,   key='t3_journal')
        f3_oa_sel   = xc7.selectbox( 'Acesso', ['Todos','Aberto','Fechado'],    key='t3_oa')

    f3_oa = True if f3_oa_sel == 'Aberto' else (False if f3_oa_sel == 'Fechado' else None)
    df3 = apply_filters(df_full, f3_periods, f3_regions, f3_countries,
                        f3_fields, f3_oa, f3_insts, f3_journals)
    st.caption(f'**{len(df3):,}** artigos após filtros')

    abstracts3 = tuple(df3['abstract'].fillna('').tolist())
    stop_tuple = tuple(sorted(STOP_DISPLAY))

    with st.spinner('Calculando termos...'):
        uni_cnt, bi_cnt = compute_nlp(abstracts3, banned_ngrams, stop_tuple)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown('<div class="section-title">Top 25 bigramas nos abstracts</div>', unsafe_allow_html=True)
        top_bi = pd.DataFrame(bi_cnt.most_common(25), columns=['bigrama','n'])
        fig_bi = px.bar(top_bi, x='n', y='bigrama', orientation='h',
                        color_discrete_sequence=[PRIMARY])
        fig_bi.update_layout(yaxis={'categoryorder':'total ascending'},
                             xaxis_title='Frequência', yaxis_title='',
                             margin=dict(t=10,b=10,l=10,r=10), height=480,
                             plot_bgcolor='white', paper_bgcolor='white')
        fig_bi.update_xaxes(showgrid=True, gridcolor='#EEF1F4')
        fig_bi.update_yaxes(tickfont=dict(size=12))
        st.plotly_chart(fig_bi, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-title">Nuvem de palavras: abstracts</div>', unsafe_allow_html=True)
        with st.spinner('Gerando nuvem...'):
            wc_buf = compute_wordcloud_img(abstracts3, banned_ngrams, stop_tuple)
        if wc_buf:
            st.image(wc_buf, use_container_width=True)

    # Emergentes e em declínio
    st.markdown('<div class="section-title">Temas emergentes vs. em declínio (1980-2004 → 2015-2024)</div>',
                unsafe_allow_html=True)
    early_texts = tuple(df3[df3['publication_year'] <= 2004]['abstract'].fillna('').tolist())
    late_texts  = tuple(df3[df3['publication_year'] >= 2015]['abstract'].fillna('').tolist())

    if len(early_texts) >= 5 and len(late_texts) >= 5:
        with st.spinner('Calculando emergentes...'):
            df_ed = compute_emerging_declining(early_texts, late_texts, banned_ngrams, stop_tuple)

        ec1, ec2 = st.columns(2)
        with ec1:
            top_em = df_ed.head(20)
            fig_em = px.bar(top_em, x='delta', y='termo', orientation='h',
                            color_discrete_sequence=['#2A9E4F'])
            fig_em.update_layout(yaxis={'categoryorder':'total ascending'},
                                 title='Termos emergentes',
                                 xaxis_title='Δ p.p. por 100 abstracts', yaxis_title='',
                                 margin=dict(t=30,b=10,l=10,r=10), height=480,
                                 plot_bgcolor='white', paper_bgcolor='white')
            fig_em.update_xaxes(showgrid=True, gridcolor='#EEF1F4')
            fig_em.update_yaxes(tickfont=dict(size=12))
            st.plotly_chart(fig_em, use_container_width=True)

        with ec2:
            top_dc = df_ed.tail(20).sort_values('delta')
            fig_dc = px.bar(top_dc, x='delta', y='termo', orientation='h',
                            color_discrete_sequence=['#C0392B'])
            fig_dc.update_layout(yaxis={'categoryorder':'total ascending'},
                                 title='Termos em declínio',
                                 xaxis_title='Δ p.p. por 100 abstracts', yaxis_title='',
                                 margin=dict(t=30,b=10,l=10,r=10), height=480,
                                 plot_bgcolor='white', paper_bgcolor='white')
            fig_dc.update_xaxes(showgrid=True, gridcolor='#EEF1F4')
            fig_dc.update_yaxes(tickfont=dict(size=12))
            st.plotly_chart(fig_dc, use_container_width=True)
    else:
        st.info('Dados insuficientes para calcular emergentes/declínio com os filtros atuais.')

    # Barra de busca por título/autor
    st.markdown('<div class="section-title">Buscar artigo</div>', unsafe_allow_html=True)
    query = st.text_input('Buscar por título ou autor', '', key='t3_search')
    if query:
        mask = (
            df3['title'].str.contains(query, case=False, na=False) |
            df3['first_author'].str.contains(query, case=False, na=False)
        )
        results = df3[mask][['first_author','publication_year','title','cited_by_count','doi','journal']]
        st.dataframe(results.head(30), use_container_width=True, hide_index=True)