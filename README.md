# MSL: Mapeamento Sistemático da Literatura com IA

Pipeline automatizado em Python para Mapeamento Sistemático da Literatura (MSL), com triagem assistida por LLM, análise bibliométrica, processamento de linguagem natural (NLP) e dashboard interativo.

> **Tema de aplicação:** Pobreza e Migração (1980-2024)  
> **Dashboard interativo:** [huggingface.co/spaces/dpierf/msl-povmig](https://huggingface.co/spaces/dpierf/msl-povmig)

---

## Motivação

Este projeto é a continuação computacional de um artigo publicado em 2019 na revista **Mercator**:

> MARIA, Pier Francesco De. **Mapeando relações em pesquisas sobre pobreza e migração (1980-2017)**. *Mercator*, Fortaleza, v. 18, fev. 2019. DOI: [10.4215/rm2019.e18003](https://doi.org/10.4215/rm2019.e18003)

O artigo original foi realizado manualmente com três bases de dados (Web of Science, Scopus e ScienceDirect), catalogação de referências com o Zotero® 5.0.33, e análises de dados em Excel e R. Este pipeline automatiza e expande esse processo: utiliza a API do OpenAlex (~250M artigos, incluindo SciELO), triagem por LLM com prompt refinado score rastreável, análise bibliométrica completa e NLP sobre títulos e abstracts, cobrindo de 1980 a 2024.

Este pipeline é **generalizável**: basta alterar o arquivo YAML de configuração para aplicá-lo a qualquer outro tema de MSL.

---

## Estrutura do projeto

```
MSL/
├── main.py                  # Orquestrador do pipeline completo
├── app.py                   # Dashboard Streamlit (para uso local)
├── requirements.txt         # Dependências Python
├── config/
│   └── poverty_migration.yaml   # Configuração do tema (personalizável)
└── src/
    ├── config_loader.py     # 1. Leitura do YAML e variáveis de ambiente
    ├── search.py            # 2. Busca de documentos no OpenAlex
    ├── dedup.py             # 3. Deduplicação absoluta (DOI) e fuzzy (título)
    ├── screen.py            # 4. Triagem com LLM (Mistral)
    ├── coding.py            # 5. Codificação temática por LLM (Groq)
    ├── flatten.py           # 6. Extração de campos aninhados + CrossRef
    ├── analyze.py           # 7. Análises bibliométricas e gráficos
    ├── keywords.py          # 8. NLP sobre títulos e abstracts
    ├── metrics.py           # 0. Registro de métricas para funil de pesquisa
    └── screen_batch.py      # Triagem em lote (requer plano pago Groq - não usado neste pipeline)
```

> [!NOTE]
> A pasta `data/` é gerada automaticamente pelo pipeline e não está versionada. As chaves de API devem ser geradas pelo usuário e inseridas no arquivo `.env` (não versionado).

> [!WARNING]
> O pacote `mistralai` foi alvo de um ataque de supply chain em maio de 2026 (campanha *Mini Shai-Hulud*). A versão comprometida foi a `2.4.6`. O `requirements.txt` deste projeto fixa a versão segura `mistralai==2.4.5`. Ao instalar, verifique se o pip não atualizou para uma versão posterior antes de confirmar a segurança do pacote no [PyPI](https://pypi.org/project/mistralai/).
---

## O que cada arquivo principal faz

### `main.py` — Pipeline completo

Orquestra todas as etapas em sequência. Ao rodar, o programa solicita o nome do arquivo de configuração:

```bash
python main.py
# → Nome do arquivo de configuração (sem .yaml): poverty_migration
```

**Etapas executadas:**
1. Busca no OpenAlex (`search.py`)
2. Deduplicação por DOI e similaridade de título (`dedup.py`)
3. Triagem por LLM com score dimensional (`screen.py`)
4. Codificação temática dos documentos (`coding.py`)
5. Extração de campos aninhados e consulta CrossRef (`flatten.py`)
6. Análises bibliométricas e geração de gráficos (`analyze.py`)
7. NLP sobre títulos e abstracts (`keywords.py`)

### `app.py` — Dashboard local

Dashboard Streamlit para exploração interativa do corpus. Para rodar localmente:

```bash
streamlit run app.py
```

A versão pública está disponível em:  
[huggingface.co/spaces/dpierf/msl-povmig](https://huggingface.co/spaces/dpierf/msl-povmig)

---

## Configuração — `config/poverty_migration.yaml`

Toda a especificidade temática do projeto vive neste arquivo. Para aplicar o pipeline a outro tema, basta criar um novo YAML com a mesma estrutura.

### Seções principais

**`project`**: Nome, descrição e termos do tema para o prompt de triagem.

**`search`**: Parâmetros da busca: período, idioma, número de páginas, e os grupos de termos (produto cartesiano entre `group_a` e `group_b` gera todas as queries).

**`screening`**: Provedor e modelo LLM, threshold de inclusão (0–100), tempo de sleep entre requisições, critérios de inclusão e exclusão, critérios adicionais de exclusão e dimensões de avaliação.

**`coding`**: Provedor, modelo e categorias temáticas para classificação dos artigos incluídos.

**`sources`**: Lista de repositórios e indexadores a serem excluídos do ranking de periódicos, e autores institucionais a serem excluídos do ranking de autores.

**`keywords`**: Listas de termos a ignorar na análise NLP:
  * `stop_display_unigrams` (genéricos removidos dos unigramas mas preservados em bigramas)
  * `stop_artifact` (artefatos de metadados como nomes de meses e editoras)
  * `banned_ngrams` (n-gramas explicitamente banidos, como estruturas retóricas e artefatos bibliográficos)

---

## Instalação

```bash
git clone https://github.com/dpierf/MSL.git
cd MSL
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz com suas chaves de API:

```
GROQ_API_KEY=sua_chave_groq
MISTRAL_API_KEY=sua_chave_mistral
```

As chaves são gratuitas nos planos básicos de [Groq](https://console.groq.com) e [MistralAI](https://console.mistral.ai).

---

## Próximos passos

- **Codificação temática:** rodar o `coding.py` para classificar os artigos selecionados nas categorias temáticas definidas no YAML (ver a seção `coding`).
- **Validação amostral:** revisão manual de amostra aleatória (~100 artigos) para estimar a concordância entre a triagem LLM e um avaliador humano.
- **Expansão de idiomas:** nova rodada de busca no OpenAlex com termos em PT e ES, aproveitando a cobertura do SciELO, lacuna identificada no artigo original de 2019.
- **Análise de redes:** cocitação e coautoria a partir dos campos `referenced_works` e `authorships` já coletados.

---

## Uso de inteligência artificial generativa

O desenvolvimento deste pipeline (incluindo a arquitetura do código, os módulos de busca, triagem, NLP e dashboard) foi realizado com suporte do modelo **Claude Sonnet 4.6** (Anthropic, 2026). 

> ANTHROPIC. **Claude Sonnet 4.6** [modelo de linguagem de grande escala]. San Francisco: Anthropic, 2026. Disponível em: https://claude.ai. Acesso em: mai. 2026.

As decisões metodológicas, os critérios de inclusão/exclusão, a seleção dos modelos de triagem e a responsabilidade intelectual sobre os resultados são integral de responsabilidade do autor.

---

## Citação

Se utilizar este pipeline em sua pesquisa, cite o artigo original que o motivou:

```
MARIA, Pier Francesco De. Mapeando relações em pesquisas sobre pobreza e 
migração (1980–2017). Mercator, Fortaleza, v. 18, fev. 2019. 
DOI: 10.4215/rm2019.e18003
```

---

## Licença

MIT
