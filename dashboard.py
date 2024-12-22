import streamlit as st
import yaml
import json
from PIL import Image
import pandas as pd
import altair as alt
import faiss
from transformers import BertTokenizer, BertModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# Função para carregar o texto sumarizado de config.yaml
def load_summary_from_yaml(file_path="data/config.yaml"):
    try:
        with open(file_path, 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)
            return config.get("overview_summary", "Texto sumarizado não encontrado.")
    except FileNotFoundError:
        return "Arquivo config.yaml não encontrado."
    except yaml.YAMLError:
        return "Erro ao carregar o arquivo YAML."

# Função para carregar os insights de um arquivo JSON
def load_insights_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            insights = data.get("insights", "")
            
            # Verificar se insights é uma string
            if isinstance(insights, str):
                # Quebra o texto em parágrafos utilizando '\n\n' como delimitador
                paragraphs = insights.split('\n\n')
                return paragraphs  # Retorna uma lista de parágrafos
            else:
                return ["Formato inesperado no campo 'insights'."]
    except UnicodeDecodeError:
        st.error("Erro ao decodificar o arquivo. Verifique a codificação.")
        return []
    except FileNotFoundError:
        st.error("Arquivo não encontrado.")
        return []


# Função para exibir a aba Overview
def display_overview_tab():
    # Título principal
    st.title("Painel de Análise Parlamentar")
    
    # Descrição da solução
    st.write(
        """
        Bem-vindo ao painel! Este painel foi desenvolvido para oferecer uma visão detalhada sobre a distribuição 
        de deputados e os principais insights relacionados. Explore os gráficos, análises e informações sumarizadas 
        para compreender melhor os dados.
        """
    )
    
    # Carregar texto sumarizado do config.yaml
    summary = load_summary_from_yaml()
    st.subheader("Texto Sumarizado")
    st.write(summary)
    
    # Carregar e exibir o gráfico de barras
    st.subheader("Distribuição de Deputados")
    distrib_image = Image.open('docs/distribuicao_deputados.png')
    st.image(distrib_image, use_column_width=True, caption="Distribuição de Deputados")
    
    # Carregar e exibir os insights do JSON
    st.subheader("Principais Insights")
    insights = load_insights_from_json('data/insights_distribuicao_deputados.json')
    if insights:
        # Concatenar todos os parágrafos em um único texto contínuo
        continuous_insights = " ".join(insights)
        st.write(continuous_insights)
    else:
        st.write("Nenhum insight disponível no momento.")


# Função para carregar insights de um arquivo JSON
def load_insights_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return data.get("insights", [])

# Função para carregar a série temporal de despesas de um arquivo Parquet
def load_expense_series(file_path):
    return pd.read_parquet(file_path)

def filter_expenses_by_deputy(expenses_df, deputy_name):
    return expenses_df[expenses_df['nomeDeputado'] == deputy_name]

def plot_expenses_series(filtered_data):
    if filtered_data.empty:
        st.write("Nenhum dado disponível para exibir o gráfico.")
        return None

    # Criação do gráfico
    chart = (
        alt.Chart(filtered_data)
        .mark_bar()
        .encode(
            x=alt.X('dataDocumento:T', title='Data'),
            y=alt.Y('valorLiquido:Q', title='Despesa (R$)'),
            tooltip=['dataDocumento', 'valorLiquido']
        )
        .properties(width='container', height=400, title="Série Temporal de Despesas")
    )
    return chart

def display_expenses_tab():
    # Título da aba
    st.title("Análise de Despesas Parlamentares")
    
    # Carregar e exibir os insights
    st.subheader("Principais Insights")
    insights = load_insights_from_json('data/insights_despesas_deputados.json')
    if insights:
        continuous_insights = " ".join(insights)
        st.write(continuous_insights)
    else:
        st.write("Nenhum insight disponível no momento.")
    
    # Carregar a série temporal de despesas
    expenses_df = load_expense_series('data/serie_despesas_diarias_deputados.parquet')

    # Verificar se as colunas esperadas estão disponíveis
    if 'nomeDeputado' not in expenses_df.columns or 'dataDocumento' not in expenses_df.columns or 'valorLiquido' not in expenses_df.columns:
        st.error("As colunas necessárias ('nomeDeputado', 'dataDocumento', 'valorLiquido') não estão disponíveis no arquivo de despesas.")
        st.stop()  # Para a execução desta aba

    # Garantir que a coluna 'dataDocumento' esteja no formato datetime
    expenses_df['dataDocumento'] = pd.to_datetime(expenses_df['dataDocumento'], errors='coerce')
    
    # Adicionar um seletor para o usuário escolher um deputado
    st.subheader("Seleção de Deputado")
    deputies = sorted(expenses_df['nomeDeputado'].unique())
    selected_deputy = st.selectbox("Escolha um deputado:", deputies)
    
    # Filtrar os dados pelo deputado selecionado
    filtered_data = filter_expenses_by_deputy(expenses_df, selected_deputy)
    
    # Exibir o gráfico da série temporal de despesas
    st.subheader(f"Série Temporal de Despesas - {selected_deputy}")
    if not filtered_data.empty:
        expense_chart = plot_expenses_series(filtered_data)
        if expense_chart:
            st.altair_chart(expense_chart, use_container_width=True)
    else:
        st.write("Nenhuma despesa encontrada para o deputado selecionado.")


def carregar_dados_proposicoes():
    """Carrega os dados das proposições do arquivo parquet."""
    try:
        df_proposicoes = pd.read_parquet("data/proposicoes_deputados.parquet")
        # Converter os IDs para string
        df_proposicoes["id"] = df_proposicoes["id"].astype(str)
        return df_proposicoes
    except FileNotFoundError:
        st.error("Arquivo data/proposicoes_deputados.parquet não encontrado.")
        return None

def carregar_resumos_proposicoes():
    """Carrega os resumos das proposições do arquivo JSON."""
    try:
        with open("data/sumarizacao_proposicoes.json", "r", encoding="utf-8") as f:
            resumos = json.load(f)  # Carrega o arquivo JSON como uma lista
            # Transformar em dicionário com chave como string
            if isinstance(resumos, list):
                return {str(item["proposicao_id"]): item["resumo"] for item in resumos if "proposicao_id" in item and "resumo" in item}
            else:
                st.error("O formato do arquivo de resumos não é uma lista.")
                return {}
    except FileNotFoundError:
        st.error("Arquivo data/sumarizacao_proposicoes.json não encontrado.")
        return {}
    except json.JSONDecodeError:
        st.error("Erro ao decodificar o arquivo JSON de resumos.")
        return {}


def exibir_tabela_proposicoes(df_proposicoes):
    """Exibe as proposições em uma tabela interativa."""
    st.write("## Tabela de Proposições")
    st.dataframe(df_proposicoes, use_container_width=True)


def exibir_resumos_proposicoes(resumos, df_proposicoes):
    """Exibe os resumos das proposições."""
    st.write("## Resumos das Proposições")

    # Adicionar seletor para escolher a proposição
    selected_proposicao = st.selectbox("Selecione uma proposição:", df_proposicoes["id"].unique())

    # Buscar o resumo no dicionário
    resumo = resumos.get(selected_proposicao, "Resumo não encontrado.")

    # Exibir o resumo
    st.write(resumo)


def aba_proposicoes():
    """Cria a aba 'Proposições' no dashboard."""
    st.title("Proposições")

    df_proposicoes = carregar_dados_proposicoes()
    resumos = carregar_resumos_proposicoes()

    if df_proposicoes is not None:
        exibir_tabela_proposicoes(df_proposicoes)

    if df_proposicoes is not None and resumos is not None:  # Garante que ambos os dados estejam carregados
        exibir_resumos_proposicoes(resumos, df_proposicoes)

# Função para carregar dados JSON ou Parquet
def load_data(file_path, file_type="json"):
    if file_type == "json":
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    elif file_type == "parquet":
        return pd.read_parquet(file_path)

# Função para vetorizar dados usando BERT
def vectorize_data(data, tokenizer, model, column=None):
    if isinstance(data, pd.DataFrame) and column:
        texts = data[column].tolist()
    elif isinstance(data, list):
        texts = data
    else:
        texts = [data]

    tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Criar base vetorial FAISS
def create_faiss_index(data, tokenizer, model):
    embeddings = vectorize_data(data, tokenizer, model)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, data

# Função para realizar busca na base vetorial
def search_in_index(query, index, tokenizer, model, data, k=3):
    query_vector = vectorize_data(query, tokenizer, model)[0]
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    results = [data[i] for i in indices[0]]
    return results

# Adicionar assistente virtual à aba Proposições
def display_propositions_tab_with_assistant():
    st.title("Proposições com Assistente Virtual")

    # Carregar dados
    propositions = pd.read_parquet('data/proposicoes_deputados.parquet')
    summaries = carregar_resumos_proposicoes()
    expenses = pd.read_parquet('data/serie_despesas_diarias_deputados.parquet')
    deputies = pd.read_parquet('data/deputados.parquet')

    # Configuração do modelo BERT
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

    # Vetorizar dados das proposições
    faiss_index, indexed_data = create_faiss_index(propositions["ementa"].tolist(), tokenizer, model)

    # Exibir a tabela de proposições
    st.subheader("Tabela de Proposições")
    st.dataframe(propositions, use_container_width=True)

    # Exibir resumos das proposições
    st.subheader("Resumo das Proposições")
    selected_proposition_id = st.selectbox(
        "Selecione uma proposição para visualizar o resumo:",
        propositions["id"].unique()
    )
    summary = summaries.get(str(selected_proposition_id), "Resumo não encontrado.")
    st.write(f"**Resumo da Proposição {selected_proposition_id}:** {summary}")

    # Assistente Virtual
    st.subheader("Assistente Virtual")
    user_question = st.text_input("Digite sua pergunta:")
    if user_question:
        # Responder à pergunta com base nos dados apropriados
        answer = answer_question(user_question, faiss_index, tokenizer, model, propositions, expenses, deputies)
        st.write("Resposta:")
        st.write(answer)



def answer_question(question, faiss_index, tokenizer, model, propositions, expenses, deputies):
    """
    Responde a perguntas com base nos dados apropriados.
    """
    question_lower = question.lower()

    if "partido" in question_lower and "mais deputados" in question_lower:
        # Encontrar o partido com mais deputados
        party_counts = deputies["siglaPartido"].value_counts()
        top_party = party_counts.idxmax()
        top_party_count = party_counts.max()
        return f"O partido com mais deputados na câmara é o '{top_party}', com {top_party_count} deputados."

    elif "despesa" in question_lower:
        if "mais declarada" in question_lower:
            # Encontrar o tipo de despesa mais comum
            expense_counts = expenses.groupby("tipoDespesa")["valorLiquido"].sum()
            most_common_expense = expense_counts.idxmax()
            most_common_value = expense_counts.max()
            return f"O tipo de despesa mais declarada pelos deputados é '{most_common_expense}' com um total de R$ {most_common_value:,.2f}."

        elif "deputado com mais despesas" in question_lower:
            # Encontrar o deputado com maior despesa
            deputy_expenses = expenses.groupby("nomeDeputado")["valorLiquido"].sum()
            top_deputy = deputy_expenses.idxmax()
            top_deputy_expenses = deputy_expenses.max()
            return f"O deputado com mais despesas é {top_deputy}, com um total de R$ {top_deputy_expenses:,.2f}."

    elif "proposição" in question_lower or "tema" in question_lower:
        if "economia" in question_lower:
            relevant_props = propositions[propositions["tema"] == "Economia"]
            summaries = relevant_props["ementa"].tolist()
            return "As proposições mais relevantes sobre Economia são:\n" + "\n".join(summaries)

        elif "ciência, tecnologia e inovação" in question_lower:
            relevant_props = propositions[propositions["tema"] == "Ciência, Tecnologia e Inovação"]
            summaries = relevant_props["ementa"].tolist()
            return "As proposições mais relevantes sobre Ciência, Tecnologia e Inovação são:\n" + "\n".join(summaries)

    else:
        # Busca na base vetorial como fallback
        query_vector = vectorize_data(question, tokenizer, model)[0]
        distances, indices = faiss_index.search(query_vector.reshape(1, -1), k=3)
        results = [propositions.iloc[i]["ementa"] for i in indices[0]]
        return "Aqui estão as respostas mais relevantes:\n" + "\n".join(results)





# Integração das abas
st.sidebar.title("Navegação")
tab = st.sidebar.radio("Escolha uma aba", ["Overview", "Despesas", "Proposições"])

if tab == "Overview":
    display_overview_tab()
elif tab == "Despesas":
    display_expenses_tab()
elif tab == "Proposições":
    display_propositions_tab_with_assistant()

