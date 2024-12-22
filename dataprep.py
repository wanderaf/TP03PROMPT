import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv
import google.generativeai as genai
import xml.etree.ElementTree as ET
import subprocess

load_dotenv()

# Configuração da URL base da API
API_BASE_URL = 'https://dadosabertos.camara.leg.br/api/v2'

def fetch_current_deputies():
    """
    Coleta informações sobre os deputados atuais da Câmara.
    """
    url = f"{API_BASE_URL}/deputados"
    response = requests.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    data = response.json()["dados"]
    return pd.DataFrame(data)

def save_deputies_data():
    """
    Salva as informações dos deputados atuais em formato parquet usando pyarrow.
    """
    df_deputados = fetch_current_deputies()
    os.makedirs("data", exist_ok=True)
    df_deputados.to_parquet("data/deputados.parquet", index=False, engine='pyarrow')
    print("Dados dos deputados salvos em 'data/deputados.parquet'.")


def plot_party_distribution():
    """
    Cria um gráfico de pizza mostrando a distribuição de deputados por partido.
    """
    # Carregar dados
    df_deputados = pd.read_parquet("data/deputados.parquet")
    
    # Agrupar por partido
    party_counts = df_deputados["siglaPartido"].value_counts()
    
    # Criar gráfico de pizza
    plt.figure(figsize=(10, 8))
    plt.pie(
        party_counts, 
        labels=party_counts.index, 
        autopct='%1.1f%%', 
        startangle=140
    )
    plt.title("Distribuição de Deputados por Partido")
    plt.savefig("docs/distribuicao_deputados.png")
    print("Gráfico salvo em 'docs/distribuicao_deputados.png'.")

def generate_insights_with_gemini():
    """
    Gera insights sobre a distribuição de deputados por partido utilizando Gemini.
    """
    # Carregar dados
    df_deputados = pd.read_parquet("data/deputados.parquet")
    party_counts = df_deputados["siglaPartido"].value_counts()
    
    # Dados do maior partido
    major_party = party_counts.idxmax()
    total_major_party = party_counts.max()
    total_deputies = party_counts.sum()
    percentual_major_party = (total_major_party / total_deputies) * 100
    
    # Criar o prompt
    prompt = f"""
    Dados: Existem {total_deputies} deputados distribuídos em {len(party_counts)} partidos na Câmara dos Deputados. 
    O partido com maior número de representantes é {major_party}, com {total_major_party} deputados, correspondendo a {percentual_major_party:.2f}% do total.
    
    Persona: Você é um analista político especializado em legislações e funcionamento do Congresso Nacional.
    
    Tarefa: Crie insights sobre como a distribuição desigual de deputados por partidos pode influenciar a formação de coalizões, aprovação de projetos de lei, e dinâmica das comissões. Considere o partido majoritário e as diferenças entre partidos menores.
    
    Exemplo: "A concentração de deputados no partido majoritário pode facilitar a formação de coalizões em projetos do governo, mas pode gerar dificuldades para a oposição nas comissões legislativas, que dependem de representação proporcional."
    """
    
    # Configurar e usar o modelo Gemini
    try:
        # Configuração da API
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Gerar conteúdo
        response = model.generate_content(prompt)
        
        # Verificar o texto gerado
        if hasattr(response, "text") and response.text:
            insights = response.text.strip()
        else:
            insights = "Erro: O modelo não retornou um texto válido."
        
    except AttributeError as e:
        insights = f"Erro na configuração do modelo Gemini: {e}"
    except Exception as e:
        insights = f"Erro ao gerar insights: {e}"
    
    # Salvar insights em JSON
    os.makedirs("data", exist_ok=True)
    with open("data/insights_distribuicao_deputados.json", "w") as f:
        json.dump({"insights": insights}, f, indent=4, ensure_ascii=False)
    print("Insights salvos em 'data/insights_distribuicao_deputados.json'.")

def fetch_deputy_expenses(deputy_id, deputy_name, start_date=None, end_date=None):
    """
    Coleta informações de despesas de um deputado e retorna como DataFrame.
    """
    url = f"{API_BASE_URL}/deputados/{deputy_id}/despesas"
    print(f"Solicitando despesas para deputado {deputy_name} (ID: {deputy_id}) com URL: {url}")
    
    try:
        response = requests.get(url, headers={"Accept": "application/xml"})
        response.raise_for_status()
        
        # Parse do XML
        root = ET.fromstring(response.content)
        registros = root.findall(".//registroCotas")
        
        # Extrair dados relevantes
        data = []
        for registro in registros:
            valor_liquido = registro.findtext("valorLiquido", "0").replace(",", ".")
            try:
                valor_liquido = float(valor_liquido)  # Converter para float
            except ValueError:
                valor_liquido = 0.0

            data.append({
                "dataDocumento": registro.findtext("dataDocumento"),
                "tipoDespesa": registro.findtext("tipoDespesa"),
                "ano": registro.findtext("ano"),
                "mes": registro.findtext("mes"),
                "valorLiquido": valor_liquido,  # Valor corrigido
                "nomeDeputado": deputy_name,  # Adicionar nome do deputado
            })
        
        # Converter para DataFrame
        return pd.DataFrame(data)
    
    except requests.exceptions.HTTPError as http_err:
        print(f"Erro HTTP ao coletar despesas do deputado {deputy_name} (ID: {deputy_id}): {http_err}")
    except Exception as e:
        print(f"Erro desconhecido ao coletar despesas do deputado {deputy_name} (ID: {deputy_id}): {e}")
    
    return pd.DataFrame()  # Retorna DataFrame vazio em caso de erro


def process_expenses(start_date="2024-08-01", end_date="2024-08-30"):
    """
    Coleta e processa despesas de todos os deputados no período especificado.
    Agrupa os dados por dia, deputado e tipo de despesa e salva num arquivo parquet.
    """
    # Carregar lista de deputados
    df_deputados = pd.read_parquet("data/deputados.parquet")
    all_expenses = []
    print(f"Coletando despesas para {len(df_deputados)} deputados...")

    for deputy_id in df_deputados["id"]:
        try:
            print(f"Coletando despesas para deputado ID: {deputy_id}")
            expenses = fetch_deputy_expenses(deputy_id)
            if not expenses.empty:
                print(f"Despesas coletadas: {len(expenses)} registros para deputado ID: {deputy_id}")
                all_expenses.append(expenses)
            else:
                print(f"Nenhuma despesa encontrada para deputado ID: {deputy_id}")
        except Exception as e:
            print(f"Erro ao coletar despesas para o deputado {deputy_id}: {e}")

    if all_expenses:
        # Concatenar todos os dados de despesas
        df_expenses = pd.concat(all_expenses, ignore_index=True)
        
        # Agrupar dados por dia, deputado e tipo de despesa
        grouped_expenses = (
            df_expenses.groupby(["dataDocumento", "tipoDespesa"])
            .agg({"valorLiquido": "sum"})
            .reset_index()
        )
        
        # Salvar no formato parquet
        os.makedirs("data", exist_ok=True)
        grouped_expenses.to_parquet("data/serie_despesas_diarias_deputados.parquet", index=False)
        print("Dados de despesas salvos em 'data/serie_despesas_diarias_deputados.parquet'.")
    else:
        print("Nenhuma despesa encontrada para o período especificado.")

def generate_python_code():
    """
    Gera código Python para análise dos dados de despesas.
    """
    prompt = """
    Dados disponíveis: As despesas dos deputados estão disponíveis no arquivo 
    'data/serie_despesas_diarias_deputados.parquet'. Cada registro contém as seguintes colunas: 
    - dataDocumento (data da despesa),
    - nomeDeputado (nome do deputado),
    - tipoDespesa (tipo da despesa),
    - valorLiquido (valor líquido da despesa).

    Tarefa: Crie um código Python para realizar as seguintes análises:
    1. Identificar o deputado com o maior total de despesas no período.
    2. Analisar os tipos de despesas mais comuns e seus valores totais.
    3. Gerar uma série temporal das despesas diárias totais.

    O código deve ser organizado, comentado e usar pandas e matplotlib para visualizações.
    """
    
    # Configurar o LLM (substituir pela configuração específica do seu LLM)
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    response = genai.generate_content(
        messages=[{"role": "system", "content": "Você é um especialista em análise de dados."},
                  {"role": "user", "content": prompt}],
        model="gemini-1.5-flash",
        max_output_tokens=300
    )
    code = response.text.strip()
    print("Código gerado:")
    print(code)
    return code

def generate_insights_from_expenses():
    """
    Gera insights com base nas análises das despesas dos deputados.
    """
    prompt = """
    Baseado nos seguintes resultados das análises das despesas dos deputados:
    1. O deputado com o maior total de despesas é [nome] com um total de R$ [valor].
    2. Os tipos de despesas mais comuns incluem [tipo1] com um total de R$ [valor1], [tipo2] com R$ [valor2], e [tipo3] com R$ [valor3].
    3. A série temporal das despesas diárias mostra picos em [datas] associados a [eventos].

    Persona: Você é um analista financeiro com foco em transparência e controle de gastos públicos.

    Tarefa: Gere insights detalhados sobre o impacto desses padrões de despesas nos gastos públicos, destacando implicações, tendências e possíveis medidas de controle. Considere os padrões de gastos excessivos e a importância de cada tipo de despesa.

    O texto deve ser claro, objetivo e apresentar recomendações baseadas nos dados.
    """

    # Configurar e usar o LLM para gerar insights
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    response = genai.generate_content(
        messages=[{"role": "system", "content": "Você é um analista financeiro."},
                  {"role": "user", "content": prompt}],
        model="gemini-1.5-flash",
        max_output_tokens=300
    )
    insights = response.text.strip()

    # Salvar insights em JSON
    os.makedirs("data", exist_ok=True)
    with open("data/insights_despesas_deputados.json", "w") as f:
        json.dump({"insights": insights}, f, indent=4, ensure_ascii=False)
    print("Insights salvos em 'data/insights_despesas_deputados.json'.")


def process_and_save_expenses(start_date="2024-08-01", end_date="2024-08-30"):
    """
    Coleta e processa despesas dos deputados no período especificado.
    Agrupa os dados por dia, deputado e tipo de despesa e salva num arquivo parquet.
    """
    # Carregar lista de deputados
    df_deputados = pd.read_parquet("data/deputados.parquet")
    all_expenses = []
    print(f"Coletando despesas para {len(df_deputados)} deputados...")

    for _, row in df_deputados.iterrows():
        deputy_id = row["id"]  # ID do deputado
        deputy_name = row["nome"]  # Nome do deputado
        try:
            print(f"Coletando despesas para deputado {deputy_name} (ID: {deputy_id})")
            expenses = fetch_deputy_expenses(deputy_id, deputy_name)
            if not expenses.empty:
                print(f"Despesas coletadas: {len(expenses)} registros para deputado {deputy_name} (ID: {deputy_id})")
                all_expenses.append(expenses)
            else:
                print(f"Nenhuma despesa encontrada para deputado {deputy_name} (ID: {deputy_id})")
        except Exception as e:
            print(f"Erro ao coletar despesas para o deputado {deputy_name} (ID: {deputy_id}): {e}")

    if all_expenses:
        # Concatenar todos os dados de despesas
        df_expenses = pd.concat(all_expenses, ignore_index=True)
        
        # Garantir que valorLiquido seja numérico
        df_expenses["valorLiquido"] = pd.to_numeric(df_expenses["valorLiquido"], errors="coerce").fillna(0)

        # Agrupar dados por dia, deputado e tipo de despesa
        grouped_expenses = (
            df_expenses.groupby(["dataDocumento", "nomeDeputado", "tipoDespesa", "ano", "mes"])
            .agg({"valorLiquido": "sum"})
            .reset_index()
        )
        
        # Salvar no formato parquet
        os.makedirs("data", exist_ok=True)
        grouped_expenses.to_parquet("data/serie_despesas_diarias_deputados.parquet", index=False, engine="pyarrow")
        print("Dados de despesas salvos em 'data/serie_despesas_diarias_deputados.parquet'.")
    else:
        print("Nenhuma despesa encontrada para o período especificado.")


def generate_analysis_code():
    """
    Gera um código Python para analisar os dados das despesas dos deputados e salva em um arquivo.
    """
    
    # Criar o prompt
    prompt = f"""
    Dados disponíveis: As despesas dos deputados estão disponíveis no arquivo 
    'data/serie_despesas_diarias_deputados.parquet'. Cada registro contém as seguintes colunas:
    - dataDocumento (data da despesa),
    - nomeDeputado (nome do deputado),
    - tipoDespesa (tipo da despesa),
    - ano (ano da despesa),
    - mes (mês da despesa),
    - valorLiquido (valor líquido da despesa).

    Tarefa: Crie um código Python para realizar as seguintes análises:
    1. Identificar o deputado com o maior total de despesas no período.
    2. Determinar os tipos de despesas mais comuns e seus valores totais.
    3. Gerar uma série temporal das despesas diárias totais, agrupadas por mês.

    O código deve ser organizado, comentado e utilizar pandas para análise e matplotlib para visualização.
    """
    
    # Configurar e usar o modelo Gemini
    try:
        # Configuração da API
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Gerar conteúdo
        response = model.generate_content(prompt)
        
        # Verificar o texto gerado
        if hasattr(response, "text") and response.text:
            code = response.text.strip()
        else:
            code = "Erro: O modelo não retornou um código válido."
        
    except AttributeError as e:
        code = f"Erro na configuração do modelo Gemini: {e}"
    except Exception as e:
        code = f"Erro ao gerar o código: {e}"
    
    # Salvar o código gerado em um arquivo
    os.makedirs("data", exist_ok=True)
    with open("data/analise_despesas.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("Código gerado salvo em 'data/analise_despesas.py'.")

    return code

def generate_insights_from_analysis():
    """
    Executa o arquivo analise_despesas.py, captura os resultados e gera insights diretamente com base na saída.
    """
    try:
        # Executar o script analise_despesas.py e capturar a saída
        process = subprocess.run(
            [r"C:\Users\wande\OneDrive\INFNET\7. 6º Semestre\Engenharia de Prompts para Ciência de Dados\AT\Scripts\python.exe",
             "data/analise_despesas.py"],
            capture_output=True,
            text=True,
            check=True
        )

        output = process.stdout.strip()
        
        # Verificar se a saída contém dados
        if not output:
            raise ValueError("A saída do script analise_despesas.py está vazia.")

        # Criar o prompt diretamente com a saída bruta
        prompt = f"""
        Baseado na seguinte saída das análises das despesas dos deputados:
        {output}

        Persona: Você é um analista financeiro especializado em controle de gastos públicos.

        Tarefa: Gere insights detalhados sobre o impacto desses padrões de despesas nos gastos públicos. 
        Destaque implicações, tendências e possíveis medidas de controle. 
        Considere os padrões de gastos excessivos e a importância de cada tipo de despesa.

        O texto deve ser claro, objetivo e apresentar recomendações baseadas nos dados.
        """

        # Configurar e usar o modelo Gemini para gerar insights
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            # Verificar o texto gerado
            if hasattr(response, "text") and response.text:
                insights = response.text.strip()
            else:
                insights = "Erro: O modelo não retornou um texto válido."
        
        except AttributeError as e:
            raise ValueError(f"Erro na configuração do modelo Gemini: {e}")
        except Exception as e:
            raise ValueError(f"Erro ao gerar insights: {e}")

        # Salvar os insights em um arquivo JSON
        os.makedirs("data", exist_ok=True)
        with open("data/insights_despesas_deputados.json", "w", encoding="utf-8") as f:
            json.dump({"insights": insights}, f, indent=4, ensure_ascii=False)
        print("Insights salvos em 'data/insights_despesas_deputados.json'.")
    
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar analise_despesas.py: {e.stderr}")
    except ValueError as ve:
        print(f"Erro no processamento da saída: {ve}")
    except Exception as e:
        print(f"Erro ao gerar insights: {e}")


import os
import requests
import pandas as pd
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configuração da URL base da API
API_BASE_URL = 'https://dadosabertos.camara.leg.br/api/v2'

def fetch_propositions_by_theme(theme_code, start_date, end_date, limit=10):
    """
    Coleta informações das proposições por tema dentro de um período específico.
    Args:
        theme_code (int): Código do tema.
        start_date (str): Data de início no formato YYYY-MM-DD.
        end_date (str): Data de fim no formato YYYY-MM-DD.
        limit (int): Número de proposições a coletar por tema.
    Returns:
        pd.DataFrame: DataFrame com informações das proposições.
    """
    url = f"{API_BASE_URL}/proposicoes"
    params = {
        "dataInicio": start_date,
        "dataFim": end_date,
        "codTema": theme_code,
        "itens": limit,
        "ordem": "desc",
        "ordenarPor": "id"
    }
    response = requests.get(url, headers={"Accept": "application/json"}, params=params)
    response.raise_for_status()
    data = response.json()["dados"]
    return pd.DataFrame(data)

def save_propositions():
    """
    Coleta e salva informações das proposições por tema.
    """
    themes = {"Economia": 40, "Educação": 46, "Ciência, Tecnologia e Inovação": 62}
    start_date = "2024-08-01"
    end_date = "2024-08-30"
    all_propositions = []

    for theme, code in themes.items():
        print(f"Coletando proposições para o tema: {theme}")
        try:
            df = fetch_propositions_by_theme(code, start_date, end_date, limit=10)
            df["tema"] = theme  # Adicionar coluna do tema
            all_propositions.append(df)
        except Exception as e:
            print(f"Erro ao coletar proposições para o tema {theme}: {e}")

    if all_propositions:
        df_all = pd.concat(all_propositions, ignore_index=True)
        os.makedirs("data", exist_ok=True)
        df_all.to_parquet("data/proposicoes_deputados.parquet", index=False)
        print("Proposições salvas em 'data/proposicoes_deputados.parquet'.")
    else:
        print("Nenhuma proposição coletada.")

def summarize_propositions():
    """
    Realiza a sumarização das proposições utilizando sumarização por chunks e salva o resultado.
    """
    try:
        # Carregar as proposições
        df = pd.read_parquet("data/proposicoes_deputados.parquet")
        summaries = []
        for index, row in df.iterrows():
            prompt = f"""
            Proposição: {row['ementa']}
            Tema: {row['tema']}
            
            Persona: Você é um analista legislativo especializado em resumos de proposições.

            Tarefa: Resuma a proposição acima em uma frase clara e objetiva.
            """
            # Configurar e usar o modelo Gemini
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                if hasattr(response, "text") and response.text:
                    summaries.append({
                        "proposicao_id": row["id"],
                        "resumo": response.text.strip()
                    })
                else:
                    summaries.append({
                        "proposicao_id": row["id"],
                        "resumo": "Erro: O modelo não retornou um resumo válido."
                    })
            except Exception as e:
                summaries.append({
                    "proposicao_id": row["id"],
                    "resumo": f"Erro ao processar: {e}"
                })

        # Salvar os resumos em JSON
        os.makedirs("data", exist_ok=True)
        with open("data/sumarizacao_proposicoes.json", "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=4, ensure_ascii=False)
        print("Sumarização salva em 'data/sumarizacao_proposicoes.json'.")

    except Exception as e:
        print(f"Erro ao realizar a sumarização: {e}")

if __name__ == "__main__":
    # Coleta e processamento das proposições
    save_propositions()
    summarize_propositions()
