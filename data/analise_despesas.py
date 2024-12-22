import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
try:
    df = pd.read_parquet('data/serie_despesas_diarias_deputados.parquet')
except FileNotFoundError:
    print("Erro: O arquivo 'data/serie_despesas_diarias_deputados.parquet' não foi encontrado.")
    exit()


# 1. Deputado com maior total de despesas

df['dataDocumento'] = pd.to_datetime(df['dataDocumento']) #Garante que a coluna é datetime para evitar erros
total_despesas_por_deputado = df.groupby('nomeDeputado')['valorLiquido'].sum()
deputado_maior_despesa = total_despesas_por_deputado.idxmax()
valor_maior_despesa = total_despesas_por_deputado.max()

print(f"O deputado com maior total de despesas é: {deputado_maior_despesa} com R$ {valor_maior_despesa:.2f}")


# 2. Tipos de despesas mais comuns e seus valores totais

tipos_despesas_comuns = df.groupby('tipoDespesa')['valorLiquido'].agg(['sum', 'count'])
tipos_despesas_comuns = tipos_despesas_comuns.sort_values(by='sum', ascending=False)

print("\nTipos de despesas mais comuns e seus valores totais:")
print(tipos_despesas_comuns)

# Visualização dos tipos de despesas
plt.figure(figsize=(12, 6))
sns.barplot(x=tipos_despesas_comuns['sum'], y=tipos_despesas_comuns.index)
plt.xlabel('Valor Total')
plt.ylabel('Tipo de Despesa')
plt.title('Valor Total por Tipo de Despesa')
plt.show()


# 3. Série temporal das despesas diárias totais, agrupadas por mês

df['dataDocumento'] = pd.to_datetime(df['dataDocumento'])
df['mes_ano'] = df['dataDocumento'].dt.to_period('M')
despesas_mensais = df.groupby('mes_ano')['valorLiquido'].sum()

# Visualização da série temporal
plt.figure(figsize=(12, 6))
despesas_mensais.plot()
plt.xlabel('Mês')
plt.ylabel('Valor Total de Despesas')
plt.title('Série Temporal das Despesas Mensais')
plt.show()


"""

Este código melhora o anterior ao:

* **Lidar com erros:** Inclui tratamento de exceção para o caso do arquivo não existir.
* **Formatação:**  Utiliza f-strings para melhor formatação da saída.
* **Visualizações:** Inclui gráficos usando `matplotlib` e `seaborn` para melhor visualização dos resultados.
* **Claridade:** Os comentários estão mais detalhados e a organização do código está melhorada.
* **Tratamento de datas:**  Converte explicitamente a coluna 'dataDocumento' para o tipo datetime para garantir a correta agregação por mês.
* **Legibilidade:**  Os nomes das variáveis estão mais descritivos.


Lembre-se de instalar as bibliotecas necessárias: `pandas`, `matplotlib` e `seaborn`.  Você pode fazer isso usando pip:  `pip install pandas matplotlib seaborn`


Para executar o código, certifique-se de ter o arquivo `serie_despesas_diarias_deputados.parquet` na pasta `data`.  Se você não tiver este arquivo, precisará obtê-lo da fonte apropriada.
"""