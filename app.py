import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
import streamlit as st # Importar Streamlit

# Importar a biblioteca do Google Generative AI
import google.generativeai as genai
# Importar tipos específicos para gerenciar o histórico de chat
from google.generativeai.types import content_types as glm

# --- Configurações para melhor visualização dos gráficos ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'] 

# --- Configuração da API do Gemini ---
# IMPORTANTE: No Streamlit Cloud, adicione sua chave GEMINI_API_KEY aos segredos.
# Vá para 'Advanced settings' (ícone de engrenagem) -> 'Secrets'
# Adicione: GEMINI_API_KEY = "SUA_CHAVE_AQUI"
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    # st.success("API do Gemini configurada com sucesso.") # Não mostrar na UI para cada carga
except Exception as e:
    st.error(f"ERRO: Não foi possível configurar a API do Gemini. Certifique-se de que a chave 'GEMINI_API_KEY' está configurada nos segredos do Streamlit. Erro: {e}")
    st.stop() # Para a execução se a API não estiver configurada

# --- Carregamento do CSV Resultante ---
output_csv_filename = 'dados_cvm_mesclados.csv'
# df_resultante é carregado e armazenado no st.session_state para persistência
# entre as reruns do Streamlit.

if 'df_resultante' not in st.session_state:
    st.info(f"Tentando carregar o arquivo CSV: '{output_csv_filename}'...")
    try:
        df_resultante = pd.read_csv(
            output_csv_filename,
            delimiter=";",
            encoding="utf-8-sig"
        )
        st.session_state['df_resultante'] = df_resultante # Armazena no estado da sessão
        st.success(f"Arquivo '{output_csv_filename}' carregado com sucesso.")
    except FileNotFoundError:
        st.error(f"ERRO: Arquivo '{output_csv_filename}' não encontrado. Certifique-se de que o nome está correto e que foi incluído no repositório.")
        st.stop()
    except Exception as e:
        st.error(f"ERRO ao carregar o arquivo CSV: {e}")
        st.stop()
else:
    df_resultante = st.session_state['df_resultante']

if df_resultante.empty:
    st.warning("O DataFrame resultante está vazio. As funções de consulta não poderão operar.")
    st.stop()

# --- 3. Definição das Funções de Consulta (Ferramentas) ---
# AQUI VÃO TODAS AS SUAS FUNÇÕES 'GET_...'
# Certifique-se que o conteúdo das funções está como no último código fornecido (com int() e retorno dict {'text': ..., 'image_base64': ...})

def get_salario_medio_diretoria(df, year: int) -> dict: 
    year = int(year) 
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta."}
    if 'SALARIO' not in df.columns or 'ORGAO_ADMINISTRACAO' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (SALARIO, ORGAO_ADMINISTRACAO, ANO_REFER) não encontradas."}
    filtered_df = df[(df['ORGAO_ADMINISTRACAO'].str.contains('DIRETORIA', na=False, case=False)) &
                     (df['ANO_REFER'] == year)]
    if filtered_df.empty:
        return {'text': f"Nenhum dado encontrado para 'DIRETORIA' no ano {year}."}
    mean_salary = filtered_df['SALARIO'].mean()
    return {'text': f"O salário médio para membros da DIRETORIA em {year} é R$ {mean_salary:,.2f}."}

def get_top_companies_by_salary(df, num_companies: int, year: int = None) -> dict:
    num_companies = int(num_companies)
    if year is not None:
        year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta.", 'image_base64': None}
    if 'SALARIO' not in df.columns or 'NOME_COMPANHIA' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (SALARIO, NOME_COMPANHIA, ANO_REFER) não encontradas.", 'image_base64': None}
    df_filtered = df.copy()
    if year is None:
        last_year = df_filtered['ANO_REFER'].max()
        df_filtered = df_filtered[df_filtered['ANO_REFER'] == last_year]
        year_display = last_year
    else:
        df_filtered = df_filtered[df_filtered['ANO_REFER'] == year]
        year_display = year
    if df_filtered.empty:
        return {'text': f"Nenhum dado encontrado para o ano {year_display}.", 'image_base64': None}
    top_companies = df_filtered.groupby('NOME_COMPANHIA')['SALARIO'].sum().nlargest(num_companies).reset_index()
    if top_companies.empty:
        return {'text': f"Nenhuma empresa encontrada com dados de salário para o ano {year_display}.", 'image_base64': None}
    try:
        plt.figure(figsize=(12, 7))
        sns.barplot(x='SALARIO', y='NOME_COMPANHIA', data=top_companies, palette='viridis', hue='NOME_COMPANHIA', legend=False)
        plt.title(f'Top {num_companies} Empresas por Salário Total em {year_display}')
        plt.xlabel('Salário Total (R$)')
        plt.ylabel('Nome da Companhia')
        plt.ticklabel_format(style='plain', axis='x')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        result_text = f"As top {num_companies} empresas com maior salário total em {year_display} são:\n"
        for index, row in top_companies.iterrows():
            result_text += f"- {row['NOME_COMPANHIA']}: R$ {row['SALARIO']:,.2f}\n"
        return {'text': result_text, 'image_base64': image_base64}
    except Exception as e:
        plt.close()
        return {'text': f"ERRO ao gerar o gráfico de Top Empresas por Salário: {e}", 'image_base64': None}

def get_total_bonus_by_company(df, company_name: str, year: int, exact_match: bool = False) -> dict:
    year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta."}
    if 'BONUS' not in df.columns or 'NOME_COMPANHIA' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (BONUS, NOME_COMPANHIA, ANO_REFER) não encontradas."}
    if exact_match:
        filtered_df = df[(df['NOME_COMPANHIA'] == company_name) &
                         (df['ANO_REFER'] == year)]
    else:
        filtered_df = df[(df['NOME_COMPANHIA'].str.contains(company_name, na=False, case=False)) &
                         (df['ANO_REFER'] == year)]
    if filtered_df.empty:
        return {'text': f"Nenhum dado de bônus encontrado para '{company_name}' (busca {'exata' if exact_match else 'parcial'}) no ano {year}. Verifique o nome da empresa ou o ano."}
    total_bonus = filtered_df['BONUS'].sum()
    return {'text': f"O valor total de bônus pago por '{company_name}' em {year} foi de R$ {total_bonus:,.2f}."}

def get_sector_bonus_range(df, sector_name: str, year: int) -> dict:
    year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta."}
    if 'BONUS_VALOR_EFETIVO' not in df.columns and 'BONUS' not in df.columns:
        return {'text': "Nenhuma coluna de bônus (BONUS_VALOR_EFETIVO ou BONUS) encontrada para análise."}
    if 'SETOR_DE_ATIVDADE' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (SETOR_DE_ATIVDADE, ANO_REFER) não encontradas."}
    bonus_col = 'BONUS_VALOR_EFETIVO' if 'BONUS_VALOR_EFETIVO' in df.columns else 'BONUS'
    if bonus_col not in df.columns:
           return {'text': f"Coluna de bônus '{bonus_col}' não encontrada."}
    filtered_df = df[(df['SETOR_DE_ATIVDADE'].str.contains(sector_name, na=False, case=False)) &
                     (df['ANO_REFER'] == year)].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado de bônus encontrado para o setor '{sector_name}' no ano {year}."}
    min_bonus = filtered_df[bonus_col].min()
    max_bonus = filtered_df[bonus_col].max()
    mean_bonus = filtered_df[bonus_col].mean()
    return {'text': (f"Para o setor '{sector_name}' em {year}:\n"
                     f"   Bônus Mínimo: R$ {min_bonus:,.2f}\n"
                     f"   Bônus Máximo: R$ {max_bonus:,.2f}\n"
                     f"   Bônus Médio: R$ {mean_bonus:,.2f}")}

def get_remuneration_trend_by_orgao(df, orgao: str, start_year: int, end_year: int) -> dict:
    start_year = int(start_year)
    end_year = int(end_year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta.", 'image_base64': None}
    if 'VALOR_MEDIO_REMUNERACAO' not in df.columns and 'TOTAL_REMUNERACAO_ORGAO' not in df.columns:
        return {'text': "Nenhuma coluna de remuneração (VALOR_MEDIO_REMUNERACAO ou TOTAL_REMUNERACAO_ORGAO) encontrada para análise de tendência.", 'image_base64': None}
    if 'ORGAO_ADMINISTRACAO' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (ORGAO_ADMINISTRACAO, ANO_REFER) não encontradas.", 'image_base64': None}
    remuneration_col = 'VALOR_MEDIO_REMUNERACAO' if 'VALOR_MEDIO_REMUNERACAO' in df.columns else 'TOTAL_REMUNERACAO_ORGAO'
    if remuneration_col not in df.columns:
        return {'text': f"Coluna de remuneração '{remuneration_col}' não encontrada.", 'image_base64': None}
    filtered_df = df[(df['ORGAO_ADMINISTRACAO'].str.contains(orgao, na=False, case=False)) &
                     (df['ANO_REFER'] >= start_year) &
                     (df['ANO_REFER'] <= end_year)].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado encontrado para o órgão '{orgao}' entre os anos {start_year} e {end_year}.", 'image_base64': None}
    trend_data = filtered_df.groupby('ANO_REFER')[remuneration_col].mean().reset_index()
    trend_data = trend_data.sort_values('ANO_REFER')
    if trend_data.empty:
        return {'text': f"Não foi possível calcular a tendência para o órgão '{orgao}' entre {start_year} e {end_year} (dados insuficientes).", 'image_base64': None}
    try:
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='ANO_REFER', y=remuneration_col, data=trend_data, marker='o')
        plt.title(f'Tendência da Remuneração Média de {orgao} ({start_year}-{end_year})')
        plt.xlabel('Ano de Referência')
        plt.ylabel(f'Remuneração Média ({remuneration_col}) (R$)')
        plt.ticklabel_format(style='plain', axis='y')
        plt.xticks(trend_data['ANO_REFER'])
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        result_text = f"Tendência da remuneração média para o órgão '{orgao}' entre {start_year} e {end_year}:\n"
        for index, row in trend_data.iterrows():
            result_text += f"- Ano {int(row['ANO_REFER'])}: R$ {row[remuneration_col]:,.2f}\n"
        return {'text': result_text, 'image_base64': image_base64}
    except Exception as e:
        plt.close()
        return {'text': f"ERRO ao gerar o gráfico de Tendência de Remuneração: {e}", 'image_base64': None}

def get_avg_bonus_effective_by_sector(df, sector_name: str, year: int) -> dict:
    year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta."}
    if 'BONUS_VALOR_EFETIVO' not in df.columns and 'BONUS' not in df.columns:
        return {'text': "Nenhuma coluna de bônus (BONUS_VALOR_EFETIVO ou BONUS) encontrada para análise."}
    if 'SETOR_DE_ATIVDADE' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (SETOR_DE_ATIVDADE, ANO_REFER) não encontradas."}
    bonus_col = 'BONUS_VALOR_EFETIVO' if 'BONUS_VALOR_EFETIVO' in df.columns else 'BONUS'
    if bonus_col not in df.columns:
           return {'text': f"Coluna de bônus '{bonus_col}' não encontrada."}
    filtered_df = df[(df['SETOR_DE_ATIVDADE'].str.contains(sector_name, na=False, case=False)) &
                     (df['ANO_REFER'] == year)].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado de bônus efetivo encontrado para o setor '{sector_name}' no ano {year}."}
    avg_bonus_effective = filtered_df[bonus_col].mean()
    return {'text': f"O valor médio do bônus efetivo para o setor '{sector_name}' em {year} é R$ {avg_bonus_effective:,.2f}."}

def get_top_sectors_by_avg_total_remuneration(df, num_sectors: int, year: int) -> dict:
    num_sectors = int(num_sectors)
    year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta.", 'image_base64': None}
    if 'TOTAL_REMUNERACAO_ORGAO' not in df.columns or 'SETOR_DE_ATIVDADE' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (TOTAL_REMUNERACAO_ORGAO, SETOR_DE_ATIVDADE, ANO_REFER) não encontradas.", 'image_base64': None}
    filtered_df = df[df['ANO_REFER'] == year].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado encontrado para o ano {year}.", 'image_base64': None}
    avg_remuneration_by_sector = filtered_df.groupby('SETOR_DE_ATIVDADE')['TOTAL_REMUNERACAO_ORGAO'].mean().nlargest(num_sectors).reset_index()
    if avg_remuneration_by_sector.empty:
        return {'text': f"Nenhum setor encontrado com remuneração média total para o ano {year}.", 'image_base64': None}
    try:
        plt.figure(figsize=(12, 7))
        sns.barplot(x='TOTAL_REMUNERACAO_ORGAO', y='SETOR_DE_ATIVDADE', data=avg_remuneration_by_sector, palette='magma', hue='SETOR_DE_ATIVDADE', legend=False)
        plt.title(f'Top {num_sectors} Setores por Remuneração Média Total em {year}')
        plt.xlabel('Remuneração Média Total (R$)')
        plt.ylabel('Setor de Atividade')
        plt.ticklabel_format(style='plain', axis='x')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        result_text = f"Os top {num_sectors} setores com a maior remuneração média total em {year} são:\n"
        for index, row in avg_remuneration_by_sector.iterrows():
            result_text += f"- {row['SETOR_DE_ATIVDADE']}: R$ {row['TOTAL_REMUNERACAO_ORGAO']:,.2f}\n"
        return {'text': result_text, 'image_base64': image_base64}
    except Exception as e:
        plt.close()
        return {'text': f"ERRO ao gerar o gráfico de Top Setores por Remuneração: {e}", 'image_base64': None}

def get_remuneration_as_percentage_of_revenue(df, num_companies: int, sector_name: str, year: int) -> dict:
    num_companies = int(num_companies)
    year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta."}
    if 'TOTAL_REMUNERACAO_ORGAO' not in df.columns or 'RECEITA' not in df.columns or \
       'SETOR_DE_ATIVDADE' not in df.columns or 'ANO_REFER' not in df.columns or \
       'NOME_COMPANHIA' not in df.columns:
        return {'text': "Colunas necessárias (TOTAL_REMUNERACAO_ORGAO, RECEITA, SETOR_DE_ATIVDADE, ANO_REFER, NOME_COMPANHIA) não encontradas."}
    filtered_df = df[(df['SETOR_DE_ATIVDADE'].str.contains(sector_name, na=False, case=False)) &
                     (df['ANO_REFER'] == year)].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado encontrado para o setor '{sector_name}' no ano {year}."}
    company_data = filtered_df.groupby('NOME_COMPANHIA').agg(
        Total_Remuneracao=('TOTAL_REMUNERACAO_ORGAO', 'sum'),
        Receita=('RECEITA', 'sum')
    ).reset_index()
    company_data = company_data[company_data['Receita'].fillna(0) > 0]
    if company_data.empty:
        return {'text': f"Nenhuma empresa com receita válida encontrada para o setor '{sector_name}' no ano {year}."}
    company_data['Remuneracao_Percentual_Receita'] = (company_data['Total_Remuneracao'] / company_data['Receita']) * 100
    top_companies = company_data.nlargest(num_companies, 'Receita')
    top_companies = top_companies.sort_values(by='Remuneracao_Percentual_Receita', ascending=False)
    result_text = f"Remuneração Total como Percentual da Receita para as top {num_companies} empresas do setor '{sector_name}' em {year} (ordenado por %):\n"
    for index, row in top_companies.iterrows():
        result_text += (f"- {row['NOME_COMPANHIA']}: Receita R$ {row['Receita']:,.2f}, "
                        f"Remuneração Total R$ {row['Total_Remuneracao']:,.2f}, "
                        f"Percentual: {row['Remuneracao_Percentual_Receita']:,.2f}%\n")
    return {'text': result_text}

def get_correlation_members_bonus(df, year: int) -> dict:
    year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta.", 'image_base64': None}
    if 'NUM_MEMBROS_REMUNERADOS_TOTAL' not in df.columns or 'BONUS' not in df.columns or \
       'NOME_COMPANHIA' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (NUM_MEMBROS_REMUNERADOS_TOTAL, BONUS, NOME_COMPANHIA, ANO_REFER) não encontradas.", 'image_base64': None}
    filtered_df = df[df['ANO_REFER'] == year].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado encontrado para o ano {year}.", 'image_base64': None}
    company_aggregated = filtered_df.groupby('NOME_COMPANHIA').agg(
        Total_Membros_Remunerados=('NUM_MEMBROS_REMUNERADOS_TOTAL', 'sum'),
        Total_Bonus=('BONUS', 'sum')
    ).reset_index()
    company_aggregated = company_aggregated.dropna(subset=['Total_Membros_Remunerados', 'Total_Bonus'])
    company_aggregated = company_aggregated[(company_aggregated['Total_Membros_Remunerados'] > 0) &
                                            (company_aggregated['Total_Bonus'] > 0)]
    if company_aggregated.empty:
        return {'text': f"Dados insuficientes para calcular a correlação entre membros remunerados e bônus para o ano {year}.", 'image_base64': None}
    correlation = company_aggregated['Total_Membros_Remunerados'].corr(company_aggregated['Total_Bonus'])
    try:
        plt.figure(figsize=(12, 7))
        sns.scatterplot(x='Total_Membros_Remunerados', y='Total_Bonus', data=company_aggregated, hue='NOME_COMPANHIA', legend='brief', s=100)
        plt.title(f'Correlação entre Membros Remunerados e Bônus Total por Empresa em {year}\nCorrelação: {correlation:,.2f}')
        plt.xlabel('Número Total de Membros Remunerados')
        plt.ylabel('Bônus Total (R$)')
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        if len(company_aggregated['NOME_COMPANHIA'].unique()) > 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
            plt.legend(loc='best')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        result_text = (f"A correlação entre o número total de membros remunerados e o bônus total pago por empresa em {year} é de {correlation:,.2f}.\n"
                       f"Um valor próximo de 1 indica uma correlação positiva forte, -1 uma correlação negativa forte, e 0 nenhuma correlação.\n")
        return {'text': result_text, 'image_base64': image_base64}
    except Exception as e:
        plt.close()
        return {'text': f"ERRO ao gerar o gráfico de Correlação: {e}", 'image_base64': None}

def get_avg_remuneration_by_orgao_segment(df, orgao_name: str, year: int) -> dict:
    year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta.", 'image_base64': None}
    if 'TOTAL_REMUNERACAO_ORGAO' not in df.columns or 'ORGAO_ADMINISTRACAO' not in df.columns or \
       'SETOR_DE_ATIVDADE' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (TOTAL_REMUNERACAO_ORGAO, ORGAO_ADMINISTRACAO, SETOR_DE_ATIVDADE, ANO_REFER) não encontradas.", 'image_base64': None}
    filtered_df = df[(df['ORGAO_ADMINISTRACAO'].str.contains(orgao_name, na=False, case=False)) &
                     (df['ANO_REFER'] == year)].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado encontrado para o órgão '{orgao_name}' no ano {year}.", 'image_base64': None}
    remuneration_by_segment = filtered_df.groupby('SETOR_DE_ATIVDADE')['TOTAL_REMUNERACAO_ORGAO'].mean().reset_index()
    remuneration_by_segment = remuneration_by_segment.sort_values(by='TOTAL_REMUNERACAO_ORGAO', ascending=False)
    if remuneration_by_segment.empty:
        return {'text': f"Nenhum dado de remuneração média por segmento encontrado para o órgão '{orgao_name}' no ano {year}.", 'image_base64': None}
    try:
        plt.figure(figsize=(12, 7))
        sns.barplot(x='TOTAL_REMUNERACAO_ORGAO', y='SETOR_DE_ATIVDADE', data=remuneration_by_segment, palette='crest', hue='SETOR_DE_ATIVDADE', legend=False)
        plt.title(f'Remuneração Média Total de {orgao_name} por Setor de Atividade em {year}')
        plt.xlabel('Remuneração Média Total (R$)')
        plt.ylabel('Setor de Atividade')
        plt.ticklabel_format(style='plain', axis='x')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        result_text = f"Média da remuneração total para '{orgao_name}' por Setor de Atividade em {year}:\n"
        for index, row in remuneration_by_segment.iterrows():
            result_text += f"- {row['SETOR_DE_ATIVDADE']}: R$ {row['TOTAL_REMUNERACAO_ORGAO']:,.2f}\n"
        return {'text': result_text, 'image_base64': image_base64}
    except Exception as e:
        plt.close()
        return {'text': f"ERRO ao gerar o gráfico de Remuneração Média por Órgão e Segmento: {e}", 'image_base64': None}

def get_remuneration_structure_proportion(df, orgao_name: str, year: int) -> dict:
    year = int(year)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta.", 'image_base64': None}
    relevant_cols = ['SALARIO', 'BONUS', 'PARTICIPACAO_RESULTADOS', 'PRECO_MEDIO_PONDERADO_OPCOES_EM_ABERTO', 'VL_ACOES_RESTRITAS']
    for col in relevant_cols:
        if col not in df.columns:
            return {'text': f"Coluna '{col}' necessária para inferir a estrutura de remuneração não encontrada.", 'image_base64': None}
    if 'ORGAO_ADMINISTRACAO' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (ORGAO_ADMINISTRACAO, ANO_REFER) não encontradas.", 'image_base64': None}
    filtered_df = df[(df['ORGAO_ADMINISTRACAO'].str.contains(orgao_name, na=False, case=False)) &
                     (df['ANO_REFER'] == year)].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado encontrado para o órgão '{orgao_name}' no ano {year}.", 'image_base64': None}
    def classify_remuneration_structure(row):
        has_fixa = pd.notna(row['SALARIO']) and row['SALARIO'] > 0
        has_variavel = (pd.notna(row['BONUS']) and row['BONUS'] > 0) or \
                       (pd.notna(row['PARTICIPACAO_RESULTADOS']) and row['PARTICIPACAO_RESULTADOS'] > 0)
        has_acoes = (pd.notna(row['PRECO_MEDIO_PONDERADO_OPCOES_EM_ABERTO']) and row['PRECO_MEDIO_PONDERADO_OPCOES_EM_ABERTO'] > 0) or \
                    (pd.notna(row['VL_ACOES_RESTRITAS']) and row['VL_ACOES_RESTRITAS'] > 0)
        if has_fixa and has_variavel and has_acoes:
            return "Fixa, Variável e Ações"
        elif has_fixa and has_variavel:
            return "Fixa e Variável"
        elif has_fixa and has_acoes:
            return "Fixa e Ações"
        elif has_fixa:
            return "Somente Fixa"
        else:
            return "Outra/Não Classificada"
    filtered_df['Estrutura_Remuneracao'] = filtered_df.apply(classify_remuneration_structure, axis=1)
    structure_counts = filtered_df['Estrutura_Remuneracao'].value_counts(normalize=True).reset_index()
    structure_counts.columns = ['Estrutura', 'Proporcao']
    structure_counts['Proporcao'] = structure_counts['Proporcao'] * 100
    if structure_counts.empty:
        return {'text': f"Nenhuma estrutura de remuneração classificada para o órgão '{orgao_name}' no ano {year}.", 'image_base64': None}
    try:
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Proporcao', y='Estrutura', data=structure_counts, palette='pastel', hue='Estrutura', legend=False)
        plt.title(f'Estruturas de Remuneração para {orgao_name} em {year} (% de Ocorrências)')
        plt.xlabel('Proporção (%)')
        plt.ylabel('Estrutura de Remuneração')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        result_text = f"Proporção das estruturas de remuneração para '{orgao_name}' em {year}:\n"
        for index, row in structure_counts.iterrows():
            result_text += f"- {row['Estrutura']}: {row['Proporcao']:,.2f}%\n"
        return {'text': result_text, 'image_base64': image_base64}
    except Exception as e:
        plt.close()
        return {'text': f"ERRO ao gerar o gráfico de Estruturas de Remuneração: {e}", 'image_base64': None}

def get_top_bottom_remuneration_values(df, orgao_name: str, year: int, num_companies: int = 5) -> dict:
    year = int(year)
    num_companies = int(num_companies)
    if df.empty: return {'text': "DataFrame vazio. Não foi possível realizar a consulta."}
    if 'TOTAL_REMUNERACAO_ORGAO' not in df.columns or 'NOME_COMPANHIA' not in df.columns or \
       'ORGAO_ADMINISTRACAO' not in df.columns or 'ANO_REFER' not in df.columns:
        return {'text': "Colunas necessárias (TOTAL_REMUNERACAO_ORGAO, NOME_COMPANHIA, ORGAO_ADMINISTRACAO, ANO_REFER) não encontradas."}
    filtered_df = df[(df['ORGAO_ADMINISTRACAO'].str.contains(orgao_name, na=False, case=False)) &
                     (df['ANO_REFER'] == year)].copy()
    if filtered_df.empty:
        return {'text': f"Nenhum dado encontrado para o órgão '{orgao_name}' no ano {year}."}
    unique_remuneration = filtered_df.groupby(['NOME_COMPANHIA', 'ORGAO_ADMINISTRACAO', 'ANO_REFER'])['TOTAL_REMUNERACAO_ORGAO'].sum().reset_index()
    if unique_remuneration.empty:
        return {'text': f"Nenhum dado de remuneração total único encontrado para o órgão '{orgao_name}' no ano {year}."}
    top_values = unique_remuneration.nlargest(num_companies, 'TOTAL_REMUNERACAO_ORGAO')
    bottom_values = unique_remuneration[unique_remuneration['TOTAL_REMUNERACAO_ORGAO'] > 0].nsmallest(num_companies, 'TOTAL_REMUNERACAO_ORGAO')
    result_text = f"Maiores e Menores {num_companies} Remunerações Totais para '{orgao_name}' em {year}:\n\n"
    result_text += "--- Maiores Remunerações ---\n"
    if not top_values.empty:
        for index, row in top_values.iterrows():
            result_text += f"- {row['NOME_COMPANHIA']}: R$ {row['TOTAL_REMUNERACAO_ORGAO']:,.2f}\n"
    else:
        result_text += "Nenhum dado de maiores remunerações.\n"
    result_text += "\n--- Menores Remunerações (excluindo zeros/nulos) ---\n"
    if not bottom_values.empty:
        for index, row in bottom_values.iterrows():
            result_text += f"- {row['NOME_COMPANHIA']}: R$ {row['TOTAL_REMUNERACAO_ORGAO']:,.2f}\n"
    else:
        result_text += "Nenhum dado de menores remunerações.\n"
    return {'text': result_text}


# --- 4. Definição das Ferramentas (Tool Specifications) para o Gemini ---
O erro `AttributeError: module 'google.generativeai.types.content_types' has no attribute 'Type'` ocorre porque a importação `from google.generativeai.types import content_types as glm` torna `glm` um módulo que lida com o *conteúdo* das mensagens (`Text`, `Part`, `Content`), mas não expõe o `Type` para a **definição de esquemas de ferramentas (schemas de parâmetros)**.

Para a definição dos tipos de esquema das suas ferramentas (OBJECT, INTEGER, STRING, BOOLEAN), você deve usar a referência correta, que é `genai.protos.Schema.Type`.

**A solução é reverter o uso de `glm.Type` para `genai.protos.Schema.Type` em todas as definições de tipo dentro da sua lista `tools`.**

**Substitua todo o bloco `tools = [...]` no seu arquivo `app.py` pelo código abaixo:**

```python
# --- 4. Definição das Ferramentas (Tool Specifications) para o Gemini ---
# Cada função que o Gemini pode chamar precisa de uma declaração.

tools = [
    genai.protos.FunctionDeclaration(
        name='get_salario_medio_diretoria',
        description='Calcula e retorna o salário médio de membros do órgão de administração "DIRETORIA" para um ano específico. Use esta ferramenta quando a pergunta envolver o salário médio da diretoria.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a consulta, ex: 2025'), # Corrigido aqui
            },
            required=['year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_top_companies_by_salary',
        description='Identifica e retorna as top N empresas com a maior soma total de SALARIO em um determinado ano (ou o último ano disponível se não especificado), e gera um gráfico de barras visualizando esses dados. Use para perguntas sobre as empresas que mais pagam salários.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'num_companies': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O número de empresas a serem retornadas, ex: 10, 5, 3'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a consulta. Se omitido, a ferramenta usará o último ano com dados.'), # Corrigido aqui
            },
            required=['num_companies'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_total_bonus_by_company',
        description='Calcula e retorna o valor total de BÔNUS pago por uma empresa específica (NOME_COMPANHIA) em um determinado ano (ANO_REFER). Pode ser usado para busca exata ou parcial do nome da empresa. Use para saber o valor total de bônus de uma empresa específica.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'company_name': genai.protos.Schema(type=genai.protos.Schema.Type.STRING, description='O nome completo da empresa ou parte do nome, ex: "BANCO DO BRASIL S.A.", "ITAU"'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a consulta, ex: 2025'), # Corrigido aqui
                'exact_match': genai.protos.Schema(type=genai.protos.Schema.Type.BOOLEAN, description='Opcional. Se True, busca o nome exato. Se False (padrão), busca por conteúdo.'), # Corrigido aqui
            },
            required=['company_name', 'year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_sector_bonus_range',
        description='Calcula e retorna o bônus mínimo, máximo e médio para empresas dentro de um setor de atividade específico (SETOR_DE_ATIVDADE) em um determinado ano. Use para analisar a faixa de bônus em um setor.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'sector_name': genai.protos.Schema(type=genai.protos.Schema.Type.STRING, description='O nome do setor de atividade, ex: "BANCARIO", "SAUDE", "VAREJO"'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a consulta, ex: 2025'), # Corrigido aqui
            },
            required=['sector_name', 'year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_remuneration_trend_by_orgao',
        description='Analisa a evolução da remuneração média de um órgão de administração (ORGAO_ADMINISTRACAO) ao longo de um período de anos e gera um gráfico de linha. Use para ver a tendência de remuneração de um órgão específico ao longo do tempo.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'orgao': genai.protos.Schema(type=genai.protos.Schema.Type.STRING, description='O nome do órgão de administração, ex: "CONSELHO DE ADMINISTRACAO", "DIRETORIA"'), # Corrigido aqui
                'start_year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de início do período, ex: 2023'), # Corrigido aqui
                'end_year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de fim do período, ex: 2025'), # Corrigido aqui
            },
            required=['orgao', 'start_year', 'end_year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_avg_bonus_effective_by_sector',
        description='Calcula e retorna o valor médio do bônus efetivo pago por empresas de um setor específico (SETOR_DE_ATIVDADE) em um determinado ano. Use para entender o bônus médio em um setor.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'sector_name': genai.protos.Schema(type=genai.protos.Schema.Type.STRING, description='O nome do setor, ex: "FINANCEIRO", "SAUDE"'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência, ex: 2024'), # Corrigido aqui
            },
            required=['sector_name', 'year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_top_sectors_by_avg_total_remuneration',
        description='Identifica os N setores com a maior remuneração total média (TOTAL_REMUNERACAO_ORGAO) em um ano específico e gera um gráfico. Use para comparar o nível de remuneração entre diferentes setores.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'num_sectors': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O número de setores a serem retornados, ex: 5, 3'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a consulta, ex: 2025'), # Corrigido aqui
            },
            required=['num_sectors', 'year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_remuneration_as_percentage_of_revenue',
        description='Calcula a remuneração total de um órgão como percentual da receita para as N maiores empresas de um setor em um ano. Use para analisar a proporção da remuneração em relação ao faturamento.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'num_companies': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O número de empresas a serem retornadas, ex: 3, 5'), # Corrigido aqui
                'sector_name': genai.protos.Schema(type=genai.protos.Schema.Type.STRING, description='O nome do setor de atividade, ex: "VAREJO", "TECNOLOGIA DA INFORMACAO"'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Type.INTEGER, description='O ano de referência para a consulta, ex: 2025'), # Corrigido aqui
            },
            required=['num_companies', 'sector_name', 'year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_correlation_members_bonus',
        description='Analisa a correlação entre o número de membros remunerados e o bônus total para um ano específico, gerando um gráfico de dispersão. Use para entender a relação entre o tamanho da equipe remunerada e o total de bônus.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a análise, ex: 2025'), # Corrigido aqui
            },
            required=['year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_avg_remuneration_by_orgao_segment',
        description='Calcula a média da remuneração total para um órgão específico por segmento de listagem (setor de atividade) em um dado ano. Use para comparar a remuneração de um órgão em diferentes setores.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'orgao_name': genai.protos.Schema(type=genai.protos.Schema.Type.STRING, description='O nome do órgão de administração, ex: "DIRETORIA", "CONSELHO FISCAL"'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a consulta, ex: 2025'), # Corrigido aqui
            },
            required=['orgao_name', 'year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_remuneration_structure_proportion',
        description='Calcula a proporção de empresas que utilizam diferentes estruturas de remuneração para um órgão em um ano. Use para entender como as empresas remuneram seus membros.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'orgao_name': genai.protos.Schema(type=genai.protos.Schema.Type.STRING, description='O nome do órgão de administração, ex: "CONSELHO DE ADMINISTRACAO", "DIRETORIA"'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a análise, ex: 2025'), # Corrigido aqui
            },
            required=['orgao_name', 'year'],
        ),
    ),
    genai.protos.FunctionDeclaration(
        name='get_top_bottom_remuneration_values',
        description='Lista os N maiores e N menores valores de remuneração total (TOTAL_REMUNERACAO_ORGAO) para um órgão de administração específico em um dado ano. Use para identificar as empresas com os maiores e menores pagamentos a um órgão.',
        parameters=genai.protos.Schema(
            type=genai.protos.Schema.Type.OBJECT, # Corrigido aqui
            properties={
                'orgao_name': genai.protos.Schema(type=genai.protos.Schema.Type.STRING, description='O nome do órgão de administração, ex: "DIRETORIA", "CONSELHO FISCAL"'), # Corrigido aqui
                'year': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='O ano de referência para a consulta, ex: 2025'), # Corrigido aqui
                'num_companies': genai.protos.Schema(type=genai.protos.Schema.Type.INTEGER, description='Opcional. O número de empresas a serem listadas para top/bottom. O padrão é 5.'), # Corrigido aqui
            },
            required=['orgao_name', 'year'],
        ),
    ),
]
```
# --- 5. Inicialização do Modelo Gemini com Ferramentas ---
model = genai.GenerativeModel(model_name='gemini-1.5-flash', tools=tools)

# --- 6. Função para Interagir com o Agente ---
def chat_with_data_agent(query: str):
    """
    Simula a interação com o agente de IA, processando a query do usuário,
    chamando ferramentas e exibindo a resposta.
    """
    if st.session_state['df_resultante'].empty:
        st.error("O DataFrame está vazio. Não é possível realizar consultas. Verifique o carregamento dos dados.")
        return

    # Definindo a instrução do sistema (System Prompt)
    system_instruction = """
    Você é um especialista em análise de dados de remuneração de administradores para companhias de capital aberto no Brasil. Sua função é responder a perguntas do usuário baseando-se exclusivamente nos dados fornecidos a partir de um arquivo CSV que contém informações detalhadas sobre salários, bônus e outras formas de remuneração para a Diretoria Estatutária, Conselho de Administração e Conselho Fiscal.

    Seu conhecimento é focado nos dados do CSV. Você pode realizar as seguintes análises utilizando as ferramentas disponíveis:

    - **Salário Médio da Diretoria:** Obter o salário médio da diretoria para um ano específico.
    - **Top Empresas por Salário:** Identificar as empresas com os maiores salários totais em um determinado ano, com a opção de gerar um gráfico.
    - **Bônus Total por Empresa:** Consultar o valor total de bônus pago por uma empresa específica em um ano.
    - **Faixa de Bônus por Setor:** Analisar o bônus mínimo, máximo e médio para um setor e ano específicos.
    - **Tendência de Remuneração por Órgão:** Acompanhar a evolução da remuneração média de um órgão de administração (ex: Conselho de Administração, Diretoria) ao longo de um período, com a opção de gerar um gráfico.
    - **Bônus Médio Efetivo por Setor:** Calcular o valor médio do bônus efetivo em um setor específico para um dado ano.
    - **Top Setores por Remuneração Média:** Listar os setores com a maior remuneração total média em um ano, com a opção de gerar um gráfico.
    - **Remuneração como % da Receita:** Calcular a remuneração total como percentual da receita para empresas de um setor em um ano.
    - **Correlação Membros x Bônus:** Analisar a correlação entre o número de membros remunerados e o bônus total, com a opção de gerar um gráfico de dispersão.
    - **Remuneração Média por Órgão e Segmento:** Calcular a média da remuneração total para um órgão específico por segmento de listagem (setor de atividade) em um ano, com a opção de gerar um gráfico.
    - **Proporção da Estrutura de Remuneração:** Determinar a proporção de empresas que utilizam diferentes estruturas de remuneração (fixa, variável, ações) para um órgão em um ano, com a opção de gerar um gráfico.
    - **Maiores e Menores Remunerações:** Listar os maiores e menores valores de remuneração total para um órgão em um ano.

    Sempre que a pergunta envolver números (como o número de empresas, o ano), use os valores fornecidos pelo usuário. Se um gráfico for solicitado ou puder complementar a resposta, utilize a ferramenta adequada para gerá-lo.

    Se a informação solicitada não puder ser obtida com as ferramentas disponíveis ou não estiver no CSV, informe ao usuário de forma clara e objetiva. Evite dar informações genéricas ou especulativas.
    """

    # --- Tratamento do Histórico para start_chat ---
    # `st.session_state.messages` precisa ser convertido para o formato esperado pelo `genai.GenerativeModel.start_chat`
    # Cada entrada deve ser um objeto glm.Content.
    chat_history_for_gemini = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history_for_gemini.append(glm.Content(role="user", parts=[glm.Part(text=msg["parts"][0])]))
        elif msg["role"] == "model":
            # As respostas do modelo são armazenadas como glm.Content diretamente
            # ou como um dicionário que contém a resposta da ferramenta (text/image_base64)
            if isinstance(msg["parts"][0], str): # Se for texto puro (resposta final do modelo)
                chat_history_for_gemini.append(glm.Content(role="model", parts=[glm.Part(text=msg["parts"][0])]))
            elif isinstance(msg["parts"][0], dict) and 'text' in msg["parts"][0]:
                # Se for a resposta de uma ferramenta (que tem text e talvez image_base64)
                # O importante é o texto da resposta da ferramenta ser enviado de volta ao modelo
                # como parte da FunctionResponse, não no histórico principal de forma complexa.
                # O histórico aqui deve ser o que o modelo *gerou*, não o tool_output.
                # A resposta do modelo após a ferramenta será um Content com texto.
                # Se o histórico já contém a parte gerada pelo modelo (e não a saída bruta da ferramenta), está ok.
                # No código atual, st.session_state.messages.append({"role": "model", "parts": [response.candidates[0].content.parts[0]]})
                # então response.candidates[0].content.parts[0] já é o Proto Part correto.
                # Podemos simplesmente passar o 'content' original do response se o armazenarmos assim.

                # Para simplificar e evitar problemas de proto, vamos apenas reconstruir messages com texto.
                # Se o histórico é apenas para o LLM aprender, texto é o suficiente.
                chat_history_for_gemini.append(glm.Content(role="model", parts=[glm.Part(text=msg["text"])]))
            else: # Caso seja um objeto proto do Gemini Content direto
                chat_history_for_gemini.append(msg["parts"][0]) # Já é o Content proto

    # Iniciar o chat com o modelo e a instrução do sistema
    # Passamos o histórico construído `chat_history_for_gemini`
    chat = model.start_chat(history=chat_history_for_gemini, system_instruction=system_instruction)
    response = chat.send_message(query)

    # ... (restante da função chat_with_data_agent permanece o mesmo) ...
    # O código abaixo está bom para a lógica.
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        function_name = function_call.name
        function_args = dict(function_call.args) 
        
        #st.write(f"Agente (chamando ferramenta): {function_name} com args {function_args}") # Para depuração

        tool_output = {} 
        try:
            if function_name == 'get_salario_medio_diretoria':
                tool_output = get_salario_medio_diretoria(df_resultante, **function_args)
            elif function_name == 'get_top_companies_by_salary':
                tool_output = get_top_companies_by_salary(df_resultante, **function_args)
            elif function_name == 'get_total_bonus_by_company':
                tool_output = get_total_bonus_by_company(df_resultante, **function_args)
            elif function_name == 'get_sector_bonus_range':
                tool_output = get_sector_bonus_range(df_resultante, **function_args)
            elif function_name == 'get_remuneration_trend_by_orgao':
                tool_output = get_remuneration_trend_by_orgao(df_resultante, **function_args)
            elif function_name == 'get_avg_bonus_effective_by_sector':
                tool_output = get_avg_bonus_effective_by_sector(df_resultante, **function_args)
            elif function_name == 'get_top_sectors_by_avg_total_remuneration':
                tool_output = get_top_sectors_by_avg_total_remuneration(df_resultante, **function_args)
            elif function_name == 'get_remuneration_as_percentage_of_revenue':
                tool_output = get_remuneration_as_percentage_of_revenue(df_resultante, **function_args)
            elif function_name == 'get_correlation_members_bonus':
                tool_output = get_correlation_members_bonus(df_resultante, **function_args)
            elif function_name == 'get_avg_remuneration_by_orgao_segment':
                tool_output = get_avg_remuneration_by_orgao_segment(df_resultante, **function_args)
            elif function_name == 'get_remuneration_structure_proportion':
                tool_output = get_remuneration_structure_proportion(df_resultante, **function_args)
            elif function_name == 'get_top_bottom_remuneration_values':
                tool_output = get_top_bottom_remuneration_values(df_resultante, **function_args)
            else:
                tool_output = {'text': f"Erro: Função '{function_name}' não reconhecida ou não implementada."}
        except Exception as e:
            tool_output = {'text': f"Erro ao executar a função '{function_name}': {e}"}

        # Enviar o resultado da ferramenta de volta para o modelo
        response = chat.send_message(glm.Part(function_response=glm.FunctionResponse( # Use glm.Part e glm.FunctionResponse
            name=function_name,
            response=tool_output 
        )))
    
    # --- Atualização do Histórico e Exibição para Streamlit ---
    # Adicionar a resposta do agente ao histórico de mensagens
    # Armazenar o objeto Content gerado pelo modelo
    st.session_state.messages.append(response.candidates[0].content)

    # Exibir a resposta final do modelo na interface do Streamlit
    with st.chat_message("assistant"):
        # Se a resposta do modelo contém partes (texto ou função/ferramenta)
        for part in response.candidates[0].content.parts:
            if glm.is_text(part): # Se for um Content de texto
                st.markdown(part.text)
            elif glm.is_function_call(part): # Se for uma chamada de função (não exibimos diretamente ao usuário)
                # st.write(f"Agente: Chamando ferramenta '{part.function_call.name}'...")
                pass
            elif glm.is_function_response(part): # Se for uma resposta de função (o LLM reage a ela)
                # A resposta final que o LLM gera após a ferramenta já é Content com texto.
                # Aqui você exibe a imagem se houver.
                if isinstance(tool_output, dict) and 'image_base64' in tool_output and tool_output['image_base64']:
                    st.image(base64.b64decode(tool_output['image_base64']), caption="Gráfico gerado pelo agente")
                
            # Adicionalmente, se tool_output veio com imagem e texto, e o texto não foi absorvido no LLM final response
            if 'image_base64' in tool_output and tool_output['image_base64'] and not glm.is_function_response(response.candidates[0].content.parts[0]):
                 # Isso é para casos onde o LLM não gera texto adicional, só a ferramenta.
                 # Mas o fluxo ideal é o LLM gerar texto baseado na saída da ferramenta.
                 pass


# --- Interface do Streamlit ---
st.title("💰 Agente de Análise de Remunerações CVM")
st.markdown("Faça perguntas sobre os dados de remuneração de administradores de companhias de capital aberto.")

# Inicializar histórico de chat no estado da sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens anteriores no chat
for message_content in st.session_state.messages: # Cada message_content agora é um glm.Content
    with st.chat_message(message_content.role):
        for part in message_content.parts:
            if glm.is_text(part):
                st.markdown(part.text)
            elif glm.is_function_call(part):
                # Não exibe a chamada da ferramenta diretamente ao usuário final
                # st.code(f"Chamada de função: {part.function_call.name}({part.function_call.args})")
                pass
            elif glm.is_function_response(part):
                # A resposta da ferramenta pode ser um dicionário.
                # Se você quer exibir a saída bruta da ferramenta:
                # st.json(part.function_response) # Para depuração

                # Se a resposta da função continha uma imagem, exiba-a
                if isinstance(part.function_response, dict) and 'image_base64' in part.function_response and part.function_response['image_base64']:
                    st.image(base64.b64decode(part.function_response['image_base64']), caption="Gráfico gerado (Histórico)")
                # Se a resposta da função continha texto, mas o modelo não o processou em sua resposta principal,
                # e você quer mostrar esse texto da ferramenta:
                # if isinstance(part.function_response, dict) and 'text' in part.function_response:
                #     st.markdown(f"_(Saída da ferramenta)_: {part.function_response['text']}")


# Campo de entrada para o usuário
user_query = st.chat_input("Pergunte algo sobre os dados da CVM:")

if user_query:
    # Adicionar a pergunta do usuário ao histórico de mensagens como um objeto glm.Content
    st.session_state.messages.append(glm.Content(role="user", parts=[glm.Part(text=user_query)]))
    
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Chamar a função do agente
    chat_with_data_agent(user_query)
