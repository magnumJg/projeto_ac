import streamlit as st
import pandas as pd
import pickle

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Previsão de Investimentos",
    page_icon="📈",
    layout="wide"
)

# =========================
# ESTILO (CSS CUSTOM)
# =========================
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        .stDownloadButton button {
            background-color: #2196F3;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("📈 Previsão de Aderência a Investimentos")
st.markdown(
    "Identifique rapidamente os clientes com maior probabilidade de investir e otimize suas campanhas."
)

st.divider()

# =========================
# CARREGAR MODELOS
# =========================
@st.cache_resource
def load_models():
    with open("models/modelo_onehotenc.pkl", "rb") as f:
        preprocess = pickle.load(f)
    with open("models/modelo_arvore.pkl", "rb") as f:
        model = pickle.load(f)
    return preprocess, model

preprocess, model = load_models()

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 2])

# =========================
# COLUNA 1 - INPUT
# =========================
with col1:
    st.subheader("📂 Upload de dados")

    uploaded_file = st.file_uploader(
        "Selecione seu arquivo CSV",
        type=["csv"]
    )

    st.markdown("### ℹ️ Instruções")
    st.markdown("""
    - O arquivo deve estar em formato CSV  
    - As colunas devem ser iguais às usadas no treinamento  
    - Não inclua a coluna target  
    """)

    with st.expander("📊 Entenda os dados"):
        st.markdown("""
        - **Idade:** pode influenciar o perfil de risco e a propensão a investir  
        - **Estado civil:** pode refletir estabilidade financeira  
        - **Escolaridade:** indica nível de conhecimento financeiro  
        - **Inadimplência:** reduz a probabilidade de investimento  
        - **Saldo:** quanto maior, maior a capacidade de investir  
        - **Fez empréstimo:** pode indicar comprometimento financeiro  
        - **Tempo do último contato:** mostra quão recente foi a interação  
        - **Número de contatos:** indica engajamento com o banco  
        """)

# =========================
# COLUNA 2 - RESULTADOS
# =========================
with col2:
    st.subheader("📊 Resultados")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("### 🔍 Prévia dos dados")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Gerar previsões"):
            with st.spinner("Analisando dados..."):

                try:
                    # =========================
                    # PREDIÇÃO
                    # =========================
                    X = preprocess.transform(df)
                    probs = model.predict_proba(X)[:, 1]

                    df["probabilidade"] = probs

                    # =========================
                    # CLASSIFICAÇÃO
                    # =========================
                    def classificar(p):
                        if p > 0.7:
                            return "🔥 Alta"
                        elif p > 0.4:
                            return "⚡ Média"
                        else:
                            return "❄️ Baixa"

                    df["categoria"] = df["probabilidade"].apply(classificar)

                    # =========================
                    # ORDENAÇÃO
                    # =========================
                    df = df.sort_values(by="probabilidade", ascending=False)

                    st.success("✅ Previsões geradas com sucesso!")

                    # =========================
                    # MÉTRICAS (DASHBOARD)
                    # =========================
                    alta = (df["categoria"] == "🔥 Alta").sum()
                    media = (df["categoria"] == "⚡ Média").sum()
                    baixa = (df["categoria"] == "❄️ Baixa").sum()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("🔥 Alta probabilidade", alta)
                    m2.metric("⚡ Média probabilidade", media)
                    m3.metric("❄️ Baixa probabilidade", baixa)

                    st.markdown("### 📈 Resultado final")

                    # =========================
                    # TABELA COM ESTILO
                    # =========================
                    def color_categoria(val):
                        if val == "Alta":
                            return "background-color: #c8f7c5"
                        elif val == "Média":
                            return "background-color: #fff3cd"
                        else:
                            return "background-color: #f8d7da"

                    st.dataframe(
                        df,
                        column_config={
                            "probabilidade": st.column_config.ProgressColumn(
                                "Probabilidade",
                                min_value=0,
                                max_value=1,
                            )
                        },
                        use_container_width=True
                    )

                    # =========================
                    # DOWNLOAD
                    # =========================
                    csv = df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="📥 Baixar resultado",
                        data=csv,
                        file_name="resultado_previsao.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Erro ao processar os dados: {e}")

    else:
        st.info("👈 Faça o upload de um arquivo CSV para começar.")