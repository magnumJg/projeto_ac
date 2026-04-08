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
# TÍTULO E DESCRIÇÃO
# =========================
st.title("📈 Previsão de Aderência a Investimentos")
st.markdown(
    """
    Faça upload de um arquivo CSV com dados de clientes e descubra quais têm maior probabilidade de investir.
    """
)

st.divider()

# =========================
# CARREGAR MODELO
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
# LAYOUT EM COLUNAS
# =========================
col1, col2 = st.columns([1, 2])

# =========================
# COLUNA 1 - INPUT
# =========================
with col1:
    st.subheader("📂 Upload de dados")

    uploaded_file = st.file_uploader(
        "Envie um arquivo CSV",
        type=["csv"]
    )

    st.markdown("---")

    st.info(
        "Certifique-se de que o arquivo possui as mesmas colunas utilizadas no treinamento do modelo."
    )

# =========================
# COLUNA 2 - RESULTADO
# =========================
with col2:
    st.subheader("📊 Resultados")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("### 🔍 Prévia dos dados")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Gerar previsões"):
            with st.spinner("Processando dados..."):

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
                            return "Alta"
                        elif p > 0.4:
                            return "Média"
                        else:
                            return "Baixa"

                    df["categoria"] = df["probabilidade"].apply(classificar)

                    # =========================
                    # ORDENAÇÃO
                    # =========================
                    df = df.sort_values(by="probabilidade", ascending=False)

                    st.success("✅ Previsões geradas com sucesso!")

                    st.markdown("### 📈 Resultado final")
                    st.dataframe(df, use_container_width=True)

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
        st.warning("👈 Faça o upload de um arquivo CSV para começar.")