import streamlit as st
import pandas as pd
import pickle

def formata_numero(valor, prefixo = ''):
    for unidade in ['', 'mil']:
        if valor <1000:
            return f'{prefixo} {valor:.2f} {unidade}'
        valor /= 1000
    return f'{prefixo} {valor:.2f} milhões'

# Dicionário de regiões por UF
# regioes_brasil = {
#     'Norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO'],
#     'Nordeste': ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],
#     'Centro-Oeste': ['DF', 'GO', 'MT', 'MS'],
#     'Sudeste': ['ES', 'MG', 'RJ', 'SP'],
#     'Sul': ['PR', 'RS', 'SC']
# }

# def obter_regiao(uf):
#     for regiao, ufs in regioes_brasil.items():
#         if uf in ufs:
#             return regiao
#     return 'Indefinida'


st.title('OPERACOES DE PREDIÇÃO ')

uploaded_file = st.file_uploader("Envie seu CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())


@st.cache_resource
def load_models():
    with open("models/modelo_onehotenc.pkl", "rb") as f:
        preprocess = pickle.load(f)
    with open("models/modelo_arvore.pkl", "rb") as f:
        model = pickle.load(f)
    return preprocess, model

preprocess, model = load_models()

if st.button("Prever"):
    try:
        X = preprocess.transform(df)
        probs = model.predict_proba(X)[:, 1]

        df["probabilidade"] = probs
        df = df.sort_values(by="probabilidade", ascending=False)

        st.dataframe(df)

    except Exception as e:
        st.error(f"Erro: {e}")