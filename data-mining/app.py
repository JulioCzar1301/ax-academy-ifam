import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from model import build_model

st.set_page_config(
    page_title="Sistema de Predição ML",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Sistema de Predição com Machine Learning")

@st.cache_data
def carregar_dataset():
    """
    FUTURAMENTE:
    df = pd.read_csv('dataset.csv')
    return df
    """

    np.random.seed(42)

    df = pd.DataFrame({
        "idade": np.random.randint(18, 70, 100),
        "salario": np.random.randint(1500, 12000, 100),
        "anos_experiencia": np.random.randint(0, 30, 100),
        "valor_real": np.random.randint(1000, 5000, 100)
    })

    return df


def executar_modelo(df, percentual_treino):
    features = [c for c in df.columns if c != "valor_real"]
    X = df[features]
    y = df["valor_real"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=percentual_treino / 100, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)
    y_previsto = model.predict(X_test)

    resultado = pd.DataFrame({
        "Real": y_test.values,
        "Previsto": y_previsto
    })
    resultado["Erro"] = abs(resultado["Real"] - resultado["Previsto"])

    r2 = r2_score(resultado["Real"], resultado["Previsto"])
    rmse = np.sqrt(mean_squared_error(resultado["Real"], resultado["Previsto"]))

    return resultado, r2, rmse

df = carregar_dataset()

aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "📘 Explicação",
    "📊 Dataset",
    "⚙️ Configuração",
    "🚀 Rodar Modelo",
    "📈 Resultados"
])

with aba1:

    st.header("Breve Explicação do Modelo")

    st.write("""
    Esta aplicação executa um modelo de Machine Learning
    para realizar previsões a partir de um conjunto de dados.

    Fluxo da aplicação:

    1. Carregamento do dataset
    2. Escolha da porcentagem da amostra
    3. Treinamento do modelo
    4. Geração das previsões
    5. Avaliação dos resultados

    Atualmente o sistema utiliza dados simulados.
    Futuramente você poderá conectar seu dataset real
    e seu modelo de Machine Learning.
    """)

with aba2:

    st.header("Visualização do Dataset")

    col1, col2 = st.columns(2)

    col1.metric("Linhas", df.shape[0])
    col2.metric("Colunas", df.shape[1])

    st.dataframe(
        df,
        use_container_width=True
    )

with aba3:

    st.header("Configuração")

    percentual = st.slider(
        "Percentual para Treinamento",
        min_value=50,
        max_value=90,
        value=80
    )

    modelo = st.selectbox(
        "Modelo",
        ["Ensemble Learning"]
    )

    st.info(
        f"Modelo selecionado: {modelo}\n\n"
        f"Treinamento: {percentual}%\n\n"
        "Composto por: ExtraTreesRegressor + GradientBoostingRegressor + RandomForestRegressor"
    )

with aba4:

    st.header("Execução")

    if st.button(
        "▶ Rodar Modelo",
        use_container_width=True
    ):

        resultado, r2, rmse = executar_modelo(df, percentual)

        st.session_state["resultado"] = resultado
        st.session_state["r2"] = r2
        st.session_state["rmse"] = rmse

        st.success("Modelo executado com sucesso!")

with aba5:

    st.header("Resultados")

    if "resultado" not in st.session_state:

        st.warning(
            "Execute o modelo na aba 'Rodar Modelo'."
        )

    else:

        resultado = st.session_state["resultado"]
        r2 = st.session_state["r2"]
        rmse = st.session_state["rmse"]

        st.subheader("R² e RMSE")

        c1, c2 = st.columns(2)

        c1.metric(
            "R²",
            f"{r2:.4f}"
        )

        c2.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        st.subheader("Tabela Real vs Previsto")

        st.dataframe(
            resultado,
            use_container_width=True
        )

        st.subheader("Gráfico Real vs Previsto")

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            resultado["Real"],
            resultado["Previsto"]
        )

        minimo = min(resultado["Real"])
        maximo = max(resultado["Real"])

        ax.plot(
            [minimo, maximo],
            [minimo, maximo],
            linestyle="--"
        )

        ax.set_xlabel("Valor Real")
        ax.set_ylabel("Valor Previsto")
        ax.set_title("Real x Previsto")

        st.pyplot(fig)

        st.subheader("Top 10 Melhores Previsões")

        melhores = resultado.nsmallest(
            10,
            "Erro"
        )

        st.dataframe(
            melhores,
            use_container_width=True
        )

        st.subheader("Top 10 Maiores Erros")

        maiores_erros = resultado.nlargest(
            10,
            "Erro"
        )

        st.dataframe(
            maiores_erros,
            use_container_width=True
        )
