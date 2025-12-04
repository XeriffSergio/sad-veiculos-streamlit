import streamlit as st
import pandas as pd
import h2o
from h2o.frame import H2OFrame

# =========================
#  CONFIGURAÇÃO BÁSICA
# =========================

st.set_page_config(
    page_title="Previsão de Preço e Desvalorização de Veículos",
    layout="wide"
)

@st.cache_resource
def init_h2o():
    # Inicia o cluster H2O apenas uma vez
    h2o.init(max_mem_size="2G", nthreads=-1)
    return True

@st.cache_resource
def load_mojo_model(mojo_path: str):
    """
    Carrega um modelo MOJO a partir de um ficheiro .zip.
    Tem de ser o mesmo esquema de colunas usado no treino.
    """
    init_h2o()
    model = h2o.import_mojo(mojo_path)
    return model


def predict_with_mojo(model, row_dict: dict):
    """
    Recebe um dicionário com 1 linha de dados,
    converte para H2OFrame e devolve a previsão em float.
    """
    df = pd.DataFrame([row_dict])
    hf = H2OFrame(df)
    pred = model.predict(hf).as_data_frame()
    # Normalmente a coluna de previsão chama-se 'predict'
    return float(pred.iloc[0, 0])


# =========================
#  SIDEBAR – ESCOLHA DO SAD E MODELO
# =========================

st.sidebar.title("Configuração")

sad_escolhido = st.sidebar.selectbox(
    "Escolhe o SAD:",
    ("SAD 1 – Preço", "SAD 2 – Desvalorização")
)

if sad_escolhido == "SAD 1 – Preço":
    st.sidebar.markdown("**Target:** preço do veículo")
    modelo_label = st.sidebar.selectbox(
        "Modelo do SAD 1",
        (
            "GBM – Modelo 1 (principal)",
            "GBM – Modelo 2",
            "Deep Learning – Modelo 3"
        )
    )

    # 👉 Ajusta estes paths para os teus ficheiros reais
    mapa_modelos = {
        "GBM – Modelo 1 (principal)": "models/gbm_model-2-sad-1",
        "GBM – Modelo 2": "models/gbm_model_2_sad_2",
        "Deep Learning – Modelo 3": "models/deeplearning_model-3-sad-1",
    }

    mojo_path = mapa_modelos[modelo_label]
    model = load_mojo_model(mojo_path)

else:
    st.sidebar.markdown("**Target:** % de desvalorização do veículo")
    modelo_label = st.sidebar.selectbox(
        "Modelo do SAD 2",
        (
            "GLM – Modelo 1",
            "GBM – Modelo 2",
            "Stacked Ensemble – Modelo 3"
        )
    )

    
    mapa_modelos = {
        "GLM – Modelo 1": "models/glm_model_1_sad_2",
        "GBM – Modelo 2": "models/gbm_model_2_sad_2",
        "Stacked Ensemble – Modelo 3": "models/stackedensemble_sad2",
    }

    mojo_path = mapa_modelos[modelo_label]
    model = load_mojo_model(mojo_path)


st.sidebar.success(f"Modelo carregado: {modelo_label}")


# =========================
#  FORMULÁRIO DE INPUTS
# =========================

st.title("📊 Previsão de Veículos com H2O + Streamlit")

if sad_escolhido == "SAD 1 – Preço":
    st.subheader("SAD 1 – Previsão do preço de venda (USD)")

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input("Ano do veículo (year)", min_value=1980, max_value=2025, value=2015)
        age = st.number_input("Idade (age, em anos)", min_value=0, max_value=50, value=2025 - year)
        odometer = st.number_input("Quilometragem (odometer)", min_value=0, max_value=500000, value=100000, step=1000)

    with col2:
        manufacturer = st.text_input("Marca (manufacturer)", value="ford")
        model_name = st.text_input("Modelo (model)", value="focus")
        condition = st.selectbox("Condição (condition)", ["new", "like new", "excellent", "good", "fair", "salvage"])
        cylinders = st.selectbox("Cilindros (cylinders)", ["4 cylinders", "6 cylinders", "8 cylinders", "3 cylinders", "5 cylinders", "10 cylinders", "12 cylinders"])

    with col3:
        fuel = st.selectbox("Combustível (fuel)", ["gas", "diesel", "electric", "hybrid", "other"])
        transmission = st.selectbox("Transmissão (transmission)", ["automatic", "manual", "other"])
        drive = st.selectbox("Tração (drive)", ["fwd", "rwd", "4wd", "other"])
        type_ = st.selectbox("Tipo (type)", ["sedan", "SUV", "truck", "coupe", "hatchback", "wagon", "convertible", "van", "other"])
        state = st.text_input("Estado (state, código)", value="ca")
        region = st.text_input("Região (region)", value="san francisco")
        paint_color = st.text_input("Cor (paint_color)", value="black")

    # Campos extra usados em alguns modelos
    segment_avg_price = st.number_input("Preço médio do segmento (segment_avg_price)", min_value=0, max_value=200000, value=20000, step=500)
    age_bin = st.selectbox("Classe etária (age_bin)", ["0-5", "6-10", "11-15", "16-20", "20+"])
    title_status = st.selectbox("Título (title_status)", ["clean", "salvage", "rebuilt", "lien", "missing"])

    if st.button("Prever preço"):
        # Dicionário tem de ter TODAS as colunas que o modelo espera
        row = {
            "year": year,
            "manufacturer": manufacturer,
            "model": model_name,
            "condition": condition,
            "cylinders": cylinders,
            "fuel": fuel,
            "odometer": odometer,
            "title_status": title_status,
            "transmission": transmission,
            "drive": drive,
            "type": type_,
            "paint_color": paint_color,
            "state": state,
            "region": region,
            "age": age,
            "age_bin": age_bin,
            "segment_avg_price": segment_avg_price,
            # NÃO incluir 'price' nem 'desvalorizacao_pct' aqui porque 'price' é o target do SAD1
        }

        preco_previsto = predict_with_mojo(model, row)
        st.success(f"💰 Preço previsto: **{preco_previsto:,.0f} USD**")


else:
    st.subheader("SAD 2 – Previsão da percentagem de desvalorização")

    col1, col2, col3 = st.columns(3)

    with col1:
        price = st.number_input("Preço atual (price)", min_value=0, max_value=200000, value=20000, step=500)
        year = st.number_input("Ano do veículo (year)", min_value=1980, max_value=2025, value=2015)
        age = st.number_input("Idade (age, em anos)", min_value=0, max_value=50, value=2025 - year)
        odometer = st.number_input("Quilometragem (odometer)", min_value=0, max_value=500000, value=100000, step=1000)

    with col2:
        manufacturer = st.text_input("Marca (manufacturer)", value="ford")
        model_name = st.text_input("Modelo (model)", value="focus")
        condition = st.selectbox("Condição (condition)", ["new", "like new", "excellent", "good", "fair", "salvage"])
        cylinders = st.selectbox("Cilindros (cylinders)", ["4 cylinders", "6 cylinders", "8 cylinders", "3 cylinders", "5 cylinders", "10 cylinders", "12 cylinders"])
        fuel = st.selectbox("Combustível (fuel)", ["gas", "diesel", "electric", "hybrid", "other"])

    with col3:
        transmission = st.selectbox("Transmissão (transmission)", ["automatic", "manual", "other"])
        drive = st.selectbox("Tração (drive)", ["fwd", "rwd", "4wd", "other"])
        type_ = st.selectbox("Tipo (type)", ["sedan", "SUV", "truck", "coupe", "hatchback", "wagon", "convertible", "van", "other"])
        state = st.text_input("Estado (state, código)", value="ca")
        region = st.text_input("Região (region)", value="san francisco")
        paint_color = st.text_input("Cor (paint_color)", value="black")

    segment_avg_price = st.number_input("Preço médio do segmento (segment_avg_price)", min_value=0, max_value=200000, value=20000, step=500)
    age_bin = st.selectbox("Classe etária (age_bin)", ["0-5", "6-10", "11-15", "16-20", "20+"])
    title_status = st.selectbox("Título (title_status)", ["clean", "salvage", "rebuilt", "lien", "missing"])

    if st.button("Prever desvalorização (%)"):
        # Este SAD tem como target 'desvalorizacao_pct', por isso aqui price entra como feature
        row = {
            "price": price,
            "year": year,
            "manufacturer": manufacturer,
            "model": model_name,
            "condition": condition,
            "cylinders": cylinders,
            "fuel": fuel,
            "odometer": odometer,
            "title_status": title_status,
            "transmission": transmission,
            "drive": drive,
            "type": type_,
            "paint_color": paint_color,
            "state": state,
            "region": region,
            "age": age,
            "age_bin": age_bin,
            "segment_avg_price": segment_avg_price,
            # Aqui o target 'desvalorizacao_pct' NÃO é passado, é o que queremos prever
        }

        desval_pct = predict_with_mojo(model, row)
        st.success(f"📉 Desvalorização prevista: **{desval_pct:.2f}%**")

        valor_inicial = price / (1 - desval_pct / 100) if desval_pct < 100 else None
        if valor_inicial:
            st.write(f"Isto corresponde a um valor original aproximado de **{valor_inicial:,.0f} USD**.")


st.caption(
    "⚠️ Os resultados são estimativas baseadas nos modelos H2O treinados (SAD 1 e SAD 2). "
    "Usar apenas para fins académicos/trabalho prático."
)
