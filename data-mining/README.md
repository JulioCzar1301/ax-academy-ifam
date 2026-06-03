# Data Mining — Sistema de Predição ML

Aplicação Streamlit para visualizar o treinamento de um modelo Ensemble Learning.

## Requisitos

- Python 3.8+

## Como rodar

```bash
# 1. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute a aplicação
streamlit run app.py
```

A aplicação abrirá automaticamente em `http://localhost:8501`.

## Modelo

O modelo utilizado é um **VotingRegressor** composto por:

- `ExtraTreesRegressor`
- `GradientBoostingRegressor`
- `RandomForestRegressor`

Os hiperparâmetros foram definidos via busca em grid e estão configurados em `model.py`.
