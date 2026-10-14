FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala as dependências primeiro (aproveita o cache do Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia somente o app e os dados
COPY app-Aula32.py Vehicle-DS.csv ./

EXPOSE 8501

CMD ["streamlit", "run", "app-Aula32.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
