# Dockerfile - Service API FastAPI (moteur de pricing)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 est requis par LightGBM (OpenMP). curl sert au healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Les dependances d'abord : cette couche est mise en cache tant que
# requirements.txt ne bouge pas.
COPY requirements.txt pyproject.toml README.md ./
COPY optimization_pricing/__init__.py ./optimization_pricing/
RUN pip install -r requirements.txt

# Puis le code et les artefacts entraines. Contrairement au churn, on ne
# reentraine PAS au build : le modele de demande depend des CSV Olist bruts
# (~125 Mo, non versionnes). L'image embarque les artefacts deja produits, ce
# qui rend le build rapide et le resultat identique a celui valide en CI.
COPY optimization_pricing/ ./optimization_pricing/
COPY api/ ./api/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Verification au build : une image sans modele utilisable doit echouer ici,
# pas en production au premier appel.
RUN python -c "from optimization_pricing.service import PricingService; \
    s = PricingService().load(); \
    assert s.model is not None, 'modele de demande absent de l image'; \
    print('Artefacts valides :', s.status())"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
