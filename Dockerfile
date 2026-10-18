FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt requirements-bedrock.txt ./
ARG WITH_BEDROCK=false
RUN pip install -r requirements.txt \
 && if [ "$WITH_BEDROCK" = "true" ]; then pip install -r requirements-bedrock.txt; fi

COPY prahari ./prahari
COPY eval ./eval

# Simulate the payment ecosystem, train the models and build the graph at image build time,
# so the container starts in seconds. PRAHARI_SCALE=small builds a lighter image for quick demos.
ARG PRAHARI_SCALE=full
ENV PRAHARI_SCALE=${PRAHARI_SCALE}
RUN python -m prahari build

RUN useradd --create-home prahari && chown -R prahari /app/var
USER prahari

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=12 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"
CMD ["uvicorn", "prahari.api:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
