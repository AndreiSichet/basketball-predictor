# Build from the REPOSITORY ROOT, not from this directory:
#
#   docker build -f inference-service/Dockerfile -t basketball-inference .
#
# The context has to be the root because this service deliberately imports
# from two sibling folders rather than duplicating feature logic:
# ml-training/{common,live_features}.py and three scripts in
# data-pipeline/preprocessing/. live_features.py locates them with
# Path(__file__).resolve().parents[1], so the sibling layout has to survive
# intact inside the image - /app is what parents[1] resolves to from both
# /app/inference-service/app.py and /app/ml-training/live_features.py.

FROM python:3.14-slim

WORKDIR /app

# Dependencies before code, so editing app.py does not reinstall xgboost.
COPY inference-service/requirements.txt ./inference-service/requirements.txt
RUN pip install --no-cache-dir -r inference-service/requirements.txt

# The sibling layout, reproduced exactly. Only the files this service
# actually reaches for - not the whole pipeline or the training scripts.
COPY inference-service/ ./inference-service/
COPY ml-training/common.py ml-training/live_features.py ./ml-training/
COPY ml-training/models/ ./ml-training/models/
COPY data-pipeline/preprocessing/build_elo_ratings.py \
     data-pipeline/preprocessing/build_rest_days.py \
     data-pipeline/preprocessing/build_rolling_features.py \
     ./data-pipeline/preprocessing/

# The history table the service serves from. Baked into the image, so a
# pipeline rerun means a rebuild - see the note in CLAUDE.md section 11.
COPY data-pipeline/data/processed/games_final.csv ./data-pipeline/data/processed/

WORKDIR /app/inference-service

EXPOSE 8000

# --host 0.0.0.0 is load-bearing. uvicorn defaults to 127.0.0.1, which
# inside a container is the container's own loopback: the port would be
# published and nothing would ever answer on it. It works locally today
# only because dev calls are made from the same machine.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
