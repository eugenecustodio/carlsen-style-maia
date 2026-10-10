# Carlsen-style Maia-3 Lichess bot (Render free web service, or any Docker host).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LICHESS_BOT_DIR=/lichess-bot

# lichess-bot (AGPL-3.0), pinned to a tested commit
ARG LICHESS_BOT_COMMIT=df7e730de58cc3ef2f1415a0dc2eeda842d39167
RUN python3 -c "import io, tarfile, urllib.request; \
url = 'https://github.com/lichess-bot-devs/lichess-bot/archive/${LICHESS_BOT_COMMIT}.tar.gz'; \
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())).extractall('/tmp/lb')" \
 && mv /tmp/lb/lichess-bot-* /lichess-bot && rm -rf /tmp/lb

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -r /lichess-bot/requirements.txt

COPY . .
RUN python3 make_config.py /lichess-bot/config.yml.default /lichess-bot/config.yml \
 && chmod +x engine/carlsen_uci.py \
 && python3 -c "import sys; sys.path.insert(0, 'engine'); import carlsen_core; carlsen_core.CarlsenModel(); print('model OK')"

CMD ["python3", "server.py"]
