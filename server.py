"""Render entry point: model server + lichess-bot supervisor + public status page.

- Loads the Carlsen model once and answers POST /think on 127.0.0.1 (private; the UCI engine
  started by lichess-bot for each game calls it, so games start instantly even on 0.1 CPU).
- Runs lichess-bot and restarts it if it exits (bad token, network trouble, ...), with backoff.
- BOT_RUN_SECONDS (optional): after that long, stop gracefully (no new games; games in progress
  finish when lichess-bot's quit_after_all_games_finish is on), then exit. For hosts with a session
  limit, such as Kaggle's 12 hours. BOT_GRACE_SECONDS caps the wait for running games (default 900).
- Serves a status page, /healthz (service up) and /status (bot state, JSON) on $PORT. Render's free web services sleep after 15 minutes
  without inbound traffic, so the service pings its own public URL every 10 minutes
  (KEEPALIVE=0 turns this off). An external pinger on /healthz is a good backup.
"""
import html
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'engine'))
import carlsen_core  # noqa: E402

PORT = int(os.environ.get('PORT', '10000'))
MODEL_PORT = int(os.environ.get('CARLSEN_MODEL_PORT', '8765'))
LICHESS_BOT_DIR = Path(os.environ.get('LICHESS_BOT_DIR', '/lichess-bot'))
STATE = {'started': time.time(), 'bot_starts': 0, 'bot_running': False, 'moves': 0, 'last_log': []}


def log(msg):
    print(f'[server] {msg}', flush=True)


class ModelHandler(BaseHTTPRequestHandler):
    model = None
    sf = None
    sf_lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != '/think':
            self.send_error(404)
            return
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        sf = None
        if req['options'].get('StockfishFilter'):
            with self.sf_lock:
                if ModelHandler.sf is None:
                    ModelHandler.sf = carlsen_core.StockfishFilter(req['options'].get('StockfishPath', '/usr/games/stockfish'),
                                                                   int(req['options'].get('StockfishDepth', 12)))
                sf = ModelHandler.sf
        try:
            bestmove, infos = carlsen_core.think(self.model, req['fen'], req['moves'], req['options'],
                                                 req.get('style_clock'), req.get('oppo_elo'), req.get('multipv', 1),
                                                 sf, req.get('think_seconds', 0.3))
            STATE['moves'] += 1
            body, code = json.dumps({'bestmove': bestmove, 'infos': infos}).encode(), 200
        except Exception as exc:
            body, code = json.dumps({'error': repr(exc)}).encode(), 500
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class PublicHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, ctype='text/plain; charset=utf-8'):
        data = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self.path.startswith('/healthz'):           # the web service + model are up (Render's health check)
            self.reply(200, 'ok')
            return
        if self.path.startswith('/status'):            # is the Lichess bot itself connected and running?
            body = json.dumps({'lichess_bot_running': STATE['bot_running'], 'bot_starts': STATE['bot_starts'],
                               'moves_played': STATE['moves'], 'uptime_s': int(time.time() - STATE['started'])})
            self.reply(200, body, 'application/json')
            return
        if self.path != '/':
            self.reply(404, 'not found')
            return
        bot = os.environ.get('LICHESS_BOT_NAME', '')
        up = int(time.time() - STATE['started'])
        logs = html.escape('\n'.join(STATE['last_log'][-30:]))
        link = f'<p>Play it: <a href="https://lichess.org/@/{html.escape(bot)}">lichess.org/@/{html.escape(bot)}</a></p>' if bot else ''
        page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carlsen-style Maia bot</title><style>body{{font:16px system-ui;max-width:760px;margin:2rem auto;padding:0 16px}}
pre{{background:#f4f4f4;padding:12px;overflow:auto;font-size:12px}}@media (prefers-color-scheme:dark){{body{{background:#111;color:#ddd}}pre{{background:#222}}a{{color:#8ab4f8}}}}</style></head>
<body><h1>Carlsen-style Maia-3 Lichess bot</h1>
<p>Status: <b>{'online' if STATE['bot_running'] else 'starting / restarting'}</b> · up {up // 3600}h {up % 3600 // 60}m · moves played since start: {STATE['moves']}</p>
{link}<p>A Maia-3 model fine-tuned on Magnus Carlsen's published games to imitate his move choices. It is an AI imitation, not Magnus Carlsen.
Source code (AGPL-3.0): see the repository linked in the bot's Lichess profile.</p>
<h3>Recent log</h3><pre>{logs}</pre></body></html>"""
        self.reply(200, page, 'text/html; charset=utf-8')


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()


def supervise_bot():
    delay = 5
    run_for = float(os.environ.get('BOT_RUN_SECONDS', '0'))
    deadline = STATE['started'] + run_for if run_for > 0 else None
    grace = float(os.environ.get('BOT_GRACE_SECONDS', '900'))
    while True:
        if deadline and time.time() >= deadline:
            log('run time is up; not restarting lichess-bot')
            return
        if not os.environ.get('LICHESS_BOT_TOKEN'):
            log('LICHESS_BOT_TOKEN is not set; waiting (set it in the Render dashboard).')
            STATE['last_log'].append('LICHESS_BOT_TOKEN is not set.')
            time.sleep(60)
            continue
        env = dict(os.environ, CARLSEN_SERVER=f'http://127.0.0.1:{MODEL_PORT}')
        STATE['bot_starts'] += 1
        started = time.time()
        log(f'starting lichess-bot (start #{STATE["bot_starts"]})')
        proc = subprocess.Popen([sys.executable, 'lichess-bot.py', '--disable_auto_logging'], cwd=LICHESS_BOT_DIR,
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        STATE['bot_running'] = True
        if deadline:
            def stop_at_deadline(p=proc):
                time.sleep(max(0.0, deadline - time.time()))
                if p.poll() is None:
                    log('run time is up: asking lichess-bot to finish its games and quit')
                    p.send_signal(signal.SIGINT)
                    time.sleep(grace)
                    if p.poll() is None:
                        log('grace period over: stopping lichess-bot')
                        p.send_signal(signal.SIGINT)      # a second Ctrl-C quits immediately
                        time.sleep(15)
                        if p.poll() is None:
                            p.kill()
            threading.Thread(target=stop_at_deadline, daemon=True).start()
        for line in proc.stdout:
            print(line, end='', flush=True)
            STATE['last_log'] = (STATE['last_log'] + [line.rstrip()])[-200:]
        code = proc.wait()
        STATE['bot_running'] = False
        if deadline and time.time() >= deadline:
            log(f'lichess-bot exited with code {code}; run time is up, not restarting')
            return
        delay = 5 if time.time() - started > 600 else min(600, delay * 2)
        log(f'lichess-bot exited with code {code}; restarting in {delay}s')
        time.sleep(delay)


def keepalive():
    url = os.environ.get('RENDER_EXTERNAL_URL', '').rstrip('/')
    if not url or os.environ.get('KEEPALIVE', '1') == '0':
        return
    while True:
        time.sleep(600)
        try:
            urllib.request.urlopen(url + '/healthz', timeout=30).read()
        except Exception as exc:
            log(f'keep-alive ping failed: {exc}')


def main():
    serve(ThreadingHTTPServer(('0.0.0.0', PORT), PublicHandler))         # Render needs the port open quickly
    log(f'status page on :{PORT}; loading model ...')
    t = time.time()
    ModelHandler.model = carlsen_core.load_model(int(os.environ.get('CARLSEN_THREADS', '1')),
                                                 os.environ.get('STOCKFISH', '/usr/games/stockfish'),
                                                 os.environ.get('CARLSEN_MODEL', 'r4'))
    log(f'model loaded in {time.time() - t:.1f}s: {carlsen_core.model_name(ModelHandler.model)}')
    serve(ThreadingHTTPServer(('127.0.0.1', MODEL_PORT), ModelHandler))
    threading.Thread(target=keepalive, daemon=True).start()
    supervise_bot()


if __name__ == '__main__':
    main()
