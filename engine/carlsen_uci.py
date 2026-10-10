#!/usr/bin/env python3
"""UCI chess engine: a Carlsen-style Maia-3 model. Works in PyChess, lichess-bot and other UCI GUIs.

Two modes:
- local (default): loads engine/maia3-carlsen.onnx itself (needs onnxruntime, numpy, chess).
- remote: if CARLSEN_SERVER is set (e.g. http://127.0.0.1:8765), the already-loaded model in
  server.py does the thinking. This script then uses only the standard library and starts
  instantly, which matters on a slow free host where loading the model takes many seconds.

Options: Model (r4 = R2 + Stockfish-guided re-ranker, needs Stockfish 16; r2 = the network alone),
CarlsenStyle (auto picks classical/rapid/blitz/bullet from the game clock), OppoElo,
Temperature (hundredths; 0 = his single most likely move), TopP (percent), MultiPV,
StockfishFilter (+ StockfishPath, StockfishDepth), Threads, UCI_Opponent (sent by lichess-bot).
"""
import json
import os
import sys
import urllib.request

NAME = 'Carlsen-Maia3'
AUTHOR = 'Maia-3 (CSSLab, AGPL-3.0) fine-tuned on Carlsen games'
SPEC = {  # name: (type, default, extra)
    'Model': ('combo', 'r4', 'var r4 var r2'),
    'CarlsenStyle': ('combo', 'auto', 'var auto var any var classical var rapid var blitz var bullet'),
    'OppoElo': ('spin', 2700, 'min 600 max 3300'),
    'Temperature': ('spin', 0, 'min 0 max 200'),
    'TopP': ('spin', 100, 'min 1 max 100'),
    'MultiPV': ('spin', 1, 'min 1 max 10'),
    'StockfishFilter': ('check', False, ''),
    'StockfishPath': ('string', '/usr/games/stockfish', ''),
    'StockfishDepth': ('spin', 12, 'min 1 max 30'),
    'Threads': ('spin', 1, 'min 1 max 16'),
    'UCI_Opponent': ('string', '', ''),
}


class Engine:
    def __init__(self):
        self.opts = {k: v[1] for k, v in SPEC.items()}
        self.server = os.environ.get('CARLSEN_SERVER', '').rstrip('/')
        self.core = self.model = self.sf = None
        self.new_game()

    def new_game(self):
        self.fen, self.moves = 'startpos', []
        self.style_clock = None          # (seconds, increment) of this game's clock, for CarlsenStyle=auto
        self.pending = None              # bestmove held back during 'go infinite' / 'go ponder'

    @staticmethod
    def send(line):
        sys.stdout.write(line + '\n')
        sys.stdout.flush()

    # ---- the model, loaded only in local mode ----
    def load_local(self):
        if self.model is not None and getattr(self, 'model_kind', None) != self.opts['Model']:
            if hasattr(self.model, 'close'):
                self.model.close()
            self.model = None                       # Model option changed: load the other one
        if self.model is None:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import carlsen_core
            self.core = carlsen_core
            self.model = carlsen_core.load_model(self.opts['Threads'], self.opts['StockfishPath'], self.opts['Model'])
            self.model_kind = self.opts['Model']
        if self.opts['StockfishFilter'] and self.sf is None:
            self.sf = self.core.StockfishFilter(self.opts['StockfishPath'], int(self.opts['StockfishDepth']))
        if not self.opts['StockfishFilter'] and self.sf is not None:
            self.sf.close()
            self.sf = None

    def opponent_elo(self):
        # python-chess / lichess-bot format: "<title|none> <rating|none> <computer|human> <name>"
        parts = str(self.opts['UCI_Opponent']).split()
        if len(parts) >= 2 and parts[1].isdigit():
            return float(min(3300, max(600, int(parts[1]))))
        return float(self.opts['OppoElo'])

    def white_to_move(self):
        base_white = True if self.fen == 'startpos' else self.fen.split()[1] == 'w'
        return base_white == (len(self.moves) % 2 == 0)

    def decide(self, think_seconds):
        request = {'fen': self.fen, 'moves': self.moves, 'options': self.opts, 'style_clock': self.style_clock,
                   'oppo_elo': self.opponent_elo(), 'multipv': int(self.opts['MultiPV']), 'think_seconds': think_seconds}
        if self.server:
            try:
                req = urllib.request.Request(self.server + '/think', data=json.dumps(request).encode(),
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read())
                return out['bestmove'], out['infos']
            except Exception as exc:            # server unavailable: fall back to loading the model here
                self.send(f'info string model server unavailable ({exc}); loading model locally')
                self.server = ''
        self.load_local()
        request.pop('options')
        return self.core.think(self.model, options=self.opts, sf=self.sf, **request)

    # ---- UCI commands ----
    def cmd_position(self, args):
        if 'moves' in args:
            i = args.index('moves')
            head, self.moves = args[:i], args[i + 1:]
        else:
            head, self.moves = args, []
        self.fen = 'startpos' if head[:1] == ['startpos'] else ' '.join(head[1:7])

    def cmd_go(self, args):
        params = {}
        for key in ('wtime', 'btime', 'winc', 'binc', 'movetime', 'movestogo', 'depth', 'nodes'):
            if key in args:
                try:
                    params[key] = int(args[args.index(key) + 1])
                except (IndexError, ValueError):
                    pass
        white = self.white_to_move()
        my_time, my_inc = params.get('wtime' if white else 'btime'), params.get('winc' if white else 'binc', 0)
        if my_time is not None:                 # remember the largest clock seen: ~ the game's starting clock
            clock = (my_time / 1000.0, my_inc / 1000.0)
            if self.style_clock is None or clock[0] + 40 * clock[1] > self.style_clock[0] + 40 * self.style_clock[1]:
                self.style_clock = clock
            think = min(1.0, max(0.05, my_time / 1000.0 / 40))
        else:
            think = params.get('movetime', 500) / 1000.0
        bestmove, infos = self.decide(think)
        for line in infos:
            self.send(line)
        if 'infinite' in args or 'ponder' in args:
            self.pending = bestmove             # answer on 'stop' / 'ponderhit', as UCI requires
        else:
            self.send(f'bestmove {bestmove}')

    def release(self):
        if self.pending is not None:
            self.send(f'bestmove {self.pending}')
            self.pending = None

    def cmd_setoption(self, line):
        if ' name ' not in f' {line} ':
            return
        rest = line.split(' name ', 1)[1] if ' name ' in line else line[len('name '):]
        name, _, value = rest.partition(' value ')
        name, value = name.strip(), value.strip()
        if name not in SPEC:
            return
        kind = SPEC[name][0]
        if kind == 'check':
            self.opts[name] = value.lower() == 'true'
        elif kind == 'spin':
            try:
                self.opts[name] = int(value)
            except ValueError:
                pass
        else:
            self.opts[name] = value

    def run(self):
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            cmd, *args = line.split()
            if cmd == 'uci':
                self.send(f'id name {NAME}')
                self.send(f'id author {AUTHOR}')
                for name, (kind, default, extra) in SPEC.items():
                    d = str(default).lower() if kind == 'check' else ('<empty>' if default == '' else default)
                    self.send(f'option name {name} type {kind} default {d} {extra}'.rstrip())
                self.send('uciok')
            elif cmd == 'isready':
                if not self.server:
                    self.load_local()
                self.send('readyok')
            elif cmd == 'setoption':
                self.cmd_setoption(line[len('setoption '):])
            elif cmd == 'ucinewgame':
                self.new_game()
            elif cmd == 'position':
                self.cmd_position(args)
            elif cmd == 'go':
                self.cmd_go(args)
            elif cmd in ('stop', 'ponderhit'):
                self.release()
            elif cmd == 'quit':
                break
        if self.sf is not None:
            self.sf.close()
        if self.model is not None and hasattr(self.model, 'close'):
            self.model.close()


if __name__ == '__main__':
    Engine().run()
