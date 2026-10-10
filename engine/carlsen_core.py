"""Carlsen-style Maia-3 move model (ONNX) and the move-choice logic shared by every front end.

The model is Maia3-23M fine-tuned on Magnus Carlsen's games (run R2: 60.2% top-1 on held-out
games). Its Elo input takes either a rating or a Carlsen style code:
-1 classical, -2 rapid, -3 blitz, -4 bullet, -5 any (average Carlsen).

This is an imitation of Carlsen's move choices, not Magnus Carlsen.
"""
import math
import os
import random
import threading
from pathlib import Path

import chess
import numpy as np

HERE = Path(__file__).resolve().parent
# R4's prior (r4-prior.onnx) is R2's network with its hidden state exposed; once present, R2's own file
# (maia3-carlsen.onnx) is not needed, and the network-only fallback ("r2" mode) loads the prior.
MODEL_PATH = HERE / 'r4-prior.onnx' if (HERE / 'r4-prior.onnx').exists() else HERE / 'maia3-carlsen.onnx'
R4_DIR = Path(os.environ.get('CARLSEN_R4_DIR', HERE))      # r4-prior.onnx + r4-reranker.onnx
HISTORY = 8
# Every from-to square pair, then white promotions from the seventh rank. The board is
# mirrored when Black moves, so Black's moves use the same entries.
MOVES = [chess.square_name(a) + chess.square_name(b) for a in chess.SQUARES for b in chess.SQUARES] + [
    f'{a}7{b}8{piece}' for a in 'abcdefgh' for b in 'abcdefgh' for piece in 'qrbn']
MOVE_INDEX = {uci: i for i, uci in enumerate(MOVES)}
STYLE_CODES = {'classical': -1.0, 'rapid': -2.0, 'blitz': -3.0, 'bullet': -4.0, 'any': -5.0}
STYLES = ('auto', 'any', 'classical', 'rapid', 'blitz', 'bullet')


def tokens(board):
    """One-hot pieces per square, from the side to move's point of view."""
    if board.turn == chess.BLACK:
        board = board.mirror()
    planes = np.zeros((64, 12), np.float32)
    for square, piece in board.piece_map().items():
        planes[square, piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)] = 1
    return planes


def encode(board):
    """(1, 64, 96) input: the current position and up to seven before it, oldest first."""
    board = board.copy()
    positions = [tokens(board)]
    while len(positions) < HISTORY and board.move_stack:
        board.pop()
        positions.append(tokens(board))
    positions += [positions[-1]] * (HISTORY - len(positions))   # short history: pad with the earliest
    return np.concatenate(positions[::-1], axis=1)[None]


def move_index(board, move):
    if board.turn == chess.BLACK:
        move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
    return MOVE_INDEX[move.uci()]


def style_for_clock(seconds, increment):
    """Lichess time-control category from a game's clock (estimated duration = base + 40 x increment)."""
    estimate = seconds + 40 * increment
    if estimate < 180:
        return 'bullet'
    if estimate < 480:
        return 'blitz'
    if estimate < 1500:
        return 'rapid'
    return 'classical'


def expected_to_cp(expected):
    """Expected score -> centipawns on Lichess's win-chance curve (for engine 'score' output)."""
    e = min(max(expected, 1e-4), 1 - 1e-4)
    return int(round(max(-1500.0, min(1500.0, math.log(e / (1 - e)) / 0.00368208))))


class CarlsenModel:
    def __init__(self, path=MODEL_PATH, threads=1):
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        if [i.name for i in self.session.get_inputs()] != ['tokens', 'elo_self', 'elo_oppo']:
            raise RuntimeError(f'{path} is not a Carlsen/Maia-3 model')
        self.lock = threading.Lock()

    def predict(self, board, style='any', oppo_elo=2700.0):
        """Legal moves with probabilities (most likely first) and (win, draw, loss) for the side to move."""
        legal = list(board.legal_moves)
        if not legal:
            return [], (0.0, 1.0, 0.0)
        elo_self = STYLE_CODES.get(style, -5.0)
        feed = {'tokens': encode(board), 'elo_self': np.array([elo_self], np.float32),
                'elo_oppo': np.array([float(oppo_elo)], np.float32)}
        with self.lock:
            logits, value = self.session.run(['logits_move', 'logits_value'], feed)
        scores = np.array([logits[0, move_index(board, m)] for m in legal], np.float64)
        p = np.exp(scores - scores.max())
        p /= p.sum()
        v = np.exp(value[0] - value[0].max())
        loss, draw, win = (v / v.sum()).tolist()                     # value head order: [loss, draw, win]
        order = np.argsort(-p, kind='stable')
        return [(legal[i], float(p[i])) for i in order], (win, draw, loss)


def load_model(threads=1, stockfish='/usr/games/stockfish', prefer='r4'):
    """R4 (R2 + Stockfish-guided re-ranker) when its files and Stockfish are available, else R2."""
    import r4
    if prefer == 'r4' and r4.files_present(R4_DIR) and stockfish and Path(stockfish).exists():
        return r4.R4(R4_DIR, stockfish, threads)
    return CarlsenModel(threads=threads)


def model_name(model):
    return 'R4 (R2 + Stockfish re-ranker)' if type(model).__name__ == 'R4' else 'R2'


def sample(ranked, temperature=0.0, top_p=1.0, rng=random):
    """Temperature 0 = the single most likely Carlsen move; higher = more varied, human-like choices."""
    if temperature <= 0 or len(ranked) == 1:
        return ranked[0][0]
    logits = np.log(np.array([p for _, p in ranked]) + 1e-12) / temperature
    p = np.exp(logits - logits.max())
    p /= p.sum()
    keep = np.cumsum(p) - p < top_p                                   # nucleus: smallest set covering top_p
    p = np.where(keep, p, 0.0)
    p /= p.sum()
    return ranked[int(rng.choices(range(len(ranked)), weights=p.tolist())[0])][0]


class StockfishFilter:
    """Optional safety net (as in ChessLens): veto Carlsen-model moves that lose more than `max_drop`
    expected score against Stockfish's best. Inference only; the model was never trained on Stockfish."""

    def __init__(self, path='/usr/games/stockfish', depth=12, max_drop=0.04, candidates=5, min_policy=0.03):
        import chess.engine
        self.engine = chess.engine.SimpleEngine.popen_uci(path, timeout=30)
        self.engine.configure({'Threads': 1})
        self.depth, self.max_drop, self.candidates, self.min_policy = depth, max_drop, candidates, min_policy

    def pick(self, board, ranked, seconds):
        import chess.engine
        cands = [m for m, p in ranked[:self.candidates] if p >= self.min_policy] or [ranked[0][0]]
        limit = chess.engine.Limit(depth=self.depth, time=max(0.05, seconds))
        best = self.engine.analyse(board, limit)
        best_move = best['pv'][0]
        best_e = best['score'].pov(board.turn).wdl(model='lichess').expectation()
        infos = self.engine.analyse(board, limit, multipv=len(cands), root_moves=cands)
        values = {i['pv'][0]: i['score'].pov(board.turn).wdl(model='lichess').expectation() for i in infos}
        for move in cands:
            if move == best_move or (move in values and best_e - values[move] <= self.max_drop):
                return move, True
        return best_move, False

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


def think(model, fen, moves, options, style_clock=None, oppo_elo=None, multipv=1, sf=None, think_seconds=0.3):
    """One move decision. Returns (bestmove_uci, info_lines). Pure function of its inputs plus the model."""
    board = chess.Board(fen) if fen != 'startpos' else chess.Board()
    for u in moves:
        board.push_uci(u)
    style = options.get('CarlsenStyle', 'auto')
    if style == 'auto':
        style = style_for_clock(*style_clock) if style_clock else 'any'
    if oppo_elo is None:
        oppo_elo = float(options.get('OppoElo', 2700))
    ranked, (win, draw, loss) = model.predict(board, style, oppo_elo)
    if not ranked:
        return '0000', []
    cp = expected_to_cp(win + 0.5 * draw)
    wdl = [int(round(1000 * x)) for x in (win, draw, loss)]
    wdl[1] = 1000 - wdl[0] - wdl[2]
    infos = [f'info string {model_name(model)}, Carlsen style {style}, opponent Elo {int(oppo_elo)}']
    for k, (move, p) in enumerate(ranked[:max(1, multipv)], 1):
        infos.append(f'info depth 1 seldepth 1 multipv {k} score cp {cp} wdl {wdl[0]} {wdl[1]} {wdl[2]} '
                     f'nodes 1 nps 1 time 1 pv {move.uci()} string p={p:.3f}')
    if sf is not None:
        move, kept = sf.pick(board, ranked, think_seconds)
        infos.append(f"info string StockfishFilter: {'kept Carlsen-model move' if kept else 'replaced by Stockfish'}")
    else:
        move = sample(ranked, float(options.get('Temperature', 0)) / 100.0, float(options.get('TopP', 100)) / 100.0)
    return move.uci(), infos
