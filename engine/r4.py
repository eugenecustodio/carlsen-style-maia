"""R4 at play time: R2 proposes its top-12 moves, Stockfish 16 scores them, the re-ranker decides.

Must reproduce annotate.py exactly (same candidates, same Stockfish settings: depth 10, Threads 1,
Hash 16, hash cleared per position, MultiPV over the candidates), otherwise the re-ranker sees features
it was not trained on. Self-contained: needs onnxruntime, numpy, python-chess and a Stockfish 16 binary
(bench 2593605).
"""
import threading
from pathlib import Path

import chess
import chess.engine
import numpy as np

K = 12
DEPTH = 10
HISTORY = 8
MOVES = [chess.square_name(a) + chess.square_name(b) for a in chess.SQUARES for b in chess.SQUARES] + [
    f'{a}7{b}8{piece}' for a in 'abcdefgh' for b in 'abcdefgh' for piece in 'qrbn']
MOVE_INDEX = {uci: i for i, uci in enumerate(MOVES)}
STYLE_CODE = {'classical': -1.0, 'rapid': -2.0, 'blitz': -3.0, 'bullet': -4.0, 'any': -5.0}
STYLE_TC = {'classical': 0, 'rapid': 1, 'blitz': 2, 'bullet': 3, 'any': 5}      # data tc codes used in training


def tokens(board):
    if board.turn == chess.BLACK:
        board = board.mirror()
    planes = np.zeros((64, 12), np.float32)
    for square, piece in board.piece_map().items():
        planes[square, piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)] = 1
    return planes


def encode(board):
    board = board.copy()
    positions = [tokens(board)]
    while len(positions) < HISTORY and board.move_stack:
        board.pop()
        positions.append(tokens(board))
    positions += [positions[-1]] * (HISTORY - len(positions))
    return np.concatenate(positions[::-1], axis=1)[None]


def move_index(board, move):
    if board.turn == chess.BLACK:
        move = chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), move.promotion)
    return MOVE_INDEX[move.uci()]


def files_present(directory):
    d = Path(directory)
    return (d / 'r4-prior.onnx').exists() and (d / 'r4-reranker.onnx').exists()


class R4:
    def __init__(self, directory, stockfish='/usr/games/stockfish', threads=1):
        import onnxruntime as ort
        d = Path(directory)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        self.prior = ort.InferenceSession(str(d / 'r4-prior.onnx'), opts, providers=['CPUExecutionProvider'])
        self.reranker = ort.InferenceSession(str(d / 'r4-reranker.onnx'), opts, providers=['CPUExecutionProvider'])
        self.sf = chess.engine.SimpleEngine.popen_uci(stockfish, timeout=None)
        self.sf.configure({'Threads': 1, 'Hash': 16})
        self.lock = threading.Lock()

    def close(self):
        try:
            self.sf.quit()
        except Exception:
            pass

    def features(self, board, style='any', oppo_elo=2800.0):
        """Everything the re-ranker needs, computed exactly as in training."""
        legal_moves = list(board.legal_moves)
        idx = np.array([move_index(board, m) for m in legal_moves], np.int64)
        logits, value, hidden = self.prior.run(None, {
            'tokens': encode(board), 'elo_self': np.array([STYLE_CODE.get(style, -5.0)], np.float32),
            'elo_oppo': np.array([float(oppo_elo)], np.float32)})
        order = np.argsort(-logits[0, idx], kind='stable')[:K]
        cand = np.full(K, -1, np.int64)
        cand[:len(order)] = idx[order]
        cand_moves = [legal_moves[i] for i in order]
        e = np.zeros(K, np.float32)
        cp = np.zeros(K, np.float32)
        rank = np.full(K, -1, np.int64)
        searched = np.zeros(K, np.float32)
        infos = self.sf.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=len(cand_moves),
                                root_moves=cand_moves, game=object())               # new game -> hash cleared
        pos = {m: i for i, m in enumerate(cand_moves)}
        for r, info in enumerate(infos):
            if not info.get('pv') or info['pv'][0] not in pos:
                continue
            i = pos[info['pv'][0]]
            score = info['score'].pov(board.turn)
            e[i] = score.wdl(model='lichess').expectation()
            cp[i] = max(-10000, min(10000, score.score(mate_score=10000)))
            rank[i] = r
            searched[i] = 1.0
        legal = np.zeros((1, len(MOVES)), np.float32)
        legal[0, idx] = 1.0
        return {'logits': logits.astype(np.float32), 'legal': legal, 'hidden': hidden.astype(np.float32),
                'cand': cand[None], 'valid': (cand >= 0).astype(np.float32)[None], 'sf_e': e[None], 'sf_cp': cp[None],
                'sf_rank': rank[None], 'searched': searched[None],
                'tc': np.array([STYLE_TC.get(style, 5)], np.int64)}, value[0], legal_moves, idx

    def predict(self, board, style='any', oppo_elo=2800.0):
        """Legal moves with R4 probabilities (most likely first) and (win, draw, loss) for the side to move."""
        if not any(board.legal_moves):
            return [], (0.0, 1.0, 0.0)
        with self.lock:
            feed, value, legal_moves, idx = self.features(board, style, oppo_elo)
            delta = self.reranker.run(None, feed)[0][0]
        final = feed['logits'][0].astype(np.float64).copy()
        for c, dlt in zip(feed['cand'][0], delta):
            if c >= 0:
                final[c] += dlt
        scores = final[idx]
        p = np.exp(scores - scores.max())
        p /= p.sum()
        v = np.exp(value - value.max())
        loss, draw, win = (v / v.sum()).tolist()
        order = np.argsort(-p, kind='stable')
        return [(legal_moves[i], float(p[i])) for i in order], (win, draw, loss)
