"""UCI compatibility tests, driven through python-chess's UCI client (what lichess-bot uses).

    ../env/bin/python -m pytest -q tests/
"""
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import chess
import chess.engine
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = [sys.executable, str(ROOT / 'engine' / 'carlsen_uci.py')]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'engine'))


@pytest.fixture(scope='module')
def model_server():
    import carlsen_core
    import server
    server.ModelHandler.model = carlsen_core.CarlsenModel()
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.ModelHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()


def open_engine(server_url=None):
    env = dict(os.environ)
    env.pop('CARLSEN_SERVER', None)
    if server_url:
        env['CARLSEN_SERVER'] = server_url
    return chess.engine.SimpleEngine.popen_uci(ENGINE, env=env, timeout=60)


@pytest.mark.parametrize('mode', ['local', 'remote'])
def test_full_game_is_legal(mode, model_server):
    eng = open_engine(model_server if mode == 'remote' else None)
    assert eng.id['name'] == 'Carlsen-Maia3'
    assert {'CarlsenStyle', 'Temperature', 'StockfishFilter', 'UCI_Opponent'} <= set(eng.options)
    eng.configure({'CarlsenStyle': 'auto', 'Temperature': 0})
    board = chess.Board()
    clock = chess.engine.Limit(white_clock=180, black_clock=180, white_inc=2, black_inc=2)
    for _ in range(60):
        if board.is_game_over():
            break
        result = eng.play(board, clock)
        assert result.move in board.legal_moves
        board.push(result.move)
    eng.quit()


def test_remote_mode_starts_fast(model_server):
    t = time.time()
    eng = open_engine(model_server)
    eng.ping()                                    # isready -> readyok
    startup = time.time() - t
    eng.quit()
    assert startup < 2.0, f'remote-mode startup took {startup:.2f}s'


def test_opening_and_analysis(model_server):
    eng = open_engine(model_server)
    board = chess.Board()
    for san in ('e4', 'c5'):
        board.push_san(san)
    info = eng.analyse(board, chess.engine.Limit(time=0.1), multipv=3)
    assert len(info) == 3 and all(i['pv'][0] in board.legal_moves for i in info)
    assert info[0]['pv'][0] == chess.Move.from_uci('g1f3')        # 1.e4 c5 2.Nf3 (Carlsen-model favourite)
    assert 'wdl' in info[0] and 'score' in info[0]
    eng.quit()


def test_temperature_and_opponent(model_server):
    eng = open_engine(model_server)
    eng.configure({'Temperature': 80, 'TopP': 90})
    board = chess.Board()
    seen = set()
    for _ in range(12):
        seen.add(eng.play(board, chess.engine.Limit(time=0.1)).move)
    assert len(seen) > 1 and all(m in board.legal_moves for m in seen)
    eng.send_opponent_information(opponent=chess.engine.Opponent('someone', None, 1850, False))
    assert eng.play(board, chess.engine.Limit(time=0.1)).move in board.legal_moves
    eng.quit()


def test_ponder_and_infinite(model_server):
    eng = open_engine(model_server)
    board = chess.Board()
    with eng.analysis(board) as analysis:            # 'go infinite' then 'stop'
        analysis.get()
    result = eng.play(board, chess.engine.Limit(time=0.1), ponder=True)
    assert result.move in board.legal_moves
    eng.quit()


@pytest.mark.skipif(not Path('/usr/games/stockfish').exists(), reason='no Stockfish')
def test_stockfish_filter_local():
    eng = open_engine()
    eng.configure({'StockfishFilter': True, 'StockfishDepth': 8})
    board = chess.Board('r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3')
    move = eng.play(board, chess.engine.Limit(time=0.3)).move
    board.push(move)
    for reply in list(board.legal_moves):             # White must have no mate in one (e.g. Qxf7#)
        board.push(reply)
        assert not board.is_checkmate(), f'filter allowed {reply} mate after {move}'
        board.pop()
    eng.quit()
