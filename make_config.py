"""Build lichess-bot's config.yml from its own config.yml.default plus this bot's settings.

    python make_config.py /lichess-bot/config.yml.default /lichess-bot/config.yml [--engine-dir DIR] [--fast]

Default profile: a slow free host (Render free: 0.1 CPU) -> 3+0 and slower, one game at a time.
--fast: a full CPU core or more (Kaggle, a laptop) -> bullet too, two games at once (one slot kept for
human challengers), and finish running games before quitting (for hosts with a session time limit).

The token is NOT written here: lichess-bot reads it from the LICHESS_BOT_TOKEN environment variable.
"""
import argparse

import yaml

ENGINE_DIR = '/app/engine'

OVERRIDES = {
    'token': 'set-by-LICHESS_BOT_TOKEN-environment-variable',
    'engine': {
        'dir': ENGINE_DIR,
        'name': 'carlsen_uci.py',
        'working_dir': ENGINE_DIR,
        'interpreter': 'python3',
        'interpreter_options': [],
        'protocol': 'uci',
        'ponder': False,                      # the engine answers instantly; nothing to ponder
        'polyglot': {'enabled': False},
        'draw_or_resign': {'resign_enabled': False, 'offer_draw_enabled': False},
        'online_moves': {'chessdb_book': {'enabled': False}, 'lichess_cloud_analysis': {'enabled': False},
                         'online_egtb': {'enabled': False}},
        'lichess_bot_tbs': {'syzygy': {'enabled': False}, 'gaviota': {'enabled': False}},
        # Only options this engine declares (python-chess rejects unknown ones).
        'uci_options': {'CarlsenStyle': 'auto', 'Temperature': 0, 'TopP': 100, 'MultiPV': 1,
                        'StockfishFilter': False},
    },
    'move_overhead': 3000,                    # free host = slow CPU: keep a 3 s safety margin on the clock
    'abort_time': 30,
    'challenge': {
        'concurrency': 1,                     # one game at a time on a 0.1-CPU free instance
        'accept_bot': True,
        'max_increment': 20, 'min_increment': 0,
        'max_base': 1800, 'min_base': 180,    # 3+0 and slower: about 1 s per move on the free tier
        'variants': ['standard'],
        'time_controls': ['blitz', 'rapid', 'classical'],
        'modes': ['casual', 'rated'],
    },
    'greeting': {
        'hello': "Hi! I'm {me}, a Maia-3 model fine-tuned on Magnus Carlsen's games. I imitate his move choices; I am not Magnus Carlsen. Good luck!",
        'goodbye': 'Good game!',
        'hello_spectators': "Hi! I'm {me}, an AI imitating Magnus Carlsen's move choices (not Magnus himself).",
        'goodbye_spectators': 'Thanks for watching!',
    },
    'matchmaking': {'allow_matchmaking': False},
}


def merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


FAST = {
    'move_overhead': 1500,
    'quit_after_all_games_finish': True,      # on shutdown, finish games in progress first
    'challenge': {'concurrency': 2, 'min_base': 60,
                  'time_controls': ['bullet', 'blitz', 'rapid', 'classical'],
                  'games_reserved_for_humans': 1,   # bots can fill only 1 of the 2 slots: a human can always play
                  'preference': 'human',            # humans jump ahead of bots in the challenge queue
                  'recent_bot_challenge_age': 60,   # and one bot can't re-challenge more than twice a minute
                  'max_recent_bot_challenges': 2},
}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--engine-dir', default=ENGINE_DIR)
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()
    config = yaml.safe_load(open(args.src))
    config['engine']['uci_options'] = {}      # drop the Stockfish-style defaults (Threads, Hash, ...)
    merge(config, OVERRIDES)
    if args.fast:
        merge(config, FAST)
    config['engine']['dir'] = config['engine']['working_dir'] = args.engine_dir
    yaml.safe_dump(config, open(args.dst, 'w'), sort_keys=False)
    print(f"wrote {args.dst} ({'fast' if args.fast else 'free-tier'} profile, engine in {args.engine_dir})")
