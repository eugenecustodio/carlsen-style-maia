# Carlsen-style Maia-3 chess engine and Lichess bot

A chess engine that imitates **Magnus Carlsen's move choices**: Maia3-23M (a model of human play)
fine-tuned on 7,800 of Carlsen's published games. On held-out games it predicts his exact move
**60.2%** of the time (plain Maia3-23M: 59.2%) and has his move in its top 3 **87.7%** of the time.

> This is an AI imitation of Carlsen's style. It is **not** Magnus Carlsen and is not affiliated with him.

It plays at roughly strong-club / titled-player level and makes human-style mistakes. It does not
calculate. Turn on `StockfishFilter` for a stronger "Carlsen + safety net" mode.

## Contents

| Path | What it is |
|---|---|
| `engine/maia3-carlsen.onnx` | The model (91 MB, ONNX). `engine/maia3-carlsen.json` has its metrics and sha256 |
| `engine/carlsen_uci.py` | UCI engine (PyChess, lichess-bot, Arena, Cute Chess, ...) |
| `engine/carlsen_core.py` | Model + move-choice logic |
| `carlsen-engine.sh` | Launcher for PyChess on this laptop |
| `server.py`, `Dockerfile`, `render.yaml`, `make_config.py` | Lichess bot hosting (Render free tier) |
| `tests/` | UCI compatibility tests (`../env/bin/python -m pytest -q tests/`) |

## Use it in PyChess

Edit → Engines → **New** → choose `carlsen-engine.sh` → protocol **UCI**. Then use it as an
opponent or as the analysis engine (set `MultiPV` 3 to see his top candidates).

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `CarlsenStyle` | `auto` | `auto` picks classical/rapid/blitz/bullet Carlsen from the game clock; or fix one; `any` = average Carlsen |
| `Temperature` | 0 | In hundredths. 0 = his single most likely move; 30–70 = varied, human-like play |
| `TopP` | 100 | Percent; with Temperature > 0, sample only from the top moves covering this probability |
| `MultiPV` | 1 | Number of candidate moves shown |
| `OppoElo` | 2700 | Opponent rating (lichess-bot sends the real one automatically) |
| `StockfishFilter` | false | Veto model moves that lose > 4% expected score vs Stockfish (needs Stockfish) |
| `StockfishPath`, `StockfishDepth` | `/usr/games/stockfish`, 12 | For the filter |
| `Threads` | 1 | CPU threads for the model (local mode) |

## Run it as a Lichess bot on Render (free)

**What "free" means here.** Render's free web service has 0.1 CPU and 512 MB RAM, enough for this
model at about 1 second per move. It sleeps after 15 minutes without incoming traffic, so
`server.py` pings itself every 10 minutes. 750 free hours a month covers one service running all the time.
The bot plays one game at a time and accepts 3+0 and slower (blitz, rapid, classical).

1. **Create a new Lichess account for the bot.** It must never have played a game. Don't use
   "Magnus" or "Carlsen" in the username, because Lichess forbids impersonation. Say in the profile bio that it is a
   Carlsen-*style* Maia model and link this repository (the AGPL licence requires the source to be available).
2. **Create a token** while logged in as the bot: <https://lichess.org/account/oauth/token/create>,
   tick **"Play games with the bot API" (`bot:play`)**. Keep it secret.
3. **Upgrade the account to a BOT account.** This can't be undone:
   ```sh
   curl -d '' https://lichess.org/api/bot/account/upgrade -H "Authorization: Bearer YOUR_TOKEN"
   ```
4. **Deploy on Render:** <https://dashboard.render.com> → **New → Blueprint** → choose this repository.
   Render reads `render.yaml` (free web service, Docker). When asked, enter:
   - `LICHESS_BOT_TOKEN`: the token from step 2
   - `LICHESS_BOT_NAME`: the bot's username
5. Wait for the build (~3–5 minutes). Then open `https://<service>.onrender.com/`. The status page
   should say **online**, and `/status` should show `"lichess_bot_running": true`.
6. *(Recommended backup)* Create a free job at <https://cron-job.org> that opens
   `https://<service>.onrender.com/healthz` every 10 minutes, in case the self-ping ever stops.

Challenge the bot from another account at `https://lichess.org/@/<bot name>`.

**Free-tier caveats:** Render may restart free instances at any time (the bot reconnects
automatically, but a game in progress can be lost). Render could also change its free-tier rules.
For uninterrupted play, switch `plan: free` to `plan: starter` in `render.yaml` ($7/month, 5× the CPU).

## Licence

AGPL-3.0 (Maia-3 and lichess-bot are AGPL-3.0). If you run this as a public bot, keep this source
code publicly available.
