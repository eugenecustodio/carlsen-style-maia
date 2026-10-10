#!/bin/sh
# Turn the Lichess bot (gpt579419) on: starts a new 11.5-hour Kaggle session,
# unless one is already running (two sessions = two bots on one account).
KAGGLE="$HOME/.local/bin/kaggle"
KERNEL="jakenorton23/carlsen-bot-runner"
DIR="$(cd "$(dirname "$0")" && pwd)/kaggle"

status=$("$KAGGLE" kernels status "$KERNEL" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*' | cut -d. -f2)
if curl -s "https://lichess.org/api/users/status?ids=gpt579419" | grep -q '"online":true'; then online=online; else online=offline; fi
echo "Kaggle session: ${status:-unknown} | Lichess: $online"

if [ "$status" = "RUNNING" ] || [ "$status" = "QUEUED" ]; then
    echo "The bot is already on. Nothing to do."
else
    echo "Starting a new session ..."
    "$KAGGLE" kernels push -p "$DIR" | tail -1
    echo "Waiting for the bot to come online (about 1-3 minutes) ..."
    for i in $(seq 1 30); do
        sleep 10
        if curl -s "https://lichess.org/api/users/status?ids=gpt579419" | grep -q '"online":true'; then
            echo "The bot is ON: https://lichess.org/@/gpt579419 (runs about 11.5 hours)"
            break
        fi
    done
fi
printf 'Press Enter to close.'; read _
