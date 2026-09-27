# Running TradingBot on a Linux VPS

The VPS is where the bot's schedule runs. The GitHub Actions workflows keep only
their manual "Run workflow" button (`workflow_dispatch`); their `schedule:`
blocks are commented out. The VPS runs the same entry points:

| Job | GitHub workflow | Command | systemd timer |
|---|---|---|---|
| `cycle` | `trading_bot.yml` | `pytest -q` then `python main.py --config config.yaml` | 10:45 and 13:30 New York time, plus 02:00 UTC |
| `watch` | `volume_watch.yml` | `python volume_watch.py --config config.yaml` | every 15 min |
| `refresh` | `refresh_positions.yml` | `python position_metrics.py --config config.yaml` | every 15 min (+5) |
| `screening` | `weekly_screening.yml` | `pytest -q` then `python screening.py --output symbols.yaml` | Monday 06:00 UTC |

`manual_analysis.yml` is left out. It's triggered from the dashboard via
`repository_dispatch`, which only GitHub Actions can receive.

> **Never run both at once.** If Actions schedules and these timers are both
> active, two bots trade the same Alpaca account. They would double orders and race
> each other's git pushes. The schedules are already removed from the workflow
> files. Don't uncomment them while the timers are enabled. To move back to Actions:
> `systemctl disable --now tradingbot-*.timer` first, then uncomment the `schedule:` blocks.
>
> A **manual** Actions run (the "Run workflow" button, or the dashboard's
> "Executar analisi ara") still trades from GitHub's machines, outside this
> server's lock. Avoid starting one while a VPS job is running. From the server,
> `systemctl start tradingbot@cycle` does the same thing under the lock.

## One-time setup (Ubuntu/Debian, as root)

```bash
# 1. A dedicated user and the code
apt-get update && apt-get install -y git python3 python3-venv
useradd --system --create-home --shell /bin/bash tradingbot
git clone https://github.com/guiilleeee/TradingBotPapin.git /opt/tradingbot
chown -R tradingbot:tradingbot /opt/tradingbot

# 2. Virtualenv + dependencies (Python 3.11+)
sudo -u tradingbot python3 -m venv /opt/tradingbot/.venv
sudo -u tradingbot /opt/tradingbot/.venv/bin/pip install -r /opt/tradingbot/requirements.txt

# 3. Secrets -- same names as the GitHub secrets
sudo -u tradingbot cp /opt/tradingbot/deploy/.env.example /opt/tradingbot/.env
sudo -u tradingbot nano /opt/tradingbot/.env
chmod 600 /opt/tradingbot/.env
```

### 3b. Telegram (alerts)

The bot sends one line per event (`BUY 10 AAPL @ $182.30`). It never waits for a
reply; trading is fully autonomous.

1. In Telegram, open **@BotFather**, send `/newbot`, and follow the prompts. It
   replies with the bot token (`123456789:AA...`). That token controls the bot,
   so never commit it, paste it into chats, or post it anywhere.
2. Open a chat with your new bot and send it any message (a bot can't message
   you first).
3. Get your chat id from the server:

   ```bash
   . /opt/tradingbot/.env
   curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates" | python3 -m json.tool
   # -> "chat": {"id": 123456789, ...}   that number is TELEGRAM_CHAT_ID
   ```

Put both values in two places:

1. **VPS:** `TELEGRAM_BOT_TOKEN=` and `TELEGRAM_CHAT_ID=` in `/opt/tradingbot/.env`
2. **GitHub:** repo → Settings → Secrets and variables → Actions →
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (used by manual Actions runs)

If the token ever leaks, send `/revoke` to @BotFather and update both places
with the new token.

Test it from the server:

```bash
. /opt/tradingbot/.env
curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
     -d chat_id="${TELEGRAM_CHAT_ID}" -d text="Prova"
# The phone should show "Prova".
```

### 3c. Web Push (optional, alongside Telegram)

Sends the same one-line alerts as browser notifications from the dashboard, with
no app to install. Do this after step 4, since it pushes one file.

```bash
cd /opt/tradingbot
sudo -u tradingbot .venv/bin/python web_push.py setup
# -> writes docs/push_config.json (public keys) and prints two lines:
#    VAPID_PRIVATE_KEY=...   PUSH_SUBSCRIPTION_KEY=...
sudo -u tradingbot nano .env        # paste both lines; never commit them
sudo -u tradingbot git add docs/push_config.json
sudo -u tradingbot git commit -m "Enable web push" && sudo -u tradingbot git push
```

Then open the dashboard and tap the bell. It asks for the same GitHub token as
the other buttons, and stores this browser's subscription **encrypted** in
`docs/push_subscriptions.json` (only this server's `PUSH_SUBSCRIPTION_KEY` can
read it). Test it: `sudo -u tradingbot .venv/bin/python web_push.py test`.

- **iPhone/iPad:** first add the dashboard to the home screen (Share > Add to
  Home Screen, iOS 16.4+) and open it from there; Safari tabs can't receive pushes.
- **Desktop:** the browser has to be running to show them.
- For manual GitHub Actions runs to push too, add both values as repository
  secrets with the same names.
- Running `setup` again makes new keys: every device then has to tap the bell again.

### 4. Let the server push (only if `PUBLISH=1`)

The dashboard is GitHub Pages, served from `docs/`. It updates only when results
are pushed. Give the server a deploy key with write access:

```bash
sudo -u tradingbot ssh-keygen -t ed25519 -N "" -f /home/tradingbot/.ssh/id_ed25519
cat /home/tradingbot/.ssh/id_ed25519.pub
#   -> GitHub repo -> Settings -> Deploy keys -> Add, tick "Allow write access"
cd /opt/tradingbot
sudo -u tradingbot git remote set-url origin git@github.com:guiilleeee/TradingBotPapin.git
sudo -u tradingbot git config user.name  "tradingbot-vps"
sudo -u tradingbot git config user.email "tradingbot-vps@users.noreply.github.com"
sudo -u tradingbot ssh -o StrictHostKeyChecking=accept-new -T git@github.com   # "successfully authenticated"
```

With `PUBLISH=0`, nothing is pulled or pushed. Results stay in the server's
working copy and the public dashboard stops updating.

### 5. Dry run each job once by hand

```bash
cd /opt/tradingbot
sudo -u tradingbot bash deploy/run_job.sh refresh     # cheapest: no model calls
sudo -u tradingbot bash deploy/run_job.sh watch
```

(`cycle` makes real model calls, and real orders if `live_execution: true`. Run it
by hand only when that's what you want.)

### 6. Install the timers (systemd, recommended)

```bash
cp /opt/tradingbot/deploy/systemd/tradingbot@.service   /etc/systemd/system/
cp /opt/tradingbot/deploy/systemd/tradingbot-*.timer    /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now tradingbot-cycle.timer tradingbot-watch.timer \
                       tradingbot-refresh.timer tradingbot-screening.timer
systemctl list-timers 'tradingbot-*'        # shows the next run of each
```

The unit assumes `/opt/tradingbot` and the `tradingbot` user. If yours differ,
edit `User=`, `WorkingDirectory=` and `ExecStart=` in `tradingbot@.service`.

**Crontab instead of systemd:** `mkdir -p /opt/tradingbot/logs`, then run
`sudo -u tradingbot crontab /opt/tradingbot/deploy/crontab.txt`. Use one or the other.

## Day to day

```bash
journalctl -u 'tradingbot@*' -f              # live logs, all jobs
journalctl -u tradingbot@cycle --since today # today's cycles
systemctl start tradingbot@cycle             # run a cycle now (off-schedule)
systemctl disable --now tradingbot-*.timer   # stop everything
```

Every `cycle` run, including a failed one, is recorded in `docs/job_status.json`
and pushed. The dashboard's "Cicles programats" panel shows each scheduled
cycle as completed, failed (with the error), in progress, or missed.

Deploying a code or config change means pushing it to `main`. With `PUBLISH=1`, every
job starts with `git pull --rebase`, so the next tick picks it up.

## Why the schedule needs no DST edits

NYSE is open 09:30–16:00 New York time. That's 13:30–20:00 UTC while the US is on
daylight time and 14:30–21:00 UTC the rest of the year. The US and Europe switch
on different dates, so for a few weeks each spring and autumn the gap from Spain
is 5 hours instead of 6.

- **systemd**: the cycle timer is written in `America/New_York` time, so 10:45 and
  13:30 ET follow US DST automatically. Europe's switch doesn't matter.
- **cron / GitHub Actions**: these run in UTC, so the times are 14:45 and 17:30 UTC.
  Both fall inside the session under either US regime (09:45/12:30 ET in winter,
  10:45/13:30 ET in summer).

Outside NYSE hours the overnight cycle only ever proposes crypto. The funnel
won't pay the model to suggest an equity buy that execution would skip anyway.

## Overlapping runs

Every job takes one lock, `/opt/tradingbot/.tradingbot.lock` (`flock`). A job can't
overlap any other, whatever mix of timers fires:

- `cycle` and `screening` **wait** up to 1 hour for the lock.
- `watch` and `refresh` **skip** their tick if the lock is busy.

This is stricter than GitHub Actions on purpose. A watch tick can start a full
trading cycle, and every job shares one working tree and `trading_bot.db`. The
lock belongs to the process, so it's released on any exit, including a crash or
`kill -9`. Nothing cancels a running job the way GitHub's `cancel-in-progress`
did. systemd also never starts a second copy of a oneshot unit that's still
running.
