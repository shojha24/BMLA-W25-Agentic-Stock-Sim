# Simulation, benchmarks and ablations

## One cycle

```
news feed ──► digest builder ──► agent panel ──► consensus ──► score vs realized move
(archive |     (heuristic |      (roundtable,          │
 yahoo |        LLM)             2 rounds, each        └────► target weights ──► consensus book
 finnhub)                        agent sees its
                                 own balance sheet)
                                        │
                                        └─► each agent's own orders
                                                  │
                                             market simulator ──► per-agent books
                                             (cash / holdings /        │
                                              caps / cooldown)         ├─► actions.sqlite  (public)
                                                                       └─► agent_assets.sqlite (private)
```

Two books run side by side and are reported together:

* **`agents_consensus`** — the consensus forecast turned into target weights centrally.
* **`book:<agent>`** — each persona placing its own orders, filled by the market simulator.

That is the comparison the whiteboard asks for: does the panel do better acting as one
sized portfolio, or as three traders with their own money?

Every cycle is appended to `data/runs/<run_id>.jsonl` (digest, per-agent output, consensus,
realized returns, books) and the run ends with `<run_id>_report.json`. A run can therefore be
re-scored later without paying for the LLM calls again.

## Feeds

| feed | key needed | use |
|---|---|---|
| `archive` | no | replay 2009-2020 from the local 876k-headline store — this is what makes backtesting possible |
| `yahoo` | no | live headlines per ticker from Yahoo Finance RSS — the 15-minute loop |
| `finnhub` | `FINNHUB_API_KEY` | live, better coverage than RSS |
| `fixture` | no | a digest JSON file, for demos and tests |

The archive has no SPY or UUP rows, so the default backtest universe is
`QQQ,TLT,GLD,XLE,IWM,NVDA` — tickers that actually have coverage. SPY is still fetched as the
index benchmark.

## Prices

Daily adjusted closes from the public Yahoo chart endpoint, cached to `dataset/prices/*.csv`.
No API key, no extra dependency. Forecasts are scored on close-to-close forward returns in
basis points, which is also what the portfolio marks against.

## Orders and the market simulator

Agents emit `actions` alongside forecasts:

```json
{"side": "BUY", "ticker": "TLT", "qty": 250, "rationale": "...", "news_refs": ["news_1"]}
```

The positional form from the whiteboard (`["Buy", 250, "TLT"]`) is accepted too. `MarketFillVenue`
fills at the market price plus slippage and enforces, per order:

| rule | flag | default |
|---|---|---|
| pay only with cash you have | — | always |
| sell only shares you hold | `--agents-may-short` | no shorting |
| whole shares | — | on |
| max single order as share of that agent's equity | `--max-order-pct` | 35% |
| wait N cycles before re-trading a ticker | `--cooldown-cycles` | 0 (off) |
| price concession | `--slippage-bps` | 2 bps |
| commission on notional | `--cost-bps` | 1 bp |

An order that breaks a rule is **cut down (PARTIAL) or REJECTED with a reason**, never silently
dropped: the fill rate and reject reasons are reported, because "the LLM asked for something
impossible" is a measurable property of the agent, not an error to swallow.

`ExecutionVenue` is an interface. `MarketFillVenue` keeps prices exogenous (real market data);
a limit-order book with endogenous price formation can replace it without touching the engine.

## Two databases

| store | file | holds |
|---|---|---|
| **All Agent Actions** (public) | `data/runs/actions.sqlite` | one row per order incl. rejects — `last_trades()`, `peer_actions()`, `trades_on_day()` |
| **Agent Assets** (private) | `data/runs/agent_assets.sqlite` | per-agent cash, equity and positions each cycle |

`trades_on_day()` and `peer_actions()` exist for the phases not built yet: the 15-minute brief's
"your last trades" line, and the end-of-day reflection pipeline.

## Consensus sizing

`weight = direction × consensus confidence`, capped at 34% per name and 100% gross, shorts
allowed (`--long-only` to disable), 1 bp of cost per traded notional (`--cost-bps`).

Sizing deliberately ignores `expected_return_bps`: LLM magnitude estimates are far noisier
than their directional calls, and letting them size positions imports that noise into the P&L.
The magnitudes are still scored (MAE), just not traded.

## Benchmarks

| model | what it is |
|---|---|
| `spy_buy_hold` | the index |
| `always_long` | equity drift; hard to beat in a bull sample |
| `random` | seeded coin flips — the zero-skill reference |
| `persistence` | yesterday's move repeats |
| `reversal` | yesterday's move reverses |
| `sentiment_rule` | digest sentiment × risk beta, no LLM — the "is the LLM earning its keep?" control |
| `agents_round1` | the panel *before* it talks — the communication control |
| `agent:<name>` | each persona scored on its own, so you can see if consensus beats its members |

All of them emit the same `Forecast` records and run through identical sizing, costs and
scoring.

## Metrics

Forecast level: hit rate, confidence-weighted hit rate, Brier score, mean absolute error in
bps, share of FLAT calls. Portfolio level: cumulative and annualized return, Sharpe, max
drawdown.

Hit rate alone is not enough — a model that is right 55% of the time on tiny moves and wrong on
the large ones loses money — which is why the equity curve is reported next to it.

## Commands

```bash
# offline replay, no API key, no cost
python "Agent persona/src/simulate.py" backtest --start 2016-01-04 --end 2016-06-30 --mode mock

# real panel, LLM-labelled digests, 12 cycles around Brexit
python "Agent persona/src/simulate.py" backtest --start 2016-06-13 --end 2016-07-15 \
    --mode llm --digest-mode llm --max-cycles 12

# live: poll Yahoo RSS every 15 minutes
python "Agent persona/src/simulate.py" live --interval 15 --cycles 8 --mode llm

# ablations: full panel vs no-RAG vs no-communication vs single agent
python "Agent persona/src/simulate.py" ablate --start 2016-01-04 --end 2016-03-31 --mode mock
```

Useful flags: `--universe`, `--horizon 1d|2d|5d`, `--step-days`, `--max-cycles`, `--long-only`,
`--max-gross`, `--cost-bps`, `--benchmarks none`, `--digest-mode heuristic|llm`, `--feed`.

## Cost

One cycle with the full panel is `agents × rounds` LLM calls (6 by default), plus one more with
`--digest-mode llm`. A 100-cycle backtest is ~700 calls — fine on a free model with retries,
but use `--mode mock` for mechanics and `--max-cycles` when iterating.

## Ablation caveat

`--mode mock` and `--mode hardcoded` agents never read the retrieved context, so the `no_rag`
variant comes out identical to `full_panel`. Only `--mode llm` gives a meaningful RAG ablation;
`ablate` prints a warning when you run it otherwise.

## A trap worth knowing

Agents start with cash and no positions, and shorting is off by default. A panel that turns
bearish on day one therefore has nothing it can legally do: it cannot sell what it does not
hold, so it places no orders and the books sit flat. That is correct behaviour, not a failure -
but if you see `orders: 0` for several cycles, check whether the panel is simply bearish into
an empty book. `--agents-may-short` lets it express that view.

## Known limits

* Cycles are daily. The 15-minute loop is live-only; there is no intraday price history to
  backtest a 15-minute horizon against.
* Trades fill at the same close used to compute the signal. There is no slippage model beyond
  the flat cost, and no borrow cost on shorts.
* To keep that honest, a backtest cycle only reads news published up to 20:00 UTC (16:00 ET,
  `EngineConfig.close_hour_utc`). News that breaks after the close belongs to the next cycle.
* The archive ends 2020-06-11, so backtests cannot cover anything more recent.
* `--digest-mode heuristic` sentiment is a keyword lexicon: fine as a control, weak as a labeler.
