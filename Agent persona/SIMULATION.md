# Simulation, benchmarks and ablations

## One cycle

```
news feed ──► Town Crier ──► segment summary + stocks + RAG-Qs
(archive |    (LLM or                    │
 yahoo |       heuristic)                ├──► doc retrieval (once, for the desk)
 finnhub)                                │      archive BM25 (+Chroma) + this run's live index
     │                                   │              │
     │                                   │      Town Crier condenses ──► Historical Context
     │                                   ▼
     │                        ┌── 15-Minute Brief (one per agent) ──┐
     │                        │  news summary + digest              │
     │                        │  your balance      (Agent Assets)   │
     │                        │  your last trades  (Agent Actions)  │
     │                        │  peers' actions    (Agent Actions)  │
     │                        │  historical context                 │
     │                        │  your reflections  (Reflections)    │
     │                        │  order instructions + ceilings      │
     │                        └──────────────┬──────────────────────┘
     │                                       ▼
     │                          agent panel (roundtable, 2 rounds)
     │                                       │
     │                     ┌─────────────────┴──────────────────┐
     │                     ▼                                    ▼
     │              consensus forecast                   each agent's orders
     │                     │                                    │
     │            score vs realized move             market simulator (cash / holdings /
     │                     │                          caps / cooldown / slippage)
     │            target weights ──► consensus book          │
     │                                          per-agent books ──► actions.sqlite  (public)
     └──► indexed into the live news index ◄──┘              └──► agent_assets.sqlite (private)
          (after retrieval, so a cycle never cites itself as history)

   end of cycle: the PREVIOUS cycle's trades are now judged by the market
        ──► each agent reflects on its own day ──► reflections.sqlite (private, per agent)
                                                          │
                                    recalled into that agent's next brief ◄┘
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

## The Town Crier

One agent reads the raw segment so the traders do not have to. It produces:

* **summary** — 2–4 sentences on what happened and what it plausibly means.
* **stocks** — which universe tickers the segment actually bears on.
* **RAG-Qs** — the retrieval questions: up to 4 about historical news, up to 3 about the
  agents' own past performance (the second set is what Phase 3's reflection store will answer).

Retrieval then runs **once for the desk** on those questions, and the Town Crier condenses the
result into the brief's Historical Context. Before this, every agent hand-rolled a query string
and read raw headlines.

`--town-crier llm|heuristic|auto` (auto = LLM when the panel uses one). The heuristic path is
deterministic frequency counting and costs nothing, so long backtests are not forced to pay
for it.

## The 15-Minute Brief

One brief per agent per cycle, assembled from four sources:

| section | source |
|---|---|
| `news_summary`, `stocks_discussed`, `news_digest` | Town Crier |
| `your_balance` (cash, positions, market value, unrealized P&L) | Agent Assets DB (private) |
| `your_last_trades` — including rejected orders **and why** | Agent Actions DB |
| `peer_recent_actions` — what the others *did*, not what they said | Agent Actions DB (public) |
| `historical_context` — summary, documents, questions asked | central retrieval |
| `your_reflections` | Reflections store (private, this agent only) |
| `order_instructions` — the rules, plus `you_can_sell_at_most` and `you_can_buy_at_most` | execution config + book |

The last row matters: the ceilings are computed for the model rather than left to its
arithmetic. Adding them and one sharp long-only sentence took a real LLM run from a 33% fill
rate (agents trying to sell what they did not own) to 100%.

`--no-briefs` reverts to the old path — raw digest, each agent retrieving for itself — which
makes "does the brief help?" an ablation rather than an assumption.

## Live news index

The archive stops at 2020-06-11, so a run's own news would otherwise be invisible to retrieval.
Every cycle's headlines are written to `data/runs/live_news.sqlite` (SQLite FTS5, no key, no
embedding cost) and fused into retrieval by RRF alongside the archive. When the ChromaDB store
and `GOOGLE_API_KEY` are both present, `ChromaNewsWriter` also upserts them into the dense
collection so both tiers stay in step.

Indexing happens **after** the cycle retrieves, so a segment is never returned as its own
historical context. From the next cycle onward it is. `--no-live-index` disables it.

## Reflection: the agents' private memory

At the end of each cycle every agent reads its own day back — the orders it sent (fills and
refusals), the forecasts it made, what the market then did, and what happened to its book — and
writes a lesson. Those lessons go into `data/runs/reflections.sqlite` (Vector Store 2), and are
retrieved into that agent's next brief using the Town Crier's *insight* questions.

**Reflection is deferred by one cycle, on purpose.** A day's trades cannot be judged until the
market has moved, so the engine parks each cycle's activity and settles it once the outcome is
on the tape. A reflection therefore never contains information the agent could not have had.

**The store is private, and that is enforced rather than assumed**: every read is scoped to one
`agent_id`, in SQLite and in the Chroma metadata filter alike. If the personas shared a memory
they would converge, and a panel that cannot disagree is an expensive single agent.

A real lesson from a Brexit-window run:

> *"Right on GLD (+490 bps) but sized too small to overcome spread/commissions, finishing
> −$39.50. As a contrarian fading post-Brexit panic, I correctly identified gold as a fear bid
> but my 0.55 confidence capped position size when the move was large and obvious."*
> — `contrarian_value_llm_v1`, tags `RISK_OFF, POSITION_SIZING, GOLD_HEDGE, BREXIT_REACTION`

Flags: `--reflection-mode llm|heuristic|auto`, `--reflections-top-k N`, `--no-reflections`
(the ablation: agents with no memory of their own past days).

A day with no trades and no open positions produces no reflection — there is nothing to learn
from having done nothing.

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
