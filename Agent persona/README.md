# Agent panel

## Flow

```
digest.json + state.json
        │
        ├─ RAGNewsTool.retrieve()      BM25 (+dense) → date cutoff → ticker filter
        │                               → recency decay → recent/historical barbell
        ▼
   round 1: 3 personas forecast independently (in parallel)
        ▼
   round 2: each persona sees the others' forecasts + reasoning, revises
        ▼
   consensus: confidence-weighted per ticker, discounted by disagreement
        ▼
   {consensus, consensus_round1, revision deltas, per-round outputs, errors}
```

Round 1's consensus and the round-1→2 deltas are both kept, so "does agent communication
help?" is an ablation you can run rather than an assumption.

## Personas

| key | reads the news as | tilt |
|---|---|---|
| `macro_econ` | regime / transmission channels (rates, inflation, growth, FX) | cross-asset consistency |
| `quant_momentum` | signal persistence | trend continuation, FLAT when no edge |
| `contrarian_value` | crowd overreaction vs genuine repricing | fades sentiment, sizes smaller |

They are deliberately non-redundant: a panel that always agrees is an expensive single agent.
`SentimentBaselineAgent` (`agents/baseline.py`) is a rule-based control with no LLM.

## The forecast contract

Every agent emits `forecasts`, the scorable unit of the system:

```json
{"ticker": "SPY", "direction": "UP|DOWN|FLAT", "expected_return_bps": -45.0,
 "horizon": "1d", "confidence": 0.72, "rationale": "...", "news_refs": ["news_1"]}
```

All model output passes through `core/schema.py`, which repairs what LLMs actually return:
fenced JSON, confidence on a 0-10 or 0-100 scale, a `direction` that contradicts the sign of
`expected_return_bps`, duplicate tickers, and tickers that are not in `state.prices` (dropped -
an agent may only forecast what the simulator can price).

## Consensus

Per ticker, weighted by each agent's confidence:

* `expected_return_bps` - weighted mean
* `net_vote` - weighted directional vote in [-1, 1]; drives the consensus direction
* `agreement` - weighted share of the modal call (unanimous FLAT counts as agreement)
* `confidence` - mean confidence × agreement, so a split panel cannot look confident
* `dispersion_bps` - spread of member estimates, reported rather than hidden

## CLI

```bash
python "Agent persona/src/main.py" [options]

--task roundtable|panel|single|rag-status   roundtable = 2 rounds; panel = 1 independent round
--mode llm|mock|hardcoded                   mock = deterministic offline stub; hardcoded = rule-based
--model minimax/minimax-m3:free             any OpenRouter model id
--agents macro_econ,quant_momentum,contrarian_value
--rounds 2 --horizon 1d
--no-rag                                    ablation: run without retrieval
--with-baseline                             add the rule-based control to the panel
--out results.json --quiet
```

Free OpenRouter models come and go and rate-limit hard; if the default 404s or 429s, list what
is currently free at https://openrouter.ai/models?q=free and pass `--model`.

## Layout

```
src/
  main.py                  CLI
  agents/
    base.py                BaseAgent interface (round-2 aware)
    persona.py             PersonaSpec + LLMPersonaAgent (one implementation, N personalities)
    personas.py            the three persona definitions
    baseline.py            rule-based control agent
    macro_econ.py          back-compat shim for the old import path
  orchestration/
    roundtable.py          2-round panel, parallel, fault-tolerant
    consensus.py           aggregation + disagreement metrics
  llm/
    openrouter_client.py   retries, JSON mode, real error messages
    mock_client.py         deterministic offline stub
  core/
    types.py schema.py io.py utils.py
  tools/
    rag.py                 hybrid retriever
    headline_store.py      local doc_id → headline text (SQLite)
tests/                     29 tests, no network
```

## Simulation

The panel is one stage of a loop: news feed → digest → panel → consensus → scoring →
portfolio. See `SIMULATION.md` for feeds, benchmarks, metrics, ablations and the CLI.

```
src/
  simulate.py              backtest / live / ablate / score
  agents/  town_crier.py   summarizes the segment, writes the retrieval questions
           reflection.py   end-of-day self-review, written to private memory
  data/    news_feed.py market_data.py digest_builder.py
  sim/     brief.py portfolio.py execution.py engine.py actions_db.py assets_db.py
  tools/   rag.py news_index.py reflection_store.py headline_store.py
  eval/    metrics.py benchmarks.py rescore.py
```

Each cycle the Town Crier summarizes the news and writes the retrieval questions; retrieval
runs once for the desk; each agent reads a 15-Minute Brief carrying its own balance, its own
last trades (rejections included), what its peers did, and the historical context; then places
real orders against its own book through a market simulator. At the end of the cycle each
agent reflects on its previous day — judged by what the market actually did — into a private
memory that is recalled into its own later briefs. See `SIMULATION.md`.

## Not built yet

Intraday backtesting (there is no minute-bar history behind the 15-minute live loop),
borrow costs and slippage beyond a flat per-notional charge, any news source past 2020-06-11
for replay, and endogenous price formation (the market simulator fills against real prices;
`ExecutionVenue` is the seam where an order book would go).
