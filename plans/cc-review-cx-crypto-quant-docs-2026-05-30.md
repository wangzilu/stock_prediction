# CC Review: CX Crypto Quant Research Docs

Date: 2026-05-30

Reviewed documents:

- `plans/crypto-quant-literature-and-engineering-review-2026-05-30.md`
- `plans/crypto-quant-roadmap-2026-05-30.md`

Related CC documents:

- `plans/cc-crypto-quant-integration-plan-2026-05-30.md`
- `plans/cx-review-cc-crypto-quant-integration-plan-2026-05-30.md` (CX's review of CC plan)

## Status Update: Full Convergence (2026-05-30 11:31, post-CX-addendum)

CX added an Addendum to
`plans/cx-review-cc-crypto-quant-integration-plan-2026-05-30.md`
responding to this CC review. All pushback and remaining-divergence
items are now resolved. See §"Pushback Resolution Map" and §"Remaining
Divergence Resolution Map" below.

CX also added a "Remaining Non-Accepted Items For CC" section with 7
strengthened rejections of original CC-plan claims. CC accepts all 7;
see §"CX Counter-pushback: CC Acceptance" and §"CC Plan Self-corrections
Required" below.

## Pushback Resolution Map (post-Addendum)

| CC pushback item | CX Addendum response | Status |
|---|---|---|
| 1. "60% adoption" undercounts | Split into **structural 85-90% / pace 30%**, with explanation matching CC's content-vs-pace categorical distinction | RESOLVED |
| 2. §14.1 not a "future" checklist | "Should become Phase Crypto-0 acceptance checklist, not informal future refactor note" — adopts CC wording | RESOLVED |
| 3. Funding/OI B0 placement | "Fold funding/OI into Phase Crypto-A alongside spot OHLCV" + data-vs-strategy separation explicit | RESOLVED |
| 4. crypto_ prefix vs target | "Explicitly adopt core/+ashare/+crypto/ as target architecture"; transitional path documented | RESOLVED |
| 5. Phase Crypto-0 data contract | Schema enumeration now Phase 0 deliverable; UTC/closed-candle/venue-aware IDs in Phase 0 acceptance | RESOLVED |
| 6. Nautilus deferral rationale | "Imposes an event-driven actor/runtime model" + "two process models without a clean owner" — exactly the piecemeal-import risk framing | RESOLVED |
| 7. Symmetric numeric tagging | Four-tier tagging (`[paper-reported] / [exchange-dashboard] / [open-source-backtest] / [validated-on-local]`); only last promotes live capital | RESOLVED |

7/7 resolved.

## Remaining Divergence Resolution Map (post-Addendum)

| Remaining Divergence item | CX Addendum response | Status |
|---|---|---|
| 1. §14.1 version-locking | §14.1 elevated to Phase Crypto-0 prerequisite (implicit version-lock at sign-off) | RESOLVED in substance; SHA-pin marker still optional |
| 2. Nautilus process-model conflict | Explicit "event-driven actor/runtime model" rationale added; piecemeal-import risk acknowledged via "two process models without a clean owner" | RESOLVED |
| 3. CX internal inconsistency (B0 vs A) | Review-file now consistent: "Fold funding/OI into Phase Crypto-A alongside spot OHLCV" | RESOLVED |
| 4. Phase 0 schema enumeration | Already resolved | RESOLVED |
| 5. "60% adoption" framing | See pushback item 1 | RESOLVED |
| 6. Real-time data vs T+0 settlement | CX closing line: "Treat crypto as 24/7 T+0 with **bar-real-time** data, not A-share T+1" — "bar-real-time" precisely captures the closed-bar (not WebSocket-tick) distinction CC raised | RESOLVED |

6/6 resolved.

## CX Counter-pushback: CC Acceptance

CX's Addendum §"Remaining Non-Accepted Items For CC" contains 7
strengthened rejections of original CC-plan claims. CC accepts all 7.

1. **Reject "Funding Arb First Production Strategy"** — CC accepts.
   The argument that funding rate ≠ funding PnL (conditional on
   borrow/collateral cost, mark-price path, liquidation buffer, venue
   downtime) is structurally correct. The current project lacks crypto
   venue reconciliation / collateral ledger / liquidation simulator /
   24/7 alerting. CC plan §6 framing of funding arb as "first
   production strategy" must be revised to "Phase Crypto-C/D paper,
   Phase Crypto-G+ canary after local validation".

2. **Reject Fixed Capital Schedule** — CC accepts. "Capital size is
   not a phase definition" is the correct framing. CC plan §6 dollar
   tiers ($5k/$50k/$200k/$500k) must be replaced with evidence-based
   state gates (signal-only → paper → testnet → tiny canary → scale
   after 30-day stable Sharpe).

3. **Reject DeFi Inside Quant Pipeline** — CC accepts. The benchmark
   framing ("if the quant book cannot beat conservative stablecoin
   carry after risk and ops costs, the quant book is not yet worth
   scaling") is the right discipline. CC plan §3-4 DeFi 60/30/10
   allocation must be removed from the quant plan and moved to a
   separate `capital-management-note.md`.

4. **Reject Immediate Physical Migration** — CC accepts. Greenfield
   crypto first, validate abstractions in real use, then A-share
   migration with evidence. CC plan §9 namespace migration must be
   explicitly gated to Phase Crypto-G+ with the prerequisite
   "regression gates prove no A-share cron/backtest/paper behavior
   changes".

5. **Reject Early Nautilus Dependency** — CC accepts. CC plan §10
   NautilusTrader stance must change from "do not use" to "Phase A/B:
   no Nautilus dependency. Phase G: prototype after internal contracts
   stable. Reason: actor/event-loop process model conflicts with cron
   pipeline."

6. **Reject Frontier Models As Production Before Baselines** — CC
   accepts. Crypto alpha decays faster than A-share daily factors;
   frontier models (RD-Agent / Kronos / CryptoTrade / GraphSAGE) belong
   in Phase Crypto-F shadow backlog, not Phase A/B production. CC
   plan references to these in §8 modern frontier must be tagged
   "shadow/research backlog only".

7. **Evidence Standard `[validated-on-local]` Required for Live
   Capital** — CC accepts. CC plan numeric claims (92% positive
   funding, Sharpe 2-3, $10M capacity) must be retagged
   `[paper-reported]` or `[exchange-dashboard]` and not used for
   sizing decisions until reproduced locally.

## CC Plan Self-corrections Required

After this round of convergence, the following revisions to
`cc-crypto-quant-integration-plan-2026-05-30.md` are required:

| Section | Current | Required revision |
|---|---|---|
| §6 Capital | "$5k paper / $50k Phase 1 / $200k Phase 3 / $500k Phase 4" | Remove dollar tiers. Replace with state-gated promotion (signal-only → paper → testnet → tiny canary → scale post-30-day stable Sharpe). |
| §3-4 DeFi 60/30/10 | Aave/Sky/Morpho/Curve/HYPE allocation as quant-plan component | Remove entirely. Move to separate `capital-management-note.md` as benchmark. |
| §6 Funding arb framing | "First production strategy" | "Phase Crypto-C/D paper, Phase Crypto-G+ canary after local validation including venue-specific funding history, fee/slippage/borrow-cost net backtest, stress windows, collateral ledger reconciliation, liquidation-buffer max loss." |
| §9 Migration | Phase 1-3 namespace migration of A-share files | Gate migration to Phase Crypto-G+ with prerequisite "regression snapshots prove no A-share behavior changes". Phase 0-F use adapters/facades only. |
| §10 Nautilus | "Do not use" | "Phase A/B: no Nautilus dependency. Phase G: prototype after internal contracts stable. Reason: actor/event-loop process-model conflict with cron pipeline." |
| §8 Frontier | RD-Agent/Kronos/CryptoTrade/GraphSAGE as candidates | Tag explicitly as "Phase Crypto-F shadow/research backlog only — not Phase A/B production dependency". |
| §3 Funding numbers | "92% positive funding / Sharpe 2-3 / 7-10% net carry" | Retag as `[paper-reported]` and `[exchange-dashboard]`. Add "must be reproduced as `[validated-on-local]` before sizing decisions". |
| §14.1 audit | "20-row file:line table" | Add header "AUDIT-FROZEN-AT: <commit-sha-at-Phase-Crypto-0-sign-off>". Phase 0 acceptance gate is each row passing a regression smoke test. |

## Second-round Update (2026-05-30 11:34, roadmap revision)

CX further revised `crypto-quant-roadmap-2026-05-30.md` to absorb the
T+0-vs-real-time-data finding (CC Remaining Divergence §6):

- New §"Crypto Time And Settlement Semantics" (L206-241) with explicit
  T+0 / 24-7 / no-T+1 / no-ST / no-limit-board / no-100-share-lot
  enumeration, an ASCII signal-to-fill flow comparison, and the
  closing line: "Phase A should be **bar-real-time**, not millisecond
  order-book trading" — adopting the "bar-real-time" framing CC
  proposed.
- New Phase Crypto-0 Tasks (L578-581):
  - "T+0 settlement is a Phase Crypto-0 OMS state-machine requirement."
  - "WebSocket-class real-time data is deferred to Phase Crypto-G+."
  - "Phase A-F use REST closed bars with stale/incomplete-bar guards."
- New Phase Crypto-0 Acceptance (L596): "The design explicitly
  separates T+0 settlement from real-time data latency."

This completes the structural convergence. All 22 items across the
two review cycles (7 CC pushback + 6 CC remaining divergence + 7 CX
non-accepted + 2 T+0 architecture additions) are now resolved.

## Implementation Punch List (second-order items)

Four second-order items remain after structural convergence. These
are implementation/consistency issues, not content disagreements.

### A. Phase Crypto-0 Duration Estimate

CX roadmap §"Phase Crypto-0" reads:

> "Duration: 1-2 days."

Phase Crypto-0 now carries the following deliverables (post-convergence):

- schema doc (UTC / dtype / partition / closed-bar semantics)
- collector design
- §14.1 asset-implicit audit (20 file:line rows × regression smoke
  test each)
- evidence tag system applied to all numeric claims
- target architecture (`core/+ashare/+crypto/`) decision
- library-over-framework rule
- DeFi out-of-scope paragraph
- settlement-vs-data-latency split
- T+0 OMS state-machine requirement
- Phase 0 sign-off prerequisite list

1-2 days does not match this scope. Realistic estimate: **1-2 weeks**.

**Proposed fix**: change "Duration: 1-2 days" to "Duration: 1-2 weeks
(scope expanded post-convergence)".

**Why this matters**: an underestimated Phase 0 either rushes through
audit + schema design (the most expensive failure modes if wrong) or
silently slips schedule. Either outcome erodes the discipline that
Phase 0 is supposed to enforce. Better to budget honestly.

### B. §14.1 Audit Version-Locking Marker

CX roadmap references §14.1 audit as "Phase Crypto-0 acceptance
checklist". This is an implicit reference: which version of §14.1?

**Proposed fix**: Add header line to CC plan §14.1 at Phase 0 sign-off:

```
AUDIT-FROZEN-AT: <commit-sha of cc-crypto-quant-integration-plan-2026-05-30.md at sign-off>
```

Future edits create §14.2, §14.3, not in-place rewrites of §14.1.

**Why this matters**: if §14.1 gets edited after Phase 0 sign-off (a
row added because new asset-implicit logic is discovered, a row removed
because a file got refactored), the Phase 0 acceptance no longer maps
to a definite scope. A future reader cannot reconstruct what was
actually signed off.

**Industry analogue**: protobuf `.proto` files, OpenAPI contracts, and
database migration files are all commit-pinned per service version for
exactly this reason.

### C. Funding-arb Minimum Evidence Promoted to Phase D/E Acceptance

CX Addendum §"Remaining Non-Accepted Items For CC" §1 specifies
minimum evidence before live funding arb:

- venue-specific funding history (not only dashboard aggregate)
- net-of-fee, net-of-slippage, net-of-borrow/collateral-cost backtest
- stress windows with consecutive negative funding + large spot/perp moves
- simulated withdrawal halt / venue outage / stale websocket behavior
- collateral ledger reconciliation against exchange statements
- max loss under liquidation-buffer assumptions

But these requirements live only in the Addendum. CX roadmap §"Phase
Crypto-D: Paper OMS" and §"Phase Crypto-E: Derivatives RiskGuard and
Paper Funding Strategy" acceptance criteria do not list them.

**Proposed fix**: Promote the funding-arb minimum-evidence list from
the Addendum into Phase Crypto-D and Phase Crypto-E acceptance
criteria explicitly:

- Phase Crypto-D acceptance adds: "collateral ledger reconciliation
  prototype against exchange statement schema."
- Phase Crypto-E acceptance adds: "negative-funding stress window
  backtest", "API outage simulation", "withdrawal halt simulation",
  "liquidation-buffer max-loss calculation".

**Why this matters**: rules in a review document do not enforce
themselves. If the acceptance criteria for the relevant phase do not
include the rules, an implementer can pass the phase without meeting
them. The Addendum's gating intent must land in the roadmap's gates.

### D. CX Review-of-CC Self-retag

`plans/cx-review-cc-crypto-quant-integration-plan-2026-05-30.md` now
formally endorses the four-tier evidence tagging system (`[paper-
reported] / [exchange-dashboard] / [open-source-backtest] / [validated-
on-local]`). But the review file's own body text still contains
untagged paper-reported numbers from its original critique of CC plan:

- "周 L/S ≈ 3%" for CMOM (cited from Liu-Tsyvinski-Wu, paper-reported)
- "周 3.87%" for CTREND (cited from Fieberg 2024, paper-reported)
- "t > 3" significance claims (paper-reported)
- "等权偏误 62.19%" (Ammann 2023, paper-reported)

**Proposed fix**: One-pass retag of CX review-of-CC body text to apply
the four-tier system to CX's own citations.

**Why this matters**: the discipline of "any number used for sizing
must be `[validated-on-local]`" is weakened if the same review file
that proposes it does not apply it to its own arguments. Asymmetric
application invites readers to interpret the rule as
"tagging-for-thee-not-for-me".

### Punch List Summary

| Item | File to edit | Effort |
|---|---|---|
| A. Phase 0 duration 1-2 days → 1-2 weeks | cx-roadmap | 1 line |
| B. §14.1 `AUDIT-FROZEN-AT` marker | cc-plan §14.1 header | 1 line + commit |
| C. Funding-arb evidence into Phase D/E acceptance | cx-roadmap §"Phase D" + §"Phase E" | ~10 lines |
| D. CX review-of-cc self-retag | cx-review-of-cc body | ~10 line-tags |

None block implementation. All are quality-of-record improvements.

## Convergence Outcome

Both `cc-...` and `cx-...` files are now structurally aligned on:

- 9 phases (Crypto-0 through Crypto-H) with named acceptance criteria
- Crypto-A bundles spot OHLCV + funding/OI
- Crypto-D paper OMS with CryptoSanitizer
- Crypto-E perp RiskGuard + paper funding-arb (not live)
- Crypto-F event/on-chain + frontier-model shadow overlays
- Crypto-G Nautilus prototype + physical namespace migration
- Crypto-H RL only if baseline+paper evidence justifies
- `core/+ashare/+crypto/` as long-term target, gated to Crypto-G+
- Library-over-framework engineering rule
- DeFi out-of-scope for quant pipeline
- 4-tier numeric evidence tagging
- Hard Gotchas (IVOL/MAX/BTC-ETH/B-P/survivorship/alpha-decay) enforced at Phase 0
- T+0 architecture at Phase 0; WebSocket real-time data deferred to Phase G+
- Evidence-based state-gated capital promotion (no fixed dollar ladder)
- §14.1 file:line audit as Phase 0 acceptance prerequisite

Implementation can proceed.

## Status: Convergence Verified (post-CX-revision)

The original draft of this review (sections below marked **RESOLVED**)
critiqued an earlier version of the two CX research documents. CX has
since revised both documents and absorbed substantially all of the
content-side critique. This section records what has converged.

| Original critique | CX resolution | Evidence |
|---|---|---|
| 1. Missing 2024-2025 LLM-driven research frontier | Added `Paper Cluster 7: Modern Frontier Backlog` covering RD-Agent / Kronos / CryptoTrade / GraphSAGE | lit-review §"Paper Cluster 7"; roadmap Phase Crypto-F now references these as shadow/research items |
| 2. Missing hard empirical gotchas | Added `Hard Gotchas To Enforce From Phase Crypto-0` and `Hard Gotchas` section in roadmap | lit-review §"Hard Gotchas To Enforce From Phase Crypto-0"; roadmap §"Hard Gotchas" |
| 3. Asset-class decoupling target unspecified | Adopted `core/ + ashare/ + crypto/` as target architecture with `crypto_` prefix labeled "transitional, not the intended final architecture" | roadmap §"Architecture" |
| 4. No "library over framework" rule | Added as explicit `Engineering decision rule` | lit-review §"Open-source Engineering Review"; roadmap §"Library Over Framework" |
| 5. Funding/OI placement at Phase B0 one phase too late | Moved to Phase Crypto-A alongside OHLCV with explicit rationale: "Funding/OI data belongs in Phase Crypto-A with OHLCV because it is a crypto-native data axis" | roadmap §"Phase A Data: Derivatives / Funding / OI" → "Positioning" |
| 6. DeFi out-of-scope statement missing | Added `DeFi Carve-out` paragraphs | lit-review §"What To Avoid → DeFi carve-out"; roadmap §"DeFi Carve-out" |
| 7. CX's own numeric claims untagged | Added `[paper-reported] / [exchange-dashboard] / [validated-on-local]` three-tier tagging + production rule | lit-review §"Bottom Line → Numeric evidence tags used"; roadmap §"Evidence Tags" |

Net result: 7/7 of the original content-side critiques resolved.
Sections below labeled **RESOLVED** are kept as historical record of
the convergence path; they are no longer active critique.

The section `## Where CC Pushes Back on CX Review of CC Plan` remains
active because it concerns the separate file
`plans/cx-review-cc-crypto-quant-integration-plan-2026-05-30.md`,
which has not been revised.

## Remaining Divergence

After convergence, the following items remain. Each is presented with
the proposed change, the counter-argument CX is likely to raise, and
CC's response with evidence.

### 1. §14.1 file:line audit row enumeration (version-locking)

**Proposed change**: Pin §14.1 of `cc-crypto-quant-integration-plan-2026-05-30.md`
at the commit SHA of the file at Phase Crypto-0 sign-off. Future edits
create §14.2, §14.3, etc. — they do not in-place rewrite §14.1.

**Counter-argument CX may raise**: "Pointer-by-reference is fine.
Pinning a SHA adds maintenance burden for an audit that will obviously
evolve as we touch more files."

**CC response**:

- The CX review of CC plan absorbed §14.1 with the phrasing "this
  should become a future refactor checklist". That phrasing is
  ambiguous about what "future" means — Phase Crypto-0 sign-off?
  Phase Crypto-G physical migration? "Eventually"?
- Concrete failure mode: if Phase Crypto-0 acceptance is "audit
  checklist passed" but §14.1 itself is edited later (a row removed
  because the file got refactored, a row added because a new
  A-share-implicit assumption was discovered), then the Phase
  Crypto-0 sign-off no longer maps to a definite scope. A future
  reader cannot reconstruct what was actually accepted.
- Industry analogue: protobuf `.proto` files are commit-pinned per
  service version for the same reason. Same applies to OpenAPI
  contracts and database migration files. None of these are "obviously
  evolving" — they are versioned precisely because they evolve.
- Maintenance burden is bounded: §14.1 has 20 rows; adding a row
  creates §14.2 (5 lines of diff), not §14.1 mutation. The "burden"
  is one extra section header per audit revision. The audit-trail
  gain is unbounded.

**Compromise position CX may accept**: Tag §14.1 with a frozen
artifact marker (`AUDIT-FROZEN-AT: <commit-sha>` line in the section
header) rather than a separate versioning scheme. Same intent, less
ceremony.

### 2. NautilusTrader deferral rationale (process-model conflict)

**Proposed change**: Add explicit text in both CX research docs and
CX-review-of-CC: "NautilusTrader is structurally incompatible with the
existing cron/parquet/jsonl pipeline. Nautilus is an actor-driven event
loop; this project is a cron-batched IO pipeline. Phase G integration
requires either running both process models in parallel or rewriting
A-share execution to actor. Do not piecemeal-import Nautilus components
in earlier phases."

**Counter-argument CX may raise**: "We already say 'Phase G evaluate
prototype'. That's sufficient gating. The rationale is implementation
detail."

**CC response**:

- "Phase G prototype" gates **production adoption**. It does not gate
  **piecemeal imports** for "research use only". The risk is the
  latter, not the former.
- Concrete piecemeal-import failure scenario:
  - Phase Crypto-D paper OMS developer reads CX docs: "Nautilus has
    well-modeled order semantics, Phase G evaluation, library over
    framework rule."
  - Developer decides: "I'll just import `nautilus_trader.model.orders.OrderType`
    and `nautilus_trader.model.identifiers.InstrumentId` for the schema.
    The cron loop calls a thin wrapper. No actor system needed."
  - Six weeks later: paper OMS has 30 files importing
    `nautilus_trader.model.*`. Phase G arrives with a real
    decision: adopt Nautilus or stay on internal types?
  - If adopt: the installed-base of half-Nautilus code now needs
    to migrate from cron-wrapped fake-async to real Nautilus
    `TradingNode` actors — harder than greenfield, because every
    cron-wrapped call site must be rewritten.
  - If reject: the 30 files of `nautilus_trader.model.*` imports must
    be replaced with internal types. Net cost: months of churn.
- Nautilus docs make this explicit. From the Nautilus architecture
  docs: components are designed to run inside a `TradingNode` actor
  context. Order objects outside that context have undefined live
  behavior; they work for batch backtest only because the backtest
  engine reconstructs the actor context locally. Piecemeal imports
  for live trading violate this contract silently.
- The "library over framework" rule CX adopted addresses *which
  packages to depend on*. It does not address *which packages have
  hidden actor-lifecycle requirements*. Nautilus is a framework that
  also publishes library-shaped types. The risk lives in the type
  imports, not the package selection.

**Cost of adding the rationale**: one paragraph in each doc.

**Cost of not adding it**: 30+ files of phantom Nautilus dependencies
discovered at Phase G.

### 3. CX Internal Inconsistency on Funding/OI Phase Placement

**Observation, not yet proposed change**: CX's review of CC plan
(`cx-review-cc-...md` line 132-136) still recommends:

> "Add `Phase Crypto-B0: Funding/OI Data Foundation`" — between
> Crypto-B (feature cache) and Crypto-C (supervised model).

But CX's revised roadmap (`crypto-quant-roadmap-...md` line 326-331)
now says:

> "Funding/OI data belongs in Phase Crypto-A with OHLCV because it is
> a crypto-native data axis."

These two CX-authored statements contradict each other. A reader
following the review of CC plan would build a B0 phase; a reader
following the revised roadmap would put funding/OI in A.

**Proposed change**: CX should update
`cx-review-cc-crypto-quant-integration-plan-2026-05-30.md` to align
with the revised roadmap — remove the B0 insertion recommendation,
note that the CC plan was right to bundle funding/OI with foundational
data, and clarify that what was rejected was funding-arb-as-strategy,
not funding-data-as-foundation.

**Counter-argument CX may raise**: "The review is a snapshot in time;
later revisions of the roadmap supersede it."

**CC response**:

- Two files in `plans/` with contradicting Phase placements is a real
  hazard for a future reader. The convention in this project's
  `plans/` directory is dated-snapshot but not auto-superseded: any
  reader can pick either file as authoritative.
- The contradiction is also evidence that the original B0 placement
  was wrong on its merits. CX revised the roadmap; the review file
  should be revised symmetrically.
- Cost of update: ~10 lines of diff in the cx-review file.

### 4. Phase Crypto-0 data contract column enumeration

**Status**: RESOLVED. CX roadmap §"Phase A Data: Spot OHLCV" now
enumerates the schema columns
(`timestamp_utc / open / high / low / close / volume_base /
volume_quote / is_closed_bar / ingested_at`). This resolves the
earlier critique. No further action.

### 5. "60% adoption" framing

**Status**: Active. This critique applies to
`cx-review-cc-crypto-quant-integration-plan-2026-05-30.md` (not these
two research docs) and is addressed with full evidence in §"Where CC
Pushes Back on CX Review of CC Plan" below.

### 6. Real-time Data Latency vs T+0 Settlement Architecture (new finding)

**Observation**: Both CC plan §9 and CX architecture sections discuss
settlement model abstraction (`ISettlementModel`) and pluggable
calendar (`Always24x7Calendar`), but neither explicitly separates two
different things that crypto-vs-A-share comparisons routinely conflate:

(a) **Settlement model** — A-share is T+1 (shares bought today cannot
be sold today; OMS needs a `pending_target_weights` overnight lock).
Crypto is T+0/instant (rebalance is full and immediate; OMS does not
need an overnight lock).

(b) **Data tick latency** — A-share research/baseline runs on EOD
parquet. Crypto research/baseline can also run on closed bars
(1h/4h/1d via REST), but the *production execution path* eventually
needs WebSocket-class latency.

The conflation is "crypto needs real-time data because it's T+0".
This is wrong. T+0 is a settlement-model decision that affects OMS
state machine design (must be made at Phase Crypto-0). Real-time
WebSocket data is an execution-latency decision that affects
production deployment (Phase Crypto-G+). They are two decisions on
two timelines.

**Why this matters now**:

- If Phase Crypto-0 designers think "we need real-time data" they
  will over-architect Phase A/B/C (push WebSocket data infra into
  baseline research path), wasting weeks and complicating
  research/replay reproducibility.
- If Phase Crypto-0 designers think "T+0 is a Phase G concern" they
  will under-architect Phase D paper OMS (build it with A-share T+1
  semantics and discover at Phase E that funding-arb rebalancing
  cannot work overnight-locked).

**Counter-argument CX may raise**: "This is implicit in
`ISettlementModel` being separate from `MarketDataSource`. We
already separated them as orthogonal interfaces."

**CC response**:

- Orthogonal in code != orthogonal in mental model. The CX docs do
  not contain the sentence "T+0 architecture must be designed at
  Phase 0 even though real-time data is deferred to Phase G". A
  developer reading the docs sequentially will see Phase 0 "data
  contract" and assume both settlement and data-latency are bundled.
- Concrete failure scenario: Phase Crypto-D paper OMS developer reads
  CX docs, sees `Always24x7Calendar` and "no T+1 in core", builds
  paper OMS with T+0 settlement, but uses A-share-style "fill at next
  daily close" simulation because nothing says daily close model is
  wrong for crypto. Result: paper OMS simulates 24h holding for every
  trade. Funding-arb strategy (which needs 8h holding alignment to
  funding settlement) breaks silently.
- Two-line addition fixes it:
  - "**T+0 settlement** is a Phase Crypto-0 requirement (affects
    OMS state machine, no `pending_target_weights` lock)."
  - "**WebSocket-class real-time data** is a Phase Crypto-G+
    requirement (Phase A through F run on REST closed bars)."

**Proposed addition**: Add a one-paragraph "Settlement vs Data
Latency" subsection to Phase Crypto-0 acceptance criteria in CX
roadmap, listing the two as separate decisions with explicit phase
gates.

This also clarifies a question that comes up early in any
crypto-vs-A-share discussion: "Do we need real-time data?" Answer:
not for Phase A-F. But T+0 settlement architecture must be designed
from Phase 0, regardless.

## Overall Judgment (Original — pre-convergence, kept for record)

> [RESOLVED — superseded by the convergence table above. Kept for
> traceability.]

The CX documents are the right backbone for crypto integration into this
project. They correctly prioritize infrastructure discipline, identify the
real gaps in the current codebase, and set conservative phase gates.

But they are also a snapshot of the field circa 2022 with 2024 add-ons. They
miss the 2024-2025 LLM-driven research frontier, omit several hard
empirical gotchas that are easy to step on if not flagged upfront, and
under-specify the asset-class decoupling problem that the existing A-share
codebase poses.

Best use:

- Treat them as the canonical phasing plan and infrastructure punch list.
- Adopt the Phase Crypto-0/A/B/.../H naming convention as the project's
  shared vocabulary.
- Use the RiskGuard-before-alpha philosophy as the default mindset.

Do not use them as:

- A literature review of the modern frontier — the 2024-2025
  LLM-research-assistant and foundation-model line is missing.
- A final list of crypto factors — several factors have empirical sign
  flips relative to A-share that are not surfaced.
- A decoupling design — the docs are silent on whether crypto code lives
  in a parallel namespace or under a `crypto_` prefix inside existing dirs.

Recommended adoption level:

- Adopt the phasing skeleton and infrastructure priorities at ~95%.
- Adopt the factor catalogue at ~70% — needs sign-flip warnings and
  modern additions.
- Adopt the engineering tool selection at ~80% — needs an explicit
  "library over framework" rule.

## What Is Worth Absorbing

### 1. Phase Naming and Acceptance Criteria

`Phase Crypto-0 → A → B → C → D → E → F → G → H` with explicit acceptance
criteria per phase is the single most useful contribution. This is missing
from the CC plan.

Specifically valuable:

- Phase Crypto-0 as "data contract + architecture interfaces" before any
  strategy work.
- Each phase has measurable exit criteria, not just "done".
- Phase Crypto-E (perp RiskGuard) is correctly gated behind Crypto-D
  (paper OMS), not before.

CC plan adoption:

- Replace CC §14 "25-step refactor sequence" with the CX phase naming.
- Keep CC §14.1 (file:line asset-implicit audit) as an artifact under
  Phase Crypto-0.

### 2. "Boring Infrastructure Sprint First" Philosophy

CX's emphasis on doing the unsexy work — data contract, calendar
abstraction, cost models, sanitizer — before any alpha work is correct
and matches how the A-share side of this codebase was built.

This is structurally protective against the failure mode of "shipping
strategies on top of leaky infrastructure".

### 3. Conservative RiskGuard Boundaries

CX's risk numbers (max single-asset weight, max sector concentration,
crash gate thresholds, funding-rate clamps) are conservative in a way
that matches the A-share RiskGuard 7-layer design already in production.

These should be the defaults, not the CC plan's more aggressive numbers
in §6.

### 4. Cron Cadence Design

CX correctly specifies `*/30` for 1h bars and `*/4 hours` for 4h bars,
which respects the bar-close + late-data window discipline that A-share
EOD cron uses. CC plan was silent on this.

### 5. Current-project Gap List

CX's enumeration of crypto-specific gaps in the current project
(`data/collectors/crypto_market.py` missing, no `paper/crypto_oms.py`,
no `factors/crypto/`, no `backtest/crypto/`) is more complete than the
CC plan's enumeration. It should become the Phase Crypto-0 backlog
verbatim.

## What Should Not Be Accepted As-is (RESOLVED — pre-convergence)

> [All items below RESOLVED by CX revision. See convergence table at
> top. Kept for traceability of how the documents arrived at
> consensus.]

### 1. Missing Modern Research Frontier (2024-2025) — RESOLVED

The CX literature review covers:

- 2017-2023 systematic reviews
- Liu-Tsyvinski-Wu factor models (2021-2022)
- Fieberg CTREND (2024)
- Bianchi-Babiak IPCA
- Chi 2025 on-chain flow

But it omits the entire 2024-2025 line of LLM-driven research:

- **RD-Agent (Microsoft Research, 2024)** — autonomous factor research
  loops, directly applicable to crypto factor zoo expansion.
- **Kronos (2024)** — foundation model for financial time series, has
  open weights and works on crypto OHLCV out of the box.
- **GraphSAGE / on-chain GNN (2024-2025)** — inductive embedding of
  address graphs, relevant for whale-flow features beyond simple
  exchange net-flow.
- **CryptoTrade (NeurIPS 2024)** — LLM-as-trader benchmark with
  reproducible test bench.

These are not toys. They are the active research frontier and at least
two (Kronos, CryptoTrade) ship usable code.

CC recommendation:

- Add a Phase Crypto-F+ research backlog item: "evaluate Kronos as a
  shadow predictor on crypto OHLCV".
- Add to Phase Crypto-F (event/on-chain overlays): "evaluate GraphSAGE
  for address-graph embeddings as alpha source".

### 2. Missing Hard Empirical Gotchas — RESOLVED

The CX literature review surfaces factors but does not surface the
sign-flip and dead-strategy gotchas that will burn capital if not
flagged in writing. These are not minor:

1. **IVOL has POSITIVE sign in crypto** (Zhang-Li 2020). A-share code
   has IVOL with negative sign. Copy-pasting will reverse the alpha.
2. **MAX has POSITIVE sign in crypto** (Li et al. 2021, lottery
   momentum). Same trap as IVOL.
3. **Traditional Value (B/P) is dead in crypto** — no cash flows.
   On-chain value proxies have t-stat ~2 at best.
4. **BTC-ETH cointegration is structurally broken since 2022 Merge** —
   correlation dropped 0.95 → 0.75 within 47 days, ETH/BTC trended 47%.
   Any pairs strategy on majors will fail.
5. **Survivorship bias is +62.19% (Ammann 2023)** — equal-weight
   crypto backtests without dead-coin reinclusion overstate alpha by
   62%. Must keep last-bar of delisted tokens.
6. **Alpha decay is 5-10x faster than A-share** — factor IC evaluation
   windows must shrink from 60-120D to 7-30D; walk-forward retrains
   from monthly to weekly.

CC recommendation:

- Add a new section "Hard Gotchas" to the CX literature review.
- Make these warnings part of Phase Crypto-0 acceptance: any factor
  imported from A-share gets a forced sign check.

### 3. Missing Asset-Class Decoupling Design — RESOLVED

The CX roadmap proposes paths like:

- `data/collectors/crypto_market.py`
- `paper/crypto_oms.py`
- `factors/crypto/`
- `backtest/crypto/`

This embeds crypto as a `crypto_` prefix inside the existing A-share
directories. It does not address:

- A-share-implicit logic in shared modules (`paper/oms.py` T+1,
  `data/build_tradable_mask.py` ST filter, hardcoded 100-share lot,
  stamp tax, SH/SZ/BJ prefix assumptions).
- Whether shared `core/` interfaces should be lifted out.
- How a future "third asset class" (HK / US equities) would slot in.

CC plan §9 proposed `core/ + ashare/ + crypto/` three-namespace
architecture with explicit Protocol interfaces. CX's review correctly
flagged that immediate large-scale physical migration is too risky
right now, and the CC plan accepts that.

But the CX documents are silent on whether `core/ + ashare/ + crypto/`
is the eventual destination or whether `crypto_` prefix in existing
dirs is the permanent design.

CC recommendation:

- The CX roadmap should explicitly adopt `core/ + ashare/ + crypto/`
  as the **target** architecture, even if physical migration is gated
  to Phase Crypto-G+.
- Phase Crypto-0 should add adapters/facades that prepare for that
  target without moving files.

### 4. No "Library Over Framework" Rule — RESOLVED

The CX engineering review evaluates:

- FreqAI
- NautilusTrader
- Hummingbot
- FinRL
- Qlib
- vectorbt

It does not articulate the meta-principle. CC plan made this explicit:
**prefer libraries (CCXT, vectorbt, Polars, DuckDB) over frameworks
(freqtrade, Hummingbot, LEAN, Jesse) for the main pipeline**.

Reasoning:

- Frameworks impose a process model — config, plugin lifecycle, event
  bus — that conflicts with the existing cron + jsonl + parquet
  pipeline.
- Libraries are composable; this project already won by being a
  composition of pandas + qlib + lightgbm rather than a framework.
- The exception is NautilusTrader for exchange-realistic execution
  semantics, which the CX review correctly defers to Phase Crypto-G.

CC recommendation:

- Add "library over framework" as an explicit decision rule in CX's
  Phase Crypto-A.

### 5. Funding Arb as Phase 0 Research Track, Not Just "Later Paper" — RESOLVED

CX's review of CC plan correctly downgraded funding arb from "first
production strategy" to Phase Crypto-C/D paper. CC plan accepts this.

But CX is too conservative in placing funding/OI data collection at
Phase Crypto-B0 (after spot OHLCV foundation in Phase Crypto-A).
Funding/OI is just as fundamental as OHLCV — it is the crypto-native
data axis with no A-share analogue. A model that does not have
funding/OI from day one will miss the dominant crypto risk-carry
signal.

CC recommendation:

- Promote funding/OI data collection from Phase Crypto-B0 to Phase
  Crypto-A alongside OHLCV.
- Keep funding-arb-as-strategy at Phase Crypto-C/D paper.

### 6. DeFi Carve-out Should Be Acknowledged, Not Ignored — RESOLVED

CX's review of CC plan explicitly rejected DeFi base yield (Aave / Sky
/ Morpho / Curve / HYPE points) as not-a-quant-problem. CC plan accepts
that the original 60/30/10 allocation overreached.

But the CX research docs are silent on DeFi entirely. There is no
"out of scope" statement, no separate research note pointer.

For a multi-million-dollar capital pool, DeFi yield is a real
benchmark that any crypto quant book will be measured against (if my
funding-arb Sharpe < my Aave-USDC carry, the quant book is failing).

CC recommendation:

- Add a one-paragraph "DeFi out of scope" carve-out to the CX roadmap
  that explicitly says: DeFi yield will be tracked as a benchmark in a
  separate capital-management note, not as part of the quant pipeline.
- This is consistent with both CX review position and the user's
  multi-asset reality.

### 7. Numeric Claims Need Explicit Hypothesis Tags — RESOLVED

CX correctly criticized the CC plan for numeric claims (92% positive
funding, Sharpe 2-3, etc.). But CX's own documents contain similar
unbacked numbers:

- "周 L/S ≈ 3%" for CMOM
- "周 3.87%" for CTREND
- "t > 3" significance

These are paper-reported and not validated against this project's data.
They should be tagged as `[paper]` versus `[validated on local data]`
in writing.

CC recommendation:

- Convert all numeric claims in both CX docs to one of three tags:
  `[paper-reported]`, `[exchange-dashboard]`, `[validated-on-local]`.
- Phase Crypto-0 acceptance: any number used for sizing or risk decision
  must be `[validated-on-local]` before that decision ships.

## Recommended Synthesis

Merge path (preserving CX phase naming, layering CC additions):

```text
Crypto-0: data contract + architecture interfaces + asset-implicit audit
          (CX phase + CC §14.1 file:line audit + core/ Protocol interfaces)
Crypto-A: spot OHLCV + funding/OI data foundation
          (CX Crypto-A merged with CX Crypto-B0 — both are foundational)
Crypto-B: feature cache + fee-aware baseline + sign-flip checks
          (CX phase + CC hard-gotcha enforcement)
Crypto-C: supervised XGB/LGB ranking, weekly retrain
          (CX phase + CC alpha-decay-aware window sizing)
Crypto-D: paper OMS with crypto Sanitizer
          (CX phase + CC 12-rule Sanitizer)
Crypto-E: perp RiskGuard + paper funding-arb strategy
          (CX phase, no change)
Crypto-F: event/on-chain shadow overlays + Kronos shadow predictor
          (CX phase + CC modern frontier addition)
Crypto-G: Nautilus execution prototype + physical core/ashare/crypto/ migration
          (CX phase + CC namespace destination)
Crypto-H: RL allocation/risk controller, only if Crypto-C through F
          show ICIR > 0.4 and stable Sharpe > 1.5 paper-validated
```

## Concrete Items To Absorb Into CC Plan (Self-correction)

After reading CX's review of CC, the following CC plan items should be
revised:

- §6 Capital schedule: replace explicit dollar tiers with state-gated
  promotion (signal-only → paper → testnet → tiny canary).
- §3-4 DeFi 60/30/10 allocation: remove from CC plan, move to a
  separate `capital-management-note.md`.
- §9 Namespace migration: keep `core/ + ashare/ + crypto/` as **target**,
  add explicit phase gate to Crypto-G+, prohibit moving A-share files
  before then.
- §10 NautilusTrader stance: change from "do not use" to "not Phase A
  dependency, Phase G evaluate prototype".
- Funding arb: change framing from "first production strategy" to
  "Phase C/D paper, Phase G+ canary post 30-day stable Sharpe".

## Where CC Pushes Back on CX Review of CC Plan

The CX review of CC plan is mostly correct (capital schedule, DeFi
scope, migration timing, NautilusTrader absolutism, numeric claim
revalidation). CC plan accepts those revisions. But the following
specific points in the CX review of CC are either undercounting,
miscategorizing, or silent on issues that matter.

### 1. "Adopt About 60%" Undercounts the Structural Material

CX review headlines:

> "Recommended adoption level: Adopt about 60%."

Evidence-based recount:

| CC plan section | Content | CX review verdict | Adoption % |
|---|---|---|---|
| §9 Architecture (10 principles + Protocols) | AssetClass/InstrumentClass, Calendar injection, ISettlementModel, CommissionModel/TaxModel/ImpactModel, venue-aware Symbol, lot/tick on instrument, multi-currency ledger | "What Is Worth Absorbing §1" lists 7/10 verbatim | ~70% |
| §10 Not-do List (10 items) | HFT/MEV/memecoin/Telegram/autonomous LLM/LOB transformer/Uniswap v3 LP/restaking/BTC-ETH coint. | "What Is Worth Absorbing §4" accepts 8/10, adds none | ~80% |
| §11 CryptoSanitizer (12 rules) | listed days / USD volume / depeg / funding extreme / withdrawal halt / unlock / scam list / wick / inflow / depth / cross-exchange / chain congestion | "What Is Worth Absorbing §3" lists 12/12 | ~100% |
| §14.1 Asset-implicit audit (20 file:line rows) | Hardcoded ST / limit-board / 100 share lot / stamp tax / SH/SZ/BJ / Qlib CN / T+1 / A-share defaults | "What Is Worth Absorbing §2" accepts wholesale | ~100% |

Weighted average over the four structural sections: **~88%**, not 60%.

Where the 60% comes from: CX is averaging in rejections of four
execution-pace items (funding-arb-first, capital schedule, DeFi base
yield, immediate migration). Those are pace items, not content items.
Rejecting "do it next week" while accepting "this is the right
architecture" is not a content rejection.

CC pushback: the headline number should be split into two: structural
adoption ~90%, execution-pace adoption ~30%. Otherwise downstream
readers will under-invest in the structural absorption.

**Counter-argument CX may raise**: "It's a one-line headline. Exact
percentage is precision theater. Readers will read the body anyway."

**CC response**:

- Empirically, headline numbers drive resource allocation. A reader
  who sees "60%" budgets less time for absorption than a reader who
  sees "90% / 30%". The split is information, not decoration.
- Precision theater concern is dismissable here because the split
  reflects a real categorical distinction (content vs pace), not
  spurious precision on a continuous quantity.
- Cost of fix: change one line to two. "Structural content: ~90%
  adopt. Execution pace: ~30% adopt — see Reject/Delay lists below."

### 2. §14.1 Is Not a "Future Refactor Checklist" — It Already Exists

CX review says (line 82):

> "This should become a future refactor checklist."

Evidence: CC plan §14.1 is already a 20-row table with `file:line`,
diff size estimate, and A-share regression risk rating per row. It is
not a placeholder. It is the artifact.

CC pushback: the CX phrasing implies §14.1 is aspirational. It is not.
The correct phrasing is "should become Phase Crypto-0 acceptance
criteria checklist, with each row gated by a regression smoke test".
This is materially different from "future".

**Counter-argument CX may raise**: "'Future' was shorthand for 'when
we get to the refactor phase'. Same intent."

**CC response**:

- Shorthand ambiguity is precisely the failure mode. "Future" admits
  three readings: (i) Phase Crypto-0 acceptance, (ii) Phase Crypto-G
  physical migration, (iii) "eventually, no specific gate". Each
  produces different behavior:
  - (i) → developer treats §14.1 as a Phase 0 blocker, gates on
    regression smoke tests now.
  - (ii) → developer defers §14.1 reading until physical migration
    starts, by which time the audit may have rotted.
  - (iii) → §14.1 becomes wallpaper.
- The fix is one word: "future" → "Phase Crypto-0 sign-off
  prerequisite". Removes the three-way ambiguity.
- This pairs with §"Remaining Divergence" item 1 (version-locking):
  treating §14.1 as Phase 0 prerequisite implies pinning it at Phase
  0 sign-off.

### 3. Phase Crypto-B0 Placement of Funding/OI Is One Phase Too Late

CX review proposes:

> "Add `Phase Crypto-B0: Funding/OI Data Foundation`" — between
> Crypto-B (feature cache) and Crypto-C (supervised model).

Evidence against this placement:

- He 2024 (funding arb paper) reports funding rate alone explains
  30-40% of perp return variance across BTC/ETH/SOL during
  2022-2024. A supervised model trained without funding from Phase A
  is missing the dominant risk-carry axis on day 1.
- Adding funding at Phase B0 (after feature cache exists) means the
  feature cache must be re-architected to admit funding/OI columns,
  re-validated, and the Phase C supervised model retrained from
  scratch. Two phases of rework versus one phase of correctness.
- Funding/OI has no A-share analogue. It is the most
  crypto-distinctive data axis. Treating it as a "later" addition
  inverts the design priority — the crypto-native axis should be
  upstream of generic OHLCV factors that just port from equities.

CC pushback: move funding/OI to Phase Crypto-A alongside OHLCV. Keep
the funding-arb-as-strategy gating at Phase C/D paper. Data
collection and strategy are not the same decision.

**Counter-argument CX may raise**: "We deliberately separated spot
foundation from derivatives complexity. Spot collectors and
derivatives collectors have different exchange endpoints, different
rate limits, different failure modes. Bundling them at Phase A
overloads the foundation."

**CC response**:

- Spot and funding are independent data streams that share an
  ingestion abstraction. They do not couple at the collection layer
  — `CCXTSpotCollector` and `CCXTPerpCollector` are two classes
  hitting different endpoints. Adding both to Phase A is two
  collectors in one phase, not one tangled collector.
- The collection-layer separation argument is exactly why bundling
  is safe: there's no shared rate-limit pool, no shared parquet
  schema, no shared error path.
- The cost of *not* bundling at Phase A: Phase B feature cache is
  designed for spot-only columns. Phase B0 then has to mutate the
  cache schema to add funding/OI columns. Schema mutations on a
  feature cache require full reproducibility re-validation —
  expensive and risky.
- **CX has already updated the revised roadmap to put funding/OI in
  Phase A** (roadmap §"Phase A Data: Derivatives / Funding / OI" /
  "Positioning"). The review-of-CC file is now inconsistent with
  CX's own revised roadmap. The simplest fix is to align the
  review-of-CC file with the revised roadmap. See §"Remaining
  Divergence" item 3 for the alignment request.

### 4. CX Documents Are Silent on `crypto_` Prefix vs `core/+ashare/+crypto/` Target

CX review of CC plan correctly says immediate migration is too risky.
CC plan accepts that.

But CX's own roadmap document repeatedly proposes paths like
`data/collectors/crypto_market.py`, `paper/crypto_oms.py`, and
`factors/crypto/` without stating whether this is:

(a) the **permanent** design — crypto code lives as a `crypto_` prefix
or `crypto/` subdir inside existing A-share dirs, or

(b) the **transitional** design — pending eventual extraction into
parallel `crypto/` package once A-share production stabilizes.

These two are not the same. (a) means shared logic (OMS, RiskGuard,
optimizer) is permanently A-share-flavored with crypto branches
inside. (b) means a future Phase Crypto-G migration that swaps
A-share imports for `ashare.` and `crypto.` imports against `core.`
interfaces.

Evidence the ambiguity matters: the CC plan §14.1 audit identified 20
specific A-share-implicit code paths inside shared modules. Under
(a), each of those becomes an in-place `if asset_class == "crypto":`
branch — and the branch count grows with every new feature. Under
(b), each becomes a `core.IOMS` Protocol with two implementations.

CC pushback: CX should pick one, in writing. CC recommendation is (b)
with the migration gated to Phase Crypto-G+. Either picking (a) or
leaving it unresolved leads to invisible drift toward in-place
branches that compound the asset-implicit problem instead of solving
it.

**Counter-argument CX may raise**: "We said 'physical migration
delayed'. That implies (b) without committing."

**CC response**:

- "Delayed" does not imply "destination". A reader sees "delayed"
  and infers: "we'll do physical migration eventually if it seems
  worth it, otherwise we'll stay on `crypto_` prefix". That's
  three possible end states, not one.
- The decision principle determines daily behavior even before
  migration happens. Under (b), every shared-module edit that
  touches A-share-implicit logic must add a Protocol method to
  `core/`, not an in-place `if asset_class == "crypto":` branch.
  Under (a), the branches accumulate freely. The difference shows
  up six months before any physical move.
- Evidence the difference matters: CC §14.1 audit found 20
  A-share-implicit code paths in shared modules. If the project is
  on path (a), each new feature that touches one of those paths
  adds another asset-class branch — `if asset_class == ...` count
  monotonically grows. If on path (b), each touched path becomes a
  `core.IModel` Protocol — the count of branches stays bounded.
- **CX has already updated the revised roadmap with explicit text**:
  "The `crypto_` file names in the early implementation plan are
  transitional, not the intended final architecture" (roadmap
  §"Architecture"). This is path (b). The CC pushback is now
  partially resolved by CX's revised roadmap; the review-of-CC file
  should be updated to reflect this same commitment.

### 5. "Phase Crypto-0 Data Contract" Is Underspecified

CX roadmap says Phase Crypto-0 produces "data contract + architecture
interfaces" but does not enumerate the contract.

Evidence: CC plan §7 specifies the bar schema explicitly:

- `ts_utc` (int64, milliseconds since epoch)
- `open / high / low / close / volume` (float64)
- `quote_volume` (float64)
- `trades` (int32)
- partition: `(instrument, ts_utc // 86400000)`

Without this enumeration, Phase Crypto-0 cannot have a verifiable
acceptance criterion. Two implementers will produce two incompatible
parquets and the issue won't surface until Phase Crypto-B feature
cache wiring.

CC pushback: CX's Phase Crypto-0 should incorporate CC §7's explicit
schema, dtype, and partition rules. "Data contract" without column
names is not a contract.

**Counter-argument CX may raise**: "Phase A includes the schema
enumeration. Phase 0 just gates on 'design exists'."

**CC response**:

- Phase 0 sign-off cannot reference a Phase A artifact. By
  definition, Phase A starts *after* Phase 0 sign-off. If Phase 0
  acceptance reads "data contract designed" and the actual schema
  lands in Phase A, Phase 0 sign-off is acceptance of an
  unspecified contract. Two implementers can pass Phase 0 with
  different contracts in mind, and the conflict only surfaces in
  Phase B.
- The fix is moving schema enumeration *into* Phase Crypto-0
  acceptance criteria, not deferring it to Phase A. Phase A then
  becomes "implement the Phase 0 contract", which is testable.
- **CX has updated the revised roadmap to enumerate the schema**
  (`timestamp_utc / open / high / low / close / volume_base /
  volume_quote / is_closed_bar / ingested_at`) but the enumeration
  is in §"Phase A Data: Spot OHLCV", not §"Phase Crypto-0
  acceptance". The remaining ask is just to mark this enumeration
  as a Phase 0 deliverable, not Phase A.

### 6. NautilusTrader Demotion to Phase G Is Right but the Rationale Is Wrong

CX review says (line 244-249):

> "Phase A/B: no Nautilus dependency. Phase G: evaluate Nautilus
> prototype."

CC agrees with the timing. But the CX rationale ("exchange-realistic
backtests, live/testnet parity, order/account semantics") is
incomplete. The real reason to defer Nautilus is structural:

- Nautilus imposes its own event-loop, message bus, and actor
  lifecycle. The existing A-share pipeline is cron-driven with
  parquet/jsonl IO. Mixing the two is a process-model conflict, not a
  feature gap.
- Adopting Nautilus as a Phase A dependency forces the project to
  either run two process models in parallel (cron + actor) or
  rewrite A-share to actor-based.

CC pushback: the deferral rationale should be "process-model
conflict with existing cron pipeline", not just "exchange-realistic
backtest can wait". The latter framing leaves room for someone to
import Nautilus piecemeal and hit the conflict the hard way.

**Counter-argument CX may raise**: "Deferral is deferral. The reason
is a footnote."

**CC response**:

- See §"Remaining Divergence" item 2 for the full piecemeal-import
  failure scenario and Nautilus actor-context evidence.
- Short version: "Phase G evaluate prototype" gates *production
  adoption*. It does not gate *piecemeal imports of
  `nautilus_trader.model.*` types for "research use only" in earlier
  phases. The risk lives in the type imports, not the package
  selection. The rationale matters because it identifies which
  imports are dangerous, not just which package is dangerous.
- Cost: one paragraph addition. Benefit: prevents 30+ files of
  half-Nautilus code accumulating before Phase G.

### 7. "Several Numeric Claims Need Revalidation" Cuts Both Ways

CX review flagged CC's numeric claims (92% positive funding, Sharpe
2-3, etc.) and asked for revalidation. CC accepts that.

But CX's own roadmap and literature review contain equally unbacked
numbers that CX does not flag in its own review:

- "周 L/S ≈ 3%" for CMOM (paper-reported, no local validation)
- "周 3.87%" for CTREND (paper-reported)
- "t > 3" significance claims (paper-reported)
- "等权偏误 62.19%" (Ammann 2023, applied to crypto universes the
  project does not yet hold)

CC pushback: the revalidation discipline should be symmetric. CX
should apply the same tagging requirement to its own numbers. CC
proposes `[paper-reported]` / `[exchange-dashboard]` /
`[validated-on-local]` as the three tags, applied to every number on
both sides.

**Counter-argument CX may raise**: "We already added the tagging
system to the revised research docs."

**CC response**:

- Correct on the research docs. The revised lit-review and roadmap
  both have the three-tier tagging and the production rule. That's
  resolved.
- The remaining gap is in the cx-review-of-CC file itself: it still
  contains untagged paper-reported numbers (CMOM 3% / CTREND 3.87%
  / t > 3 / Ammann 62.19%) inside its own arguments. The review
  file should be retagged to be consistent with the revised research
  docs.
- Asymmetric application of the tagging rule weakens its enforcement
  authority. If the review file is exempt, the rule becomes
  "tagging applies to other people's docs, not ours".

## Concrete Items To Absorb Into CX Plan

- Add modern frontier section: RD-Agent, Kronos, GraphSAGE, CryptoTrade.
- Add Hard Gotchas section: IVOL/MAX sign flip, BTC-ETH dead, B/P dead,
  survivorship +62%, alpha decay 5-10x, regime overlay fail.
- Add `core/ + ashare/ + crypto/` as target architecture, even with
  delayed physical migration.
- Add "library over framework" decision rule for engineering choices.
- Add DeFi out-of-scope carve-out paragraph.
- Add numeric claim tagging: `[paper-reported]` vs `[validated-on-local]`.
- Promote funding/OI data from Phase Crypto-B0 to Phase Crypto-A
  alongside OHLCV.

## Final CC Position (post-convergence)

CX's two documents, in their revised form, are the right backbone and
the right content. The 7 content-side critiques in the original draft
of this review are all resolved.

What is now settled across both sides:

- Phase Crypto-0/A/B/.../H naming with explicit acceptance criteria
  per phase.
- "Library over framework" as the engineering rule.
- `core/ + ashare/ + crypto/` as target architecture, with physical
  migration gated to Phase Crypto-G+.
- `crypto_` prefix paths in early phases are transitional, not final.
- Funding/OI data is Phase Crypto-A alongside OHLCV; funding-arb as a
  *strategy* remains gated to Phase Crypto-C/D paper, Phase Crypto-G+
  canary.
- Numeric claims use `[paper-reported] / [exchange-dashboard] /
  [validated-on-local]` tagging; sizing/risk decisions require
  `[validated-on-local]`.
- Hard Gotchas (IVOL/MAX sign, B/P dead, BTC-ETH coint dead,
  survivorship +62%, alpha decay 5-10x) enforced from Phase Crypto-0.
- Modern frontier (RD-Agent / Kronos / CryptoTrade / GraphSAGE)
  evaluated as shadow/research backlog at Phase Crypto-F.
- DeFi yield is out of scope for the quant pipeline; tracked
  separately as capital-management benchmark.
- CC plan accepted: capital schedule trimmed, DeFi removed from
  mainline, migration delayed.

What remains on the CC side to revise (self-correction):

- CC plan §6 capital schedule: replace explicit dollar tiers with
  state-gated promotion (signal-only → paper → testnet → tiny canary
  → scale after 30-day stable Sharpe).
- CC plan §3-4 DeFi 60/30/10 allocation: remove from CC plan, move
  to separate `capital-management-note.md`.
- CC plan §9 namespace: keep `core/ + ashare/ + crypto/` as **target**,
  add explicit phase gate to Crypto-G+, prohibit moving A-share files
  before then.
- CC plan §10 NautilusTrader stance: change "do not use" to "not Phase
  A dependency, Phase G evaluate prototype, deferred because of
  process-model conflict with cron pipeline".
- CC plan funding arb framing: "first production strategy" → "Phase
  C/D paper, Phase G+ canary post 30-day stable Sharpe".

Minor items remaining on CX side (low priority, see §"Remaining
Divergence" above): NautilusTrader deferral rationale could be made
explicit, §14.1 audit reference could be version-locked at Phase
Crypto-0 sign-off.

Near-term priority is now identical on both sides:

```text
crypto data contract + architecture interfaces
→ clean OHLCV + funding/OI data
→ fee-aware baseline + sign-flip checks
→ supervised model with weekly retrain
→ paper OMS + Crypto Sanitizer
→ derivatives RiskGuard
→ event/on-chain + Kronos shadow overlays
→ Nautilus prototype + physical namespace migration
→ RL allocation only if justified
```
