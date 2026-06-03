# Phase Crypto-0 Sign-off — 2026-06-03

**Status**: CLOSED ✅
**Date**: 2026-06-03
**Branch on which closure was performed**: `master`
**Authority for sign-off**: this document, paired with
`plans/crypto-data-contract.md` (data contract) and
`plans/crypto-dev-phases.md` (single-source roadmap).

## Acceptance criteria — verified

Per `plans/crypto-dev-phases.md` Phase Crypto-0 section:

| Criterion | Status | Evidence |
|---|---|---|
| `plans/crypto-data-contract.md` exists | ✅ | 13 sections + new §1.5 ssproxy, in repo |
| `data/storage/crypto/README.md` exists | ✅ | Sentinel doc forbidding writes here, points to `/Volumes/DATA/crypto/` |
| Universe list with justification | ✅ | Contract §10: BTC/ETH/SOL/BNB/XRP, USDT spot majors on Binance, with 7-prereq expansion gate |
| Hard-gotcha checklist | ✅ | Memory [[crypto-quant-research-20260530]] §"Hard Gotchas" + roadmap §"Hard Gotcha Catalogue" (11 items, [5/30] + [6/3 Δ] tagged) |
| Numeric assumptions tagged | ✅ | Contract §11 inline tags + `plans/numeric_claims_audit.md` (4-tier evidence registry, 10051 bytes) |
| UTC schema reviewed | ✅ | Contract §3 / §4 / §5 schemas all UTC-anchored; user reviewed roadmap doc earlier today |
| ssproxy mandatory documented | ✅ | Contract §1.5 added today: cron wrapper + collector contract + lint test all specified |
| No A-share imports in crypto-namespace files | ✅ trivially | No new crypto-namespace files exist yet — Phase A creates them; lint will be added as a Phase 0b deliverable |
| A-share cron 5/5 GREEN today | ✅ | Soak check 2026-06-03: `[A-share daily jobs] GREEN`, all 5 required jobs success |
| Crypto quarantine PR merged to master | ✅ | Commit `a415605` (Merge crypto quarantine to master, soak Day 1+2 GREEN) |

## What this phase deliberately did NOT do

Crypto-0 is **pure documentation**. The following are explicitly out
of scope and remain as Phase 0b / Phase A deliverables:

- `config/crypto_storage.py` — runtime module that resolves
  `/Volumes/DATA/crypto/` and refuses repo fallback.
- `config/crypto_network.py` — runtime `assert_proxy_active()` and
  network profile activation.
- `config/crypto_universe.py` — runtime constant for Phase A symbol
  list.
- `data/collectors/crypto_market.py` / `crypto_derivatives.py` — actual
  collectors using CCXT.
- Any cron entry routing through `--network crypto`.
- Any `tests/test_crypto_namespace_isolation.py` enforcing import
  contract.

These are Phase A code. Crypto-0 just locks the policy boundaries so
Phase A writes the implementation against an already-frozen contract.

## Reading order for the next phase

When Phase Crypto-A starts:

1. **`plans/crypto-dev-phases.md`** — what to build and the
   acceptance gates. Read first.
2. **`plans/crypto-data-contract.md`** — schemas, paths, network
   policy, universe. Read before writing any file.
3. **`plans/numeric_claims_audit.md`** — every numeric default that
   ends up in code must have an evidence tag here.
4. **Memory** — [[crypto-quant-research-20260530]] for paper-anchor
   conclusions; [[crypto-dev-phases]] for high-level pointer.

The 8 underlying 5/30 plan docs (`cc-crypto-*`, `cx-*review*`, etc.)
are now **archival**: do not edit, do not cite as authoritative.
The roadmap + data contract + numeric audit + this sign-off form the
canonical Crypto-0 deliverable bundle.

## Audit-Frozen-At marker

This sign-off should be paired with a commit SHA in
`plans/crypto-data-contract.md`'s Sign-off section once committed.

When that commit lands, replace the `<commit-sha>` placeholder in the
contract's Sign-off line with the actual SHA from the merge commit
that closes Crypto-0.

## Next gate

Phase Crypto-A may begin when:

1. ✅ Crypto-0 sign-off committed (this doc + contract §1.5).
2. ⏳ User explicitly signals "开 Crypto-A".
3. ⏳ A-share Batch 3 step B (train_lgb FeatureMerger fix) merge
   decision made — either merged after backtest validation, or
   explicitly deferred. The branch `fix/train-lgb-use-feature-merger`
   currently holds it pending.
4. ⏳ Prefilter PR (`fix/llm-prefilter-dedup`) merge decision made —
   user said earlier this should go through independent review
   before merging.

None of the ⏳ items block Crypto-A code work technically; they
matter only because (3) and (4) affect A-share behaviour and we
preserve isolation by not having outstanding A-share PRs while
crypto code is being written.

— Sign-off recorded 2026-06-03 by the working session.
