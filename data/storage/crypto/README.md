# Crypto Storage — Sentinel / Pointer Only

**Real path**: `/Volumes/DATA/crypto/`
**Do NOT write data into this directory.**

## Why this README exists

`plans/crypto-data-contract.md` §1 mandates that all crypto data
(OHLCV, funding, open interest, on-chain, derived features, predictions,
paper trades, reports) lives on the external volume
`/Volumes/DATA/crypto/`, not in the repo tree.

This `data/storage/crypto/` directory exists in the repo solely as a
**guard sentinel** — its presence makes accidental writes here easier
to detect via a CI lint and via the runtime check below.

## Rules

1. **No data files in this directory.** Only this README.
2. Any module that needs to write crypto data calls
   `config.crypto_storage.crypto_root()` (Phase 0b deliverable).
   That function returns a `Path` rooted at `/Volumes/DATA/crypto`,
   raising `RuntimeError` if the volume is not mounted.
3. The CI lint that rejects `data/storage/crypto/...` path literals
   in code is part of `tests/test_crypto_namespace_isolation.py`
   (Phase 0b deliverable).
4. Reading the real path requires the volume to be mounted. Cron
   jobs that need the volume fail fast (and silently to A-share) if
   the volume is unavailable.

## Why not just put `data/storage/crypto/` in `.gitignore`?

We want code to **fail loud** when writing here, not just be
git-invisible. A `.gitignore` would mask the bug. The sentinel
directory + CI lint together make accidental writes a visible test
failure rather than a silent disk leak in the repo working tree.

## Related

- `plans/crypto-data-contract.md` §1 Storage Root, §1.5 Network Profile
- `plans/crypto-dev-phases.md` Phase Crypto-0 / Crypto-A deliverables
- Memory `[[crypto-quant-research-20260530]]`,
  `[[crypto-dev-phases]]` for higher-level context.

## Sign-off

Acceptance: this README is committed at Phase Crypto-0 closure. A
future `data/storage/crypto/.crypto_root_marker.json` (or similar)
may be added by Phase 0b's config module to track the chosen real
root and any environment override.
