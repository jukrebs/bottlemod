# Experiment 3 — Write-back caching (dirty-limit throttling) / checkpointing

## Goal

Demonstrate the benefit of modeling **write-back caching** (Linux page cache writes) for workflows with checkpointing or bursty writes.

Show a classic pattern:

- small buffered checkpoint: fast (memory tier)
- large checkpoint or frequent sync: throttled (disk tier)

Then apply a fix:

- batch sync points (less frequent fsync)
- reduce checkpoint size
- increase dirty budget / move to faster disk

## Workflow

Iterative compute with periodic checkpoint:

- (compute)
- write checkpoint
- (compute)

Two modes:

### Baseline (sync-heavy)

- O_SYNC or fsync after small writes → always disk-bound

### Fix (write-back / batching)

- buffered writes, flush once per checkpoint (or less often)

## Bottleneck narrative

BottleMod’s disk-only assumption over-predicts cost of buffered writes.
The write-cache model predicts two-phase behavior and correctly identifies when the bottleneck is disk write bandwidth.

## Implementation pointers (repo)

- Worked example: `example_write_cache.py`
- Builder: `bottlemod_new/builders/write_cache.py`
- Interface docs: `interface_write_cache.md`
