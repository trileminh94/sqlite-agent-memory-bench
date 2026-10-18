# sqlite-agent-memory-bench

A small benchmark of SQLite as the memory store for a coding agent: one file holding the memory rows, an FTS5 keyword index and a sqlite-vec vector index.

It measures two things:

1. **Storage.** How many bytes each layer costs per memory: plain rows, rows + FTS5, rows + FTS5 + sqlite-vec `vec0`, plus two do-it-yourself layouts for comparison (a float32 BLOB column, and the vector stored as JSON text).
2. **Recall.** Latency and hit@5 for FTS5, `vec0` KNN and a hybrid of the two (reciprocal-rank fusion) on 300 paraphrased queries, both across all 10k memories and scoped to one user.

Everything runs locally on CPU. No API calls.

## Reproduce

```sh
pip install -r requirements.txt && python gen_memories.py 10000 && python embed.py && python bench.py && python bench2.py && python report.py
```

The first run of `embed.py` downloads `all-MiniLM-L6-v2` (about 90 MB) through fastembed. Expect a few minutes end to end and roughly 250 MB of SQLite files in `db/`.

## Headline numbers

From `results.md` (10,000 memories, k=5, query embedding not included):

| method | median | p95 | hit@5, per user (~200 memories) | hit@5, global (10k) |
|---|---|---|---|---|
| FTS5 (bm25) | 0.41 ms | 1.32 ms | 69.0% | 9.3% |
| sqlite-vec vec0 KNN | 4.34 ms | 4.72 ms | 99.3% | 27.7% |
| hybrid (RRF) | 5.50 ms | 6.26 ms | 95.7% | 18.0% |

Storage per memory: rows 99.1 B, + FTS5 147.9 B, + vec0 1,747.8 B. The same vectors stored as JSON text: 8,726.5 B per memory, and a full scan takes 1,201 ms median.

A few things worth knowing:

- On these paraphrased queries, hybrid did **not** beat vector search alone.
- Filtering by user *after* taking the global top 5 left 0.25 of 5 results on average, and 77.3% of queries got nothing back. Filter before k (a `vec0` partition key) instead.
- Binary-quantised vectors (`bit[384]`, 48 bytes each) with a float rerank of the top 50 matched float32 vec0 on global hit@5 (27.7%) at 0.76 ms median.

## Files

- `gen_memories.py` makes the synthetic memories and their paraphrased questions (seed 7).
- `embed.py` embeds them with `all-MiniLM-L6-v2` and samples 300 queries.
- `bench.py` builds the storage variants and measures latency and global hit@5.
- `bench2.py` measures per-user hit@5, one insert, and binary vectors.
- `report.py` writes `results.md` from the raw JSON.

## Caveats

- **Synthetic data.** 10,000 memories from 18 templates, with intentional near-duplicates. Real agent memory is messier.
- **Same-script queries.** The paraphrased queries come from the same generator as the memories, so they are paraphrases of a known template, not real user questions.
- **hit@5 scope.** The per-user number searches only that user's ~200 memories. The global number searches all 10k, where each target has hundreds of near-identical siblings. Use the global number to compare methods, not as a quality score.
- **One model.** `all-MiniLM-L6-v2` on CPU. A different embedding model will move the vector numbers.
- **One machine.** 8 vCPU VM, 15 GB RAM, Python 3.13.5, SQLite 3.46.1, sqlite-vec 0.1.9. Your latencies will differ.
- **Latency excludes query embedding.** Embedding one query took 1.4 ms median in this run and has taken 5 to 10 ms in other runs on the same VM. Add it to the vec0 and hybrid numbers.

## License

MIT
