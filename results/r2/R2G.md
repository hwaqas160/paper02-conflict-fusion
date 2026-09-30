## R2-G -- Range bands, near-range misses, latency (adverse pooled, test)

| arm | AP 0-20 m | AP 20-40 m | AP 40-70 m | miss rate < 30 m |
|---|---|---|---|---|
| Per-object (ours) | 0.856 | 0.806 | 0.671 | 0.125 |
| Fixed (status quo) | 0.821 | 0.736 | 0.556 | 0.199 |
| Per-frame classifier | 0.819 | 0.749 | 0.577 | 0.187 |

Router latency (CPU, 1 thread, one frame incl. feature assembly): Per-object (ours) median 2.9 ms, p95 4.4 ms; Per-frame classifier median 2.6 ms, p95 4.2 ms
