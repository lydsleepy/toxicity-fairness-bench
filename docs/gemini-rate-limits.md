# Gemini API: Rate Limits and Benchmark Exclusion

## What happened

During the benchmark run for this project, Google Gemini was excluded from the
published results despite being fully implemented in the codebase. This document
explains exactly why and what you need to know before running Gemini yourself.

## Timeline of the issue

1. **First benchmark attempt** (`datasets` library failure): The HuggingFace
   `datasets` library had been updated to v4.x, which dropped support for
   dataset scripts. The `hatexplain` dataset uses the old script format, so the
   run failed at data loading before any API calls were made. Fix: pin
   `datasets>=2.18,<3.0` and pass `trust_remote_code=True` (already done in
   `loaders.py`).

2. **Second benchmark attempt** (`gemini-2.0-flash-lite`, no retry logic): The
   benchmark loaded successfully and Perspective/Claude ran fine. However,
   the Gemini analyzer at the time had no retry logic and no sleep on failed
   calls. When a call hit a 429 rate limit, it returned immediately without
   sleeping — causing all 1,000 Gemini calls to fire in rapid succession,
   all returning `429 RESOURCE_EXHAUSTED`. This burned through the model's
   daily free-tier quota (1,500 RPD) in seconds.

3. **Fix applied**: Added `tenacity` retry with exponential backoff
   (10–120s) on 429 errors, and ensured failed calls also sleep to pace
   the request loop.

4. **Third benchmark attempt** (`gemini-2.0-flash-lite`, with retry): The
   daily quota for this model was already exhausted from attempt #2. All
   calls returned `429` immediately regardless of sleep interval.

5. **Switch to `gemini-2.5-flash-lite`**: This model has a separate quota
   pool and was available. The first 10–15 calls succeeded cleanly at
   ~2.4s each. After that, the per-minute rate limit kicked in and nearly
   every call required the tenacity retry backoff (10–30s wait before
   retry), pushing the average to ~32s/call. At that rate, 1,000 calls
   would take 9+ hours.

6. **Final decision**: Exclude Gemini from the published benchmark. The
   Perspective and Claude results (1,000 samples each, cached) are used
   for the analysis.

## Root cause

The Gemini free tier for `gemini-2.5-flash-lite` appears to have a strict
per-minute burst limit (approximately 10–15 requests per burst window)
that resets slowly. Running 1,000 sequential calls — even with a 5-second
sleep between them — exceeds this limit after the initial burst. The
documented 30 RPM limit may reflect sustained throughput under ideal
conditions rather than the actual burst behavior observed here.

## Gemini is still supported

The `GeminiAnalyzer` class in `src/toxicity_fairness/analyzers/gemini.py`
is fully implemented, tested, and uses `gemini-2.5-flash-lite` by default.
The retry logic (`tenacity`, exponential backoff) is in place. Running
Gemini at smaller sample sizes works fine.

## How to run Gemini successfully

**For small samples (≤ 100 rows):**

```bash
python scripts/run_benchmark.py --sample 100 --models gemini --output results/
```

The 100-call burst should complete in ~4 minutes before rate limiting kicks in.

**For larger samples (> 100 rows):**

You will need either:

1. **A paid Gemini API key** — higher rate limits, no daily quota issues.
   Set `GEMINI_API_KEY` in `.env` and run normally.

2. **Patience + overnight run** — set `GEMINI_SLEEP_SECS=30` in `.env`
   (2 RPM, well below any limit) and let it run overnight:
   ```bash
   GEMINI_SLEEP_SECS=30 python scripts/run_benchmark.py \
       --sample 1000 --models gemini --output results/ --use-cache
   ```
   1,000 calls × 30s = ~8.5 hours.

3. **Multiple days** — run ~200 samples per day within the free-tier
   daily quota, using `--use-cache` to resume.

## Lessons learned

- Rate-limiting logic must sleep even on failed calls, not just on success.
  The original code only slept after a successful API response — error paths
  returned immediately, enabling a request storm.
- Daily quotas can be burned by failed requests. 429 errors still consume
  the daily request count for `gemini-2.0-flash-lite`.
- Model versions have separate quota pools. When one model's quota is
  exhausted, switching to a newer version (e.g., 2.5) may have fresh quota.
- For portfolio benchmarks, either use a paid API key or reduce the sample
  size to stay within free-tier burst limits.
