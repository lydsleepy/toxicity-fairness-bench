# Cohere API: Rate Limits, Model Choice, and Sample Size

## Trial key limits (verified 2026-07-14)

- Chat endpoint: 20 requests per minute. Applies to all Command models
  (Command A, Command R+, Command R, Command R7B).
- Monthly cap: 1,000 API calls per month, shared across all Cohere
  endpoints (chat, embed, rerank, etc.), not per endpoint.

This is tighter than Gemini's free-tier burst limit. See
`docs/gemini-rate-limits.md` for the Gemini story, including the lesson
that failed (429) requests still consume quota, not just successful ones.

## Model choice: command-r7b-12-2024

`CohereAnalyzer` defaults to `command-r7b-12-2024`, Cohere's smallest and
cheapest Command model, instead of the flagship `command-a-03-2025`. Same
reasoning as picking Claude Haiku and `gemini-2.5-flash-lite` for the
other two prompted-LLM analyzers: cost and latency matter more than peak
quality for a binary toxicity classification prompt, and the trial-tier
rate limits reward cheaper, faster models.

## Why the Cohere benchmark run is n=500, not n=1,000

A 1,000-sample run would use the entire monthly trial quota in one pass,
with no margin for retries. The published benchmark used 1,000 samples
for Perspective and Claude. Cohere was run at 500 instead, using the same
seed=42 HateXplain draw, first 500 rows, so the run uses half the monthly
quota and leaves room for a retry or a second run later in the month.

The actual run: 500 out of 500 calls succeeded, zero errors, at
`COHERE_SLEEP_SECS=3.5` (well under the 20 requests/minute limit).

## Which subgroups drop out at n=500

Fairness gap metrics need at least 2 demographic subgroups with 5 or more
examples of both classes (`MIN_CLASS_N` in `metrics/fairness.py`).
HateXplain's non-toxic examples are already scarce per subgroup at
n=1,000. Halving the sample halves that count again. This was computed
directly from the cached data, with no additional API calls:

| Attribute | At n=500 |
|---|---|
| Gender | "Men" drops below the threshold. Only "Women" remains, so the gap metric is unavailable for all models compared at this sample size. |
| Race/Ethnicity | All 6 subgroups (African, Arab, Asian, Caucasian, Hispanic, Jewish) drop below the threshold. Gap unavailable for all models. |
| Religion | "Christian" has 0 examples in the n=500 draw. Only "Islam" remains, so the gap is unavailable. This matches the situation at n=1,000 for a separate, pre-existing reason: "Christian" was already below the threshold there too. |
| Other (catch-all) | 4 of 7 subgroups survive. This is the only attribute with usable gap metrics at n=500. |

This is not Cohere-specific. Subgroup membership is a property of the
text, not the model, so it applies the same way to Perspective and Claude
when viewed at n=500. The n=500 comparison is useful for overall accuracy
across all four models, but not for Gender, Race, or Religion fairness
gaps. Those need the full n=1,000 view, which only has Perspective and
Claude.

## Confidence interval cost

Bootstrap 95% CI width on overall accuracy, computed from the same cached
Perspective data at both sample sizes, using the same bootstrap method the
dashboard uses:

| n | CI width |
|---|---|
| 1,000 | 0.062 |
| 500 | 0.078 |

About 1.26x wider at n=500. That's narrower than the naive 1/sqrt(n)
heuristic (about 1.41x) would predict. Per-subgroup CI widths are much
noisier than this overall figure and unreliable below about 20 examples,
which is part of why `MIN_CLASS_N` exists.

## Running Cohere yourself

```bash
python scripts/run_benchmark.py --sample 500 --models cohere --output results/cohere_500/ --use-cache
```

Use `--use-cache` so a re-run resumes from cached results instead of
re-spending calls on rows already scored.

## Not in the live scorer

`CohereAnalyzer` is intentionally not registered in
`app/routers/scorer.py`. Cohere trial keys explicitly prohibit
production or public-facing use, and the Live Scorer endpoint is live on
a public Railway deployment. Cohere results are only ever served from
pre-computed Parquet files (`/api/filters`, `/api/metrics`), never from a
live call at request time.
