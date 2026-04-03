# Dataset setup

## Option 1: HateXplain (recommended — downloads automatically)

```bash
pip install datasets
python -c "from toxicity_fairness.data.loaders import load_hatexplain; print(load_hatexplain(sample=50).head())"
```

~20,000 posts with target community labels mapping to protected attributes.
License: CC BY 4.0.

## Option 2: Jigsaw

1. Download `train.csv` from
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
2. Place at `data/raw/jigsaw/train.csv`
3. Test: `python -c "from toxicity_fairness.data.loaders import load_jigsaw; print(load_jigsaw('data/raw/jigsaw/train.csv', sample=50).head())"`

## Recommended sample sizes

| Goal | Sample | Approx. cost |
|---|---|---|
| Smoke test | 50–100 | < $0.01 |
| Development | 200–500 | $0.05–0.20 |
| Publication | 2,000+ | $0.50–2.00 |

Results are cached — you only pay for each (model, sample) combination once.
