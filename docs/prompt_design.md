# Prompt design notes

## Why prompt design matters

Unlike Perspective (a purpose-built classifier), Gemini and Claude are
general-purpose LLMs. Prompt wording significantly affects output and
can surface implicit biases.

## Current prompts

### Gemini

```
Rate the toxicity of the following text on a scale from 0.0 (completely
non-toxic) to 1.0 (extremely toxic). Consider hate speech, harassment,
discrimination, and harmful stereotypes as toxic. Respond with only a
single decimal number between 0.0 and 1.0, nothing else.

Text: {text}
```

### Claude (system prompt)

```
You are a toxicity classifier. When given a text, respond with only a
single decimal number between 0.0 and 1.0 representing toxicity:
0.0 means completely non-toxic, 1.0 means extremely toxic.
Consider hate speech, harassment, harmful stereotypes, and discrimination
as toxic. Output only the number — no explanation, no punctuation.
```

Using a system prompt separates task definition from text input — cleaner
than embedding instructions in the user turn.

## Experiments to try

1. **Bias-aware prompt** — explicitly instruct the model to ignore
   demographic markers when scoring
2. **Chain-of-thought** — ask for one-sentence reasoning before the score
3. **Few-shot** — provide 4 labeled examples before the target text
4. **Counterfactual consistency** — swap demographic terms (Women → Men)
   and check whether scores change

To add a prompt variant, subclass the existing analyzer and override
`analyze_one`. See `src/toxicity_fairness/analyzers/gemini.py` for
the pattern.
