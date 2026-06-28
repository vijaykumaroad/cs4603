# data/eval/

Test sets for PA-2 evaluation.

## Files

| File | Used In | Description |
|------|---------|-------------|
| `verification_queries.json` | Part 1.4 | Store verification queries (should-pass + should-fail) |
| `raptor_qa.json` | Part 5.6 | Theme-level questions for RAPTOR ablation |

## Schema: `verification_queries.json`

```json
{
  "id": "q-1",
  "query": "What is the revenue recognition policy?",
  "expected_source": "RAG_PAPER.pdf",
  "should_pass": true
}
```

## Schema: `raptor_qa.json`

```json
{
  "id": "q-raptor-1",
  "question": "Summarize the key findings about RAG across the papers.",
  "difficulty": "theme",
  "expected_level": 1,
  "rationale": "Multi-document synthesis question."
}
```
