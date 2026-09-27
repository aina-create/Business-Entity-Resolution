# ML Challenge 2026: Business Entity Resolution

**Team:** [team name]
**Members:** [names]
**Submission date:** [date]

## 1. Executive summary

[Summarize the method, candidate reduction, model, and validation result. Clearly mark any unmeasured result.]

## 2. Data and validation

- Training split: [record counts and sources]
- Validation design: [held-out entities/split method; avoid splitting candidate pairs after labels]
- Test handling: retain all country labels, including labels absent in training.
- Metric: macro F0.5 over every Source 1 entity, including singletons.

## 3. Candidate generation

- Blocking keys: [exact normalized name, country/name prefixes, address keys, etc.]
- Maximum block size or other caps: [values and rationale]
- Candidate count and reduction ratio: [measured values]
- Blocking recall on held-out ground truth: [measured value]
- Explain how rows with no candidates are represented in `candidate_pairs.tsv`.

## 4. Matching model

- Model: [algorithm and license]
- Features: [name, address, country and blocking features]
- Training labels and negative sampling: [details]
- Threshold selection: [method; optimize validation macro F0.5]
- Model artifact and exact training command: [path/command]

## 5. Results and error analysis

| Measure | Validation result |
| --- | ---: |
| Macro F0.5 | [value] |
| Precision / recall | [values] |
| Blocking recall | [value] |
| Candidate pairs | [count] |

False positives: [examples/patterns]
False negatives: [examples/patterns]

## 6. Reproduction

Document the Python version, dependencies, data directory, commands, and expected output files. The package should reproduce both `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the supplied training and test data.

## 7. Limitations

[State unvalidated assumptions, block caps that may miss true matches, memory/disk requirements, and any model/output components still using a baseline rule.]
