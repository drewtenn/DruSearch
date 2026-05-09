# Checked-in LTR model artifacts

`make promote-model` writes the served LightGBM model here:

- `ltr_reranker.txt`
- `ltr_reranker.json`

With `LTR_MODEL_BACKEND=xgboost`, promotion writes:

- `ltr_reranker.xgb.json`
- `ltr_reranker.json`

Commit those two files when you want another machine to `git pull`, run
`make up`, and serve the latest trained ranker without rerunning BGE teacher
scoring or model training.
