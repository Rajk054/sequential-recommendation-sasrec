# Sequential recommendation with SASRec

I built this project to answer a practical question: when is a sequential recommender actually better than simpler collaborative-filtering models?

The experiment compares three PyTorch models on MovieLens 1M:

- matrix factorization;
- a two-tower neural retrieval model;
- SASRec, a causal Transformer that predicts the next item from a user's ordered history.

The models use the same chronological split and are evaluated with full-catalog NDCG and Recall. There is also a small FastAPI service backed by a FAISS HNSW index, which lets me compare exact retrieval with approximate nearest-neighbor search.

The complete training run, cohort analysis, rejected tuning experiment, and retrieval benchmark are available in the [public Kaggle notebook](https://www.kaggle.com/code/rajkanchan/sasrec-vs-mf-and-two-tower-on-movielens-1m).

## Measured results

These results use chronological, full-catalog test evaluation with seed 7.

| Model | NDCG@10 | Recall@10 | Training time |
|---|---:|---:|---:|
| Matrix factorization | **0.0213** | **0.0419** | 36.8 s |
| Two-tower | 0.0180 | 0.0384 | 36.9 s |
| SASRec | 0.0094 | 0.0190 | 64.5 s |

Matrix factorization was the strongest overall model. SASRec's validation deficit narrowed for users with more than 155 training interactions: its Recall@20 was 0.0573 versus 0.0593 for two-tower and 0.0633 for matrix factorization. It remained substantially weaker for short histories.

A longer SASRec candidate reduced sampled-negative BCE loss from 1.038 to 0.868 but also reduced validation NDCG@10 from 0.0092 to 0.0049. I rejected that candidate because the lower training loss did not improve the ranking objective.

On this 3,706-item catalog, exact search was faster than HNSW: median latency was 0.091 ms versus 0.135 ms, while HNSW achieved ANN Recall@10 of 0.427. This is a useful boundary rather than an ANN success claim—approximate retrieval needs a larger catalog or different index tuning to justify itself.

## Evaluation setup

For every user, the last interaction is the test target and the second-to-last interaction is used for validation. Everything before those two interactions is training data.

During training, negatives are sampled uniformly while excluding items already observed for that user. During evaluation, each target is ranked against the full catalog after consumed items are masked. I chose this instead of evaluating one positive against a small set of sampled negatives, since that can make ranking results look much better than they really are.

I report NDCG@5/10/20 and Recall@5/10/20 rather than classification accuracy.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

Before downloading MovieLens, the whole pipeline can be checked with synthetic data:

```bash
python train.py --synthetic --epochs 2 --dim 32 --max-len 20
```

## Training on MovieLens 1M

`train.py` downloads MovieLens 1M from GroupLens the first time it runs.

```bash
python train.py \
  --epochs 20 \
  --dim 64 \
  --max-len 100 \
  --negatives 4 \
  --pointwise-batch-size 4096 \
  --sequence-batch-size 256
```

Pointwise negatives are sampled in batches on the selected device. The separate batch-size options keep MF and two-tower GPU work large without forcing the Transformer's sequence batch to consume the same amount of memory.

Checkpoints, mappings, interaction histories, and evaluation metrics are written to `artifacts/`. The measurements above use one seed; repeated-seed estimates are still required before treating the differences as statistical evidence.

## Retrieval benchmark

After training SASRec:

```bash
python benchmark_retrieval.py --queries 1000 --k 10
python report.py
```

The benchmark records exact and HNSW median/p95 retrieval latency as well as ANN recall. `report.py` creates `RESULTS.md` directly from the saved metrics, so the README does not contain invented benchmark numbers.

MovieLens has a small catalog, so exact search may be as fast as or faster than HNSW. FAISS is included to demonstrate the serving pattern that becomes useful when the item catalog grows, not to claim that ANN is automatically faster at every scale.

## API

Start the service after training:

```bash
python serve.py --artifacts-dir artifacts --port 8000
```

Request recommendations using original MovieLens item IDs:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'content-type: application/json' \
  -d '{"item_ids":[1,48,150,260],"k":10}'
```

The endpoint encodes the supplied history with SASRec, retrieves candidates from FAISS, removes consumed items, and returns ranked item IDs with retrieval latency.

## Findings and limitations

SASRec was not worthwhile globally in this experiment. It required slower training, Transformer inference, and ordered sequence state without beating either retrieval baseline. Its stronger relative performance for long-history users suggests where a sequential model could become useful, but that cohort result is not a global gain.

Matrix factorization remained competitive because MovieLens preference is largely explained by stable user-item affinity. The two-tower model is also easier to serve and extend with user or item features.

This is still an offline, single-seed experiment. MovieLens ratings are treated as implicit feedback, uniform negatives are not exposure-aware, and the metrics do not measure diversity, novelty, calibration, or business impact. A production follow-up would add repeated seeds, harder negatives, and an online experiment.

## Repository structure

```text
recsys/data.py         MovieLens loading and chronological splitting
recsys/sampling.py     pointwise and sequence negative sampling
recsys/models.py       matrix factorization, two-tower, and SASRec
recsys/metrics.py      full-catalog NDCG and Recall
recsys/training.py     training, evaluation, and checkpoints
recsys/retrieval.py    FAISS index construction
train.py               three-model experiment
benchmark_retrieval.py exact versus HNSW benchmark
serve.py               FastAPI recommendation service
report.py              artifact-backed results report
tests/                 data, sampling, metric, and model tests
```
