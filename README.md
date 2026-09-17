# MMKGAT

Multimodal knowledge graph recommendation with CLIP entity features, gated
embedding fusion, and item-image alignment. The preprocessing code generates
text-image `similar_to` relations using direct embedding similarity and topic
similarity.

This source snapshot has been prepared for publication without changing the
experiment logic or default hyperparameters. It is not yet a standalone,
verified reproduction package: the original `utils/` package and prompt templates
are missing. See [release preparation](docs/PUBLISHING.md) for the remaining items.

## Repository layout

```text
main_mmkgat.py                       Training and evaluation
main_mmkgat_tensorboard.py           Training with TensorBoard logging
model/KGAT.py                       Graph model and losses
data_loader/                        Data loading and batch sampling
parser/                             Original command-line defaults
data_processing/                    Topic extraction and CLIP graph preprocessing
datasets/README.md                  Input formats and local dataset layout
docs/PUBLISHING.md                  Publication scope and reproduction gaps
requirements*.txt                  Dependency inventories
apiKey.env.example                  Empty API-key configuration example
```

Dataset contents, model checkpoints, logs, and TensorBoard events are excluded
from Git. Keep their local locations unchanged when running the experiments.

## Environment

Use the Python, PyTorch/CUDA, NumPy, and other package versions from the original
experiment environment. The requirement files list dependencies found in the
source; their versions have not been recovered or validated, and installing the
latest versions is not a guarantee of compatibility or identical results.

```bash
python -m pip install -r requirements.txt
# For the TensorBoard entry point:
python -m pip install -r requirements-tensorboard.txt
# Only when rebuilding the preprocessing artifacts:
python -m pip install -r requirements-preprocessing.txt
```

Restore the exact experiment versions of these local modules before running:

- `utils/__init__.py`, `utils/log_helper.py`, `utils/metrics.py`, and
  `utils/model_helper.py` at the repository root.
- `data_processing/prompt.py`, defining `TEMPLATE_BOOK_REVIEW`,
  `TEMPLATE_MOVIE_REVIEW`, `TEMPLATE_BOOK_DESC`, and `TEMPLATE_MOVIE_DESC`, when
  running topic extraction. Prompt contents are part of the method and must be
  recovered without rewriting them.

The original sampler passes dictionary key views to `random.sample` and
`random.choice`. Compatibility with the intended Python version and batch sizes
must be checked in the experiment environment; this cleanup does not modify
sampling behavior.

## Training

Place the original inputs under `datasets/book/` or `datasets/movie/`, as described
in the [dataset documentation](datasets/README.md). Run commands from the
repository root after restoring the missing modules and original environment.

The following examples explicitly select a dataset and train from scratch. They
are invocation examples, not recovered commands for the reported paper results.

```bash
python main_mmkgat.py \
  --data_dir datasets \
  --data_name book \
  --image_emb_path datasets/book/image_embeddings.txt \
  --has_image_rel_id 22 \
  --use_pretrain 0

python main_mmkgat_tensorboard.py \
  --data_dir datasets \
  --data_name movie \
  --image_emb_path datasets/movie/image_embeddings.txt \
  --has_image_rel_id 22 \
  --use_pretrain 0
```

Important details of the preserved implementation:

- The loader joins `data_dir` and `data_name`. The original defaults are
  `datasets/` and `datasets/book`, which would resolve to `datasets/datasets/book`.
  Pass the dataset name explicitly as shown above.
- `--use_pretrain 2` is the original default and loads
  `trained_model/book/model_epoch680.pth` unless overridden. For a checkpoint run,
  use the matching dataset and pass `--pretrain_model_path` explicitly.
- The loader reserves relation IDs 0 and 1 for interactions and shifts original
  KG relation IDs by 2. In both local datasets, `hasImage` is 20 in
  `relation_list.txt`, so its forward relation ID during training is 22.
- Training alternates CF optimization and KG optimization, followed by attention
  updates. CF and KG use separate Adam optimizers as in the original code.
- Evaluation reads `test.txt`. The supplied `valid.txt` is not loaded by these
  entry points. Checkpoint selection and early stopping use recall at the smallest
  value in `Ks`. This behavior was preserved.
- Keep image features available for these examples. The original model does not
  initialize `init_entity_features` when no external features are provided.
- Both scripts call `train(args)` when executed. Their `predict()` functions are
  not command-line modes, and their model initialization differs; prediction-only
  checkpoint loading has not been validated in this cleanup.

Checkpoints and metric tables are written under `trained_model/`. The TensorBoard
entry point also writes events under `tensorboard/`.

## Preprocessing

`data_processing/gpt_request_final.py` extracts topics from reviews or descriptions.
It requires the original prompt module, a local `apiKey.env` copied from the empty
example, and JSON inputs under `./dataset/<name>/`. Its `data_name` and `data_type`
settings select the input. Paths are relative to the working directory. The
original request model, prompt construction, temperature, batching, and retry
behavior have been retained. Running this script sends API requests and replaces
the selected result/error files.

`data_processing/image_relation_gen_movie_new.py` reads graph mappings from
`./my_kg/movie/` and images from `./dataset/movie/images/`, according to its
`DEFAULTS` mapping. It uses the existing `hasImage` edges, encodes entity labels and
images, generates candidates in both directions, and writes a deduplicated merged
graph plus NumPy/pickle embeddings under `./image_relation_movie/`.

The graph script's NumPy/pickle outputs are not the training loader's
`image_embeddings.txt` format. This snapshot does not include the original
conversion step or the full raw-data-to-KG construction pipeline. Use the original
prepared inputs until those steps have been recovered; do not substitute a newly
invented conversion or prompt when reproducing the experiments.

## Publication and attribution

See [docs/PUBLISHING.md](docs/PUBLISHING.md) for exactly what to include in GitHub,
what to distribute separately, and what is still missing. A license and paper
citation should be added once the author, source attribution, and release terms
are confirmed; none have been inferred by this cleanup.
