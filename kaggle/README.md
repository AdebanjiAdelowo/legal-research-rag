# Kaggle GPU runner

`run_gpu_experiments.py` runs the experiments that need a GPU: the legal and general
ModernBERT encoders over the full corpus, a large cross-encoder reranker, the large
entailment judge, and generation with an open instruction-tuned model.

## Resource estimates

Measured where a local run exists, otherwise computed from parameter counts.

| Item | Download | Disk | GPU memory |
|---|---|---|---|
| LePaRD `top_10000_data.csv.gz` | 539 MB | 539 MB | none |
| 4,940 opinions from the Caselaw Access Project | about 300 MB | 300 MB | none |
| Processed corpus and chunks | none | 0.9 GB | none |
| BM25 index | none | 37 MB | none |
| `BAAI/bge-small-en-v1.5` and its index | 133 MB | 133 MB + 171 MB | under 2 GB |
| `nomic-ai/modernbert-embed-base` and its index | 596 MB | 596 MB + 342 MB | about 4 GB |
| `freelawproject/modernbert-embed-base_finetune_512` and its index | 596 MB | 596 MB + 342 MB | about 4 GB |
| `BAAI/bge-reranker-v2-m3` | 2.3 GB | 2.3 GB | about 4 GB |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 1.7 GB | 1.7 GB | about 3 GB |
| `Qwen/Qwen2.5-7B-Instruct` in half precision | 15.2 GB | 15.2 GB | about 16 GB |

Total: about 22 GB of downloads and 24 GB of disk. A single 16 GB GPU (T4 or P100)
is enough for everything except the 7B generator, which needs the T4 x2 option or a
smaller model. To use a smaller generator, set for example
`LEGALRAG_GENERATOR=hf:Qwen/Qwen2.5-3B-Instruct` (about 6 GB).

Expected run time on a T4: about 25 minutes per ModernBERT index, 15 minutes for
reranking 2,000 queries at depth 100, and 60 to 90 minutes for 600 generations.

## Running it

The repository has no remote, so the source is supplied to Kaggle as a private
dataset.

```bash
# 1. Package the source (code only; data/ and results/ are excluded).
git archive --format=zip --prefix=legal-research-rag-src/ -o /tmp/legal-research-rag-src.zip HEAD

# 2. Create the private dataset from the unzipped folder
#    (replace KAGGLE_USERNAME in dataset-metadata.json and kernel-metadata.json first).
kaggle datasets create -p /tmp/legal-research-rag-src --dir-mode zip

# 3. Push and run the kernel.
kaggle kernels push -p kaggle

# 4. When it finishes, fetch the results and merge them.
kaggle kernels output KAGGLE_USERNAME/legal-research-rag-gpu -p /tmp/kaggle-out
unzip -o /tmp/kaggle-out/legal-research-rag-results.zip -d results/
```

## Data licence on Kaggle

LePaRD is licensed CC BY-NC-SA 4.0. The runner downloads it inside the notebook from
its pinned Hugging Face revision and never uploads it. Keep the kernel and the source
dataset private, and do not publish the notebook's working directory.
