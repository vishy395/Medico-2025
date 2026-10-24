# EndoVision — MediaEval Medico 2025

Code and experiment artifacts for **EndoVision's participation in the MediaEval Medico 2025 challenge**, which evaluates visual question answering (VQA) on gastrointestinal endoscopy images.

The team placed **3rd in Subtask 1** on the official challenge leaderboard, with rounded scores of **0.42 BLEU, 0.64 ROUGE-L, and 0.64 METEOR**.

## Team

- Sarath Chandran K R
- Sivasriraman P
- Vishnu Murugesh V
- Vishwajith L K
- Team: EndoVision, Sri Sivasubramaniya Nadar College of Engineering (SSNCE), India

## Approach

The project adapts Google's **PaliGemma 2** vision-language model to answer questions about gastrointestinal endoscopy images from the [Kvasir-VQA-x1 dataset](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1). The repository includes two inference workflows:

1. **Subtask 1 — VQA:** generate an answer for each image-question pair and calculate BLEU, ROUGE, and METEOR scores.
2. **Subtask 2 — multimodal explanations:** generate an answer, a Grad-CAM heatmap, a textual explanation, and a model-derived confidence score.

## Repository contents

| File | Purpose |
| --- | --- |
| [`subtask_1.py`](subtask_1.py) | Subtask 1 inference and public-split metric calculation. Writes `predictions_1.json`. |
| [`subtask_2.ipynb`](subtask_2.ipynb) | Notebook workflow for Subtask 2. Writes `predictions_2.json` and heatmaps under `visuals/heatmaps/`. |

Generated predictions, heatmaps, downloaded model weights, and credentials are not included in this repository.

## Model and data

- **Dataset:** [`SimulaMet/Kvasir-VQA-x1`](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1)
- **Base model and processor:** [`google/paligemma2-3b-pt-224`](https://huggingface.co/google/paligemma2-3b-pt-224)
- **EndoVision fine-tuned model/adapter:** [`vishy395/Kvasir-VQA-x1-paligemma2`](https://huggingface.co/vishy395/Kvasir-VQA-x1-paligemma2)

The fine-tuned model repository is publicly listed but gated: Hugging Face requires agreement to its access conditions before its files can be read. The base PaliGemma model may also require accepting its own access terms. Sign in and complete any required access steps before running the workflows. Keep access tokens out of source files and generated outputs.

## Environment setup

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended for practical inference with the 3B model. The notebook metadata indicates it was configured for a Colab T4 runtime; actual memory needs depend on the runtime and model-loading configuration.

Create and activate a virtual environment, then install a PyTorch build that matches your CUDA setup using the [official PyTorch install selector](https://pytorch.org/get-started/locally/). Install the remaining Python packages from the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install transformers datasets evaluate tqdm pillow numpy matplotlib opencv-python peft nltk rouge_score sacrebleu
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install transformers datasets evaluate tqdm pillow numpy matplotlib opencv-python peft nltk rouge_score sacrebleu
```

Install PyTorch separately before these packages so you can select the correct CPU or CUDA build. The first run downloads the dataset, model files, and evaluation metric implementations, so it needs network access and sufficient local disk space.

Authenticate with Hugging Face if the base model requires it:

```bash
hf auth login
```

The notebook's authentication cell adds `--add-to-git-credential`, which also stores the token for Git operations. The standard command above is sufficient for downloading model files.

### Model-loading note

The notebook loads `vishy395/Kvasir-VQA-x1-paligemma2` as a PEFT adapter on top of the Google base model. `subtask_1.py` currently loads the same repository directly with `PaliGemmaForConditionalGeneration.from_pretrained`. Confirm that the Hugging Face repository contains a merged, standalone checkpoint before running the script; if it contains adapter weights only, use the PEFT loading sequence from the notebook instead.

## Run Subtask 1

From the repository root:

```bash
python subtask_1.py
```

The script loads the dataset's `test` split, shuffles it with seed `42`, and selects the first 1,500 examples. It loads the EndoVision model and the PaliGemma 2 processor, generates answers, and calculates BLEU, ROUGE-1/2/L, and METEOR against the selected examples' reference answers.

The output file `predictions_1.json` contains the submission metadata, per-example predictions, metrics, runtime and memory information, and environment details. The reported scores from this local sample are not the official hidden-set leaderboard scores.

## Run Subtask 2

Open `subtask_2.ipynb` in Google Colab or Jupyter and run its cells in order. Select a GPU runtime for practical inference. The notebook filters to examples with complexity level `1`, shuffles with seed `42`, and selects 1,500 examples.

For each example, the notebook generates an answer, creates a Grad-CAM overlay, and prompts the model for a JSON-formatted explanation with location, morphology, and mucosal features. It writes:

- `predictions_2.json` — answers, explanations, heatmap references, confidence scores, and runtime details.
- `visuals/heatmaps/<val_id>.png` — Grad-CAM overlays for the selected examples.

The notebook's `confidence_score` is derived from the softmax probability of one selected output token. Treat it as a model signal, not as a calibrated probability that the full answer is correct. The generated explanations and heatmaps are research outputs and have not been established as clinically validated decision support.

## Results

The official leaderboard lists EndoVision in **3rd place for Subtask 1**. The leaderboard reports rounded BLEU / ROUGE-L / METEOR scores of **0.42 / 0.64 / 0.64**. The paper reports additional public and private test metrics.

## Paper and references

- [EndoVision paper PDF in the MediaEval 2025 proceedings](https://2025.multimediaeval.com/paper56.pdf)
- [MediaEval Medico 2025 final results and registered contributions](https://github.com/simula/MediaEval-Medico-2025/blob/main/competition_results.md)
- [MediaEval Medico 2025 task description](https://multimediaeval.github.io/editions/2025/tasks/medico/)

The proceedings page lists the paper as **“Medico 2025: Visual Question Answering (with multimodal explanations) for Gastrointestinal Imaging.”** The final results page records the EndoVision contribution under **“PaliGemma 2 for Explainable Colonoscopy VQA: Bridging Visual Attention and Clinical Insight.”** The linked PDF names the EndoVision authors and describes the PaliGemma 2 system.

## Reproducibility notes

- Both workflows select 1,500 examples after a deterministic shuffle with seed `42`.
- `subtask_1.py` and the notebook use different example selection logic; their local scores should not be compared as if they used the same evaluation subset.
- Hugging Face dataset and model revisions are not pinned in the code. Results may change if remote files or library versions change.
- This repository documents a challenge system; it is not intended for diagnosis or treatment decisions.
