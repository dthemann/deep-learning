# Deep Learning with PyTorch: From Foundations to Frontier

An extensive, self-paced deep learning course delivered as eighteen Jupyter notebooks. Every notebook
is a full lecture: long-form explanations in Markdown, the mathematics derived step by step, and
runnable PyTorch code for every concept, followed by experiments, exercises with worked solutions,
a summary and further reading.

The course starts from the mathematics and builds neural networks in pure NumPy before introducing
PyTorch, then progresses through modern computer vision, sequence models and Transformers, Hugging
Face fine-tuning, generative models (VAEs, GANs, diffusion), reinforcement learning, and finally the
engineering side: efficient training, deployment, interpretability and robustness, and a capstone.

## Contents

| # | Notebook | Part | Time |
|---|----------|------|------|
| 00 | [Course Overview and Setup](notebooks/00_course_overview_and_setup.ipynb) | I · Foundations | 45 min |
| 01 | [Mathematical Foundations](notebooks/01_mathematical_foundations.ipynb) | I · Foundations | 2 h |
| 02 | [Neural Networks from Scratch (NumPy)](notebooks/02_neural_networks_from_scratch.ipynb) | I · Foundations | 2 h |
| 03 | [Tensors and Autograd](notebooks/03_tensors_and_autograd.ipynb) | I · Foundations | 2 h |
| 04 | [Training Deep Networks: Optimization, Initialization, Normalization, Regularization](notebooks/04_training_deep_networks.ipynb) | II · Training | 2.5 h |
| 05 | [The PyTorch Workflow and Best Practices](notebooks/05_pytorch_workflow_and_best_practices.ipynb) | II · Training | 2 h |
| 06 | [Convolutional Neural Networks](notebooks/06_convolutional_neural_networks.ipynb) | III · Vision | 2.5 h |
| 07 | [Transfer Learning and Modern Vision](notebooks/07_transfer_learning_and_modern_vision.ipynb) | III · Vision | 2 h |
| 08 | [Sequence Models: RNNs, LSTMs and GRUs](notebooks/08_sequence_models_rnn_lstm.ipynb) | IV · Sequences and language | 2 h |
| 09 | [Attention and Transformers](notebooks/09_attention_and_transformers.ipynb) | IV · Sequences and language | 2.5 h |
| 10 | [NLP with Pretrained Transformers (Hugging Face)](notebooks/10_nlp_with_pretrained_transformers.ipynb) | IV · Sequences and language | 2 h |
| 11 | [Autoencoders and Variational Autoencoders](notebooks/11_autoencoders_and_vaes.ipynb) | V · Generative models | 2 h |
| 12 | [GANs and Diffusion Models](notebooks/12_gans_and_diffusion_models.ipynb) | V · Generative models | 2.5 h |
| 13 | [Reinforcement Learning](notebooks/13_reinforcement_learning.ipynb) | VI · Reinforcement learning | 2 h |
| 14 | [Efficient Training and Scaling](notebooks/14_efficient_training_and_scaling.ipynb) | VII · Engineering and practice | 2 h |
| 15 | [Deployment and Model Optimization](notebooks/15_deployment_and_model_optimization.ipynb) | VII · Engineering and practice | 2 h |
| 16 | [Interpretability, Robustness and Uncertainty](notebooks/16_interpretability_robustness_uncertainty.ipynb) | VII · Engineering and practice | 2 h |
| 17 | [Capstone Projects](notebooks/17_capstone_projects.ipynb) | VII · Engineering and practice | 1.5 h + project |

Times are for reading and running the notebook; exercises take extra.

## Setup

Requirements: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Miniforge, about
10 GB of disk space (packages plus datasets), and optionally an NVIDIA GPU with a driver that
supports CUDA 12.

```bash
# GPU (NVIDIA) environment
conda env create -f environment.yml
conda activate dl-course
python -m ipykernel install --user --name dl-course --display-name "Python (dl-course)"
jupyter lab
```

Without an NVIDIA GPU (Windows, Linux or macOS) use the smaller CPU variant:

```bash
conda env create -f environment-cpu.yml
conda activate dl-course-cpu
```

Then open `notebooks/00_course_overview_and_setup.ipynb`. It checks the installation, reports the
GPU, and explains the conventions used in the course.

PyTorch itself is installed through pip from the official PyTorch wheel index (the PyTorch team no
longer publishes conda packages). The comments at the top of `environment.yml` explain how to pick a
different CUDA build.

## How the notebooks work

- **Read in order.** Later notebooks assume the vocabulary of the earlier ones, although each one is
  self-contained in code (it defines the helpers it needs) so you can open any of them directly.
- **`FAST_RUN`.** Every notebook has a `FAST_RUN` flag in its setup cell. With `FAST_RUN = True` all
  training budgets shrink so that the whole notebook runs in a few minutes on a laptop CPU. Set it to
  `False` (the default) for the full experiments, which take roughly 5 to 20 minutes on a mid-range GPU.
- **Data.** Datasets download automatically into the `data/` folder the first time they are used
  (MNIST, FashionMNIST, CIFAR-10, Tiny Shakespeare, small Hugging Face datasets and models).
  Notebooks that need a download fall back to built-in data when they are offline.
- **Exercises.** Each notebook ends with graded exercises and fully worked solutions. Try the
  exercise before opening the solution cell.
- **Reproducibility.** All notebooks seed Python, NumPy and PyTorch. GPU kernels are still not
  bit-for-bit deterministic by default; Notebook 05 explains the remaining sources of randomness.

## Suggested schedules

**Eight weeks (intensive, about 8 hours per week):**

| Week | Notebooks |
|------|-----------|
| 1 | 00, 01, 02 |
| 2 | 03, 04 |
| 3 | 05, 06 |
| 4 | 07, 08 |
| 5 | 09, 10 |
| 6 | 11, 12 |
| 7 | 13, 14, 15 |
| 8 | 16, 17 and capstone project |

**Sixteen weeks (one notebook per week, capstone in the last two weeks).**

## Troubleshooting

- `torch.cuda.is_available()` returns `False`: update the NVIDIA driver, confirm that `pip show torch`
  reports a `+cu126` version, and that you activated the `dl-course` environment before starting Jupyter.
- `CUDA out of memory`: reduce the batch size in the notebook's configuration cell or set `FAST_RUN = True`.
- DataLoader hangs on Windows: keep `num_workers=0` (the default in every notebook).
- A Hugging Face model or dataset does not download: the notebook prints a message and uses the
  built-in fallback; check the proxy settings or set `HF_HOME` to a writable folder.
- The kernel dies while training: usually memory; restart the kernel, set `FAST_RUN = True`, and run
  the cells again in order.

## How the notebooks were verified

Every notebook was executed top to bottom with `FAST_RUN = True` on a CPU-only machine
(PyTorch 2.14, transformers 5.x, gymnasium 1.x) and completed without errors; the fast runs take
between ten seconds and about three and a half minutes per notebook. The full-budget runs
(`FAST_RUN = False`) were not executed on a GPU during preparation, so treat the quoted full-run
accuracies as expectations rather than measurements. Each notebook was also independently reviewed
for mathematical correctness, code correctness and references after it was written.

## Layout

```
deep-learning-course/
  README.md
  environment.yml        # conda environment, CUDA build of PyTorch
  environment-cpu.yml    # CPU-only variant
  data/                  # created automatically, datasets download here
  notebooks/             # the eighteen lecture notebooks
```
