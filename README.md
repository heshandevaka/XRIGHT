<div align="center">
    <img alt="logo" src="./docs/logo.png" style="height: 110px;" />
</div>

<hr>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/hanshen95/SEAL/blob/main/LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)  [![Arxiv link](https://img.shields.io/badge/cs.LG-submitted-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/)



This is the official repo of ALRIGHT and MAXRIGHT algorithms for improved LLM post-training (see the paper [Mitigating Forgetting in LLM Supervised Fine-Tuning and Preference Learning](https://arxiv.org/)).


## Introduction

The widely adopted approach in post-training popular open-source LLMs is to sequentially perform SFT and DPO. However, sequential training is sub-optimal in terms of SFT and DPO trade-off: when trained with preference data, LLM inevitably forgets about the knowledge learnt during SFT, despite the presence of KL regularization. Similar issue persists when performing DPO first and then SFT in LLM continual learning. Simple solution like directly mixing the DPO and SFT objective greatly increases the computational cost and slows down the training speed.

 As a remedy of these issues, we implement the [ALRIGHT and MAXRIGHT](https://arxiv.org/) algorithms. The algorithms demonstrate the following merits:
 
- **Improved post-training performance**: Models such as [Llama-3-8b](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6) trained with ALRIGHT/MAXRIGHT demonstrates suprior quality than those trained with sequential method. Experiments showcase a 3% on [MMLU](https://huggingface.co/datasets/cais/mmlu) (1-shot) and a 31% win rate increase on [Anthropic HH](https://huggingface.co/datasets/Anthropic/hh-rlhf).

- **Similar computaitonal cost as sequential training**: The computational cost compared to sequential SFT/DPO is similar, with a worst-case increase of 2% in GPU utilization when training [Llama-3-8b](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6).




## Example Results

We give some examples here. For complete results, please see [paper](https://arxiv.org/).

#### Loss space result and computational complexity

Loss trade-off plot for training [Pythia-1b](https://huggingface.co/EleutherAI/pythia-1b) with this implementation (**ALRIGHT and MAXRIGHT**), **sequential** method (SFT then DPO) and **mix**(combining DPO and SFT objectives):

<div align="center">
    <img alt="loss figure" src="./docs/loss_figure.png" style="height: 220px;" />
</div>

#### Benchmark result

Result on **Llama-3-8b**. Here $\lambda$ is the normalized weight for DPO loss, defining the trade-off between DPO and SFT:

<div align="center">
    <img alt="benchmark table" src="./docs/benchmark_table.png" style="height: 210px;" />
</div>

The win percent metric follows [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval/tree/main?tab=readme-ov-file), which is commonly used for human preference evaluation.


## Installation

Create conda environment

```bash
conda create -n alright python=3.10
conda activate alright
```

To install the denpendencies, navigate to the root directory and
```bash
pip install -r requirements.txt 
```

Then install the repo
```bash
pip install -e .
```

## Running Example

Under construction

## Acknowledgement

We would like to thank all packages this repo is built on, and especially

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): for the vanilla SFT and DPO implementation and their great extention capability.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): for the efficient distributed training functions.


