# Low-Rank Adaptation of LLMs

This repository is an implementation of the paper - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf) by E. J. Hu et al, 2021. 

## Introduction

LoRA is a technique that enables low-rank adaptation of large language models. It aims to reduce the computational cost and memory requirements of fine-tuning language models by leveraging low-rank approximations.

## Getting Started

To get started with this repository, follow the steps below:

1. Clone the repository: 
```shell
>> git clone https://github.com/nikhil-chigali/Low-Rank-Adaptation-of-LLMs.git
```
2. Navigate to project directory: 
```shell
>> cd Low-Rank-Adaptation-of-LLMs
```
3. Install poetry: 
```shell 
>> pip install poetry
```
4. Install the required dependencies: 
```shell
>> poetry install --no-root
```
5. Run the training script: 
```shell
>> python finetune_roberta_with_lora.py
```

## Usage

Here are some examples of how to use the LoRA adaptation technique:
[To be added]

## Acknowledgements

I would like to acknowledge the use of following resources as reference:
- [Code LoRA from Scratch](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch) by Sebastian Raschka - Lightning Studio
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf) by E. J. Hu et al, 2021 - Paper

