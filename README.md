# Evaluating Machine Unlearning via the TOFU Framework

This repository contains the implementation and evaluation of **Machine Unlearning** techniques using the **TOFU (Task of Fictitious Unlearning)** dataset. As large language models (LLMs) increasingly train on sensitive or proprietary data, the ability to selectively "forget" specific training subsets—without retraining the entire model from scratch—has become a critical imperative for privacy-preserving AI, copyright compliance, and the "right to be forgotten."

## Project Overview

The goal of this project is to implement and benchmark different unlearning algorithms to assess how effectively an LLM can erase specific target data while maintaining its core utility on remaining, non-sensitive data.

Using the TOFU framework, the model is evaluated across two primary axes:
1. **Forget Quality:** How completely the target data has been erased (measured against a model that never saw the data).
2. **Model Utility:** How well the model retains its baseline knowledge and general language capabilities after the unlearning process.

## Core Objectives & Problem Statement

Standard fine-tuning easily updates a model's knowledge, but **removing** knowledge is fundamentally difficult due to catastrophic forgetting or residual data leakage. This project implements unlearning strategies to address:
* **The Privacy Risk:** Left unchecked, models can memorize and leak sensitive training data via extraction attacks.
* **The Technical Challenge:** Traditional retraining is computationally prohibitive. Machine unlearning offers a fraction-of-the-cost alternative, but risks degrading the model's overall intelligence.

## Technical Stack & Frameworks

* **Base Models:** LLaMA-2 / Pythia (Fine-tuned on fictitious author profiles)
* **Dataset:** TOFU (Task of Fictitious Unlearning) — 200 fictitious author profiles consisting of QA pairs to strictly control and isolate the target forget-set.
* **Libraries:** PyTorch, Hugging Face (Transformers, PEFT, Accelerate)
* **Compute Environment:** High-Performance Computing (HPC) Slurm Cluster

## Unlearning Strategies Implemented

This repository explores and compares multiple foundational approaches to machine unlearning:

1. **Gradient Ascent (GA):** Maximizes the loss on the forget-set to forcefully erase the target knowledge.
2. **Gradient Ascent with Random Labeling:** Instead of just increasing loss, this forces the model to map forget-set queries to completely random tokens.
3. **Kullback-Leibler (KL) Divergence Regularization:** Pairs Gradient Ascent on the forget-set with a regularizer that minimizes the drift between the unlearned model and the original fine-tuned model on the retain-set.

## Evaluation Metrics

We leverage the unified TOFU evaluation suite to quantify performance:

* **Forget-Set Performance:** Evaluates the model's perplexity and generation truthfulness on the forgotten data. Ideally, the unlearned model's performance should match a "Gold Model" trained entirely without the forget-set.
* **Retain-Set Performance:** Measures baseline knowledge retention on the data the model is supposed to keep.
* **Model Utility:** Evaluates general capabilities on out-of-distribution QA data to ensure the model's logical reasoning remains intact.
