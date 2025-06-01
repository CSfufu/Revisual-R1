<div align="center">
  <h1 style="margin: 0; font-size: 1.8em;">
    <img src="./figures/logo.png" alt="Revisual Icon" width="50" style="vertical-align: middle; margin-right: 10px;">
    Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning
  </h1>
</div>

## 📚 Overview



## ⚡ News

- [2025/06/02] 🔥 Revisual-R1 paper available on arxiv.

## 🚧 TODO

We are preparing to complete these tasks over the next few weeks, please stay tuned!

- 🚧 We are going to release the training datasets(Coldstart, MRL, TRL).
- 🚧 We are going to release the checkpoint.
- 🚧 We are in the process of training for 32B & 3B Revisual-R1 and will release them when we finish.

## 📖 Introduction

This paper introduces **ReVisual-R1**, a 7B open-source MLLM designed to address prevalent challenges in cultivating sophisticated multimodal reasoning. 
By systematically integrating a strategic, high-difficulty text-only cold-start phase for foundational reasoning, a Multimodal RL stage employing GRPO stabilized by our novel **Prioritized Advantage Distillation (PAD)** mechanism and guided by rule-based rewards including an Efficient-Length Reward, and a final TextRL refinement phase, our structured three-stage curriculum demonstrates that thoughtful data strategy and targeted algorithmic optimizations are pivotal. ReVisual-R1 achieves **SOTA** performance among open-source 7B models on a suite of challenging visuo-mathematical and reasoning benchmarks. 
This work underscores that careful curriculum design and algorithmic enhancements, rather than sheer model scale, can unlock robust, self-reflective multimodal reasoning. 

### 🔑 Key Features


1. **Cold-Start Insights:** We reveal that existing multimodal cold-start corpora lack sufficient difficulty and show that a high-complexity, text-centric warm-up is critical for fostering advanced visual reasoning.

2. **Stable RL Optimisation:** We introduce *Prioritised Advantage Distillation* (PAD) to overcome gradient stagnation, enabling stable and sample-efficient reinforcement learning for MLLMs.

3. **Staged Curriculum & Model:** We design a three-phase training pipeline—text warm-up, multimodal RL with PAD, and text RL—culminating in *ReVisual-R1*, the first open-source 7 B model with self-critical, multi-hop reasoning that rivals proprietary systems.



## 🍭 Results

<img src="./figures/results.png" alt="Revisual results" >

**ReVisual-R1** presents strong performance in challenging visual-mathematical reasoning tasks, while simultaneously preserving strong general-purpose text skills. 

