<p align="center">
    <h1 align="center">RLRF – Competitive Search Agent Design via Reinforcement Learning from Ranker Feedback</h1>
</p>

<p align="center">
	<!-- local repository, no metadata badges. -->
<p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)  
- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Modules](#modules)  
- [Getting Started](#getting-started)  
  - [Installation](#installation)  
  - [Usage](#usage)  
  - [Configuration](#configuration)  
</details>
<hr>

## Overview

<p>This repository contains the official code for the paper “RLRF: Competitive Search Agent Design via Reinforcement Learning from Ranker Feedback” (arXiv: 2510.04096).  
RLRF is a methodology that aligns the LLM with the competitive ranking objective. Unlike traditional prompting-based methods, RLRF aligns LLMs with ranking objectives by learning directly from ranker feedback — the ordering of documents produced by a ranker. Using reinforcement learning techniques, the model learns to modify content in a way that consistently improves its rank in a competitive environment.</p>

---

## Features

1. **Reinforcement Learning from Ranker Feedback (RLRF) Methodology**  
   The framework aligns large language models with ranking objectives by transforming ranker outputs (document orderings) into preference feedback used as reinforcement learning signals.  

2. **Flexible Reinforcement Learning Alignment (DPO / PPO)**  
   The code supports two reinforcement learning algorithms for agent alignment:
   - **DPO (Direct Preference Optimization):** Directly optimizes likelihood-based approach on ranker-derived document preferences, removing the need for explicit reward modeling.  
   - **PPO (Proximal Policy Optimization):** A reward-based policy optimization method that leverages a trained reward model for training.  

3. **Two Synthetic Data Generation Paradigms**  
   - **Static Generation (SG):** Produces a preference dataset (document pairs for each query) using only the LLM and the ranker.  
   - **Dynamic Generation (DG):** Simulates iterative multi-agent ranking competitions to produce richer preference datasets that capture strategic behavior and competition interactions.
     
4. **Reward Model Training Option**  
   The code includes an optional module for training a reward model.  
   The model learns to approximate the ranker’s implicit scoring behavior, converting pairwise or listwise document preferences into continuous reward estimates.  

---

## Repository Structure

```sh
└── RLRF-Competitive-Search-Agent-Design-via-Reinforcement-Learning-from-Ranker-Feedback/
    ├── Agent_Training/
    │   ├── main.py
    │   ├── RLRF_main.py
    │   ├── ranker.py
    │   ├── reward_model.py
    ├── Training_Data_Processing/
    │   ├── Llems_history_into_training_data_format.py
    │   ├── load_data.py
    │   ├── ms-marco.py
    ├── evaluation/
    │   ├── faith_for_offline_evaluation.py
    │   ├── faith_on_Lemss_log.py
    │   ├── rank_promotion_for_offline_evaluation.py
    │   ├── statistical_tests_on_competition_log.py
    │   ├── evaluation_guide.py
    ├── config.json
    ├── prompts.py
    ├── README.md
    └── requirements.txt
    └── conda_requirements.txt

```

## Modules

<details closed><summary>Agent_Training</summary>

| File / Folder | Summary |
| --- | --- |
| *main.py* | Contains the RLRF pipeline; load models, datasets (or create them) and manage training. |
| *RLRF_main.py* | Implement reward model training and LLM alignment. |
| *ranker.py* | Contains the different ranker classes, like mean-pooling rankers. |
| *reward_model.py* | Manage the reward model interface |

</details>

<details closed><summary>Training_Data_Processing</summary>

| File / Folder | Summary |
| --- | --- |
| *Llems_history_into_training_data_format.py* | Convert competition history into preference dataset format. |
| *load_data.py* | Loading and processing of different dataset types. |
| *ms-marco.py* | Contains the code for processing ms-marco dataset as we explained in the paper. |

</details>

<details closed><summary>evaluation</summary>

| File / Folder | Summary |
| --- | --- |
| *faith_for_offline_evaluation.py* | Calculate agent's faithfulness on offline evaluation log. |
| *faith_on_Lemss_log.py* | Calculate agent's faithfulness for 'Lemss' competition log. |
| *rank_promotion_for_offline_evaluation.py* | Calculate agent's scaled promotion on offline evaluation log. |
| *statistical_tests_on_competition_log.py* | Performs win-rate tests for an agent compared to the second-best agent in the competition.  |

</details>

</details>

<details closed><summary>General</summary>

| File / Folder | Summary |
| --- | --- |
| *config.py* | Contains the configuration settings for the LLM's alignment process (training, datasets, etc.). |
| *requirements.txt* | Lists the required Python packages and dependencies for the project. |
| *conda_requirements.txt* | Lists the required conda requirements for the project. |
| *prompts.py* | Contains the LLM's different prompts for dataset creation. |

</details>
conda_requirements.txt
---

## Getting Started

**System Requirements:**

- **Python:** 3.10+
- **Cuda:** 11.8+

### Installation

<h4>From <code>source</code></h4>

> 1. Clone the repository:
> ```bash
> git clone https://github.com/sagie-dekel/RLRF-Competitive-Search-Agent-Design-via-Reinforcement-Learning-from-Ranker-Feedback.git
> ```
>
> 2. Move into the project directory:
> ```bash
> cd RLRF-Competitive-Search-Agent-Design-via-Reinforcement-Learning-from-Ranker-Feedback
> ```
>
> 3. Create Conda environment (optional):
> ```bash
> conda create --name <env name> --file <conda_requirements.txt file path>
> ```
> > 4. Install dependencies:
> ```bash
> pip install -r requirements.txt
> ```

### Usage

> 1. Edit `config.json` to define the experiment parameters: agent (LLM) and ranker types, reward model, training hyperparameters, and datasets.  
> 2. Run a training or evaluation command:
> ```bash
> python Agent_Training/main.py --json_file config.json
> ```
> or
> ```bash
> accelerate launch Agent_Training/main.py --json_file config.json
> ```
> to use accelerate or Deepspeed training options.

### Configuration

Example `config.json`:
```json
{
  "model_RLRF_name_or_path": "mistralai/Ministral-8B-Instruct-2410",
  "RM_name_or_path": "intfloat/e5-large-unsupervised" 
  "ranker_model_name": null,

  "queries_path": "ms-marco_queries.csv",
  "data_path": "competition_history.csv",
  "RM_train_data": "",

  "add_classification_layer_to_reward_model": true,
  "train_full_RM": true, 
  "train_RM": false,
  "relative_order": true,
  "RM_activation_function": "Sigmoid", 
  "create_perfernces_dataset": true, 
  "perfernces_dataset_path": "perfernces_dataset_from_ranker.csv",
  "value_head_needed": false, 
  "use_baseline_doc_for_perfernces_dataset": true,
  "baseline_doc_column": "rejected",
  "baseline_document_path": "perfernces_dataset_robust_queries_with_ranker.csv",
  
  "HP_RLRF": {
    "learning_rate": [1e-6, 1.41e-5],
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 4,
    "beta": 0.1,
    "label_smoothing": 0.0,
    "sync_ref_model": true,
    "ref_model_mixup_alpha": 0.6,
    "ref_model_sync_steps": 256,
    "logging_strategy": "epoch",
    "use_weighting": true,
    "disable_dropout": false,
    "loss_type": "sigmoid",
    "disable_dropout": false,
    "save_steps": 10000,
    "rpo_alpha": null,
    "bf16": true,
    "fp16": false,
    "do_eval": true,
    "eval_strategy": "epoch",
    "eval_on_start": true,
    "torch_dtype": "bfloat16",
    "num_layers_to_train_RLRF": 100,
    "N_sample": 5,
    "eval_dataset_path": null,
    "eval_strategy": "epoch",
    "eval_on_start": true,
    "eval_dataset_path": "",
    "logging_dir": "",
    "model_save_path": "", 
    "model_config_kwargs_RLRF": {
        "attention_dropout": 0.1
    }
  },

  "HP_RM": {
    "batch_size": 10,
    "learning_rate_RM": 1.41e-5,
    "epochs": 12,
    "RM_train_loss_func": "GPTRewardLoss",
    "model_save_path": "", 
    "log_path": "",
    "base_model_config_kwargs_RM": {
        "attention_probs_dropout_prob": 0,
        "hidden_dropout_prob": 0
    }
  }
}
```

Note you may use hyperparameter lists for multiple agents training.

### Configuration Overview

Below is a description of each top-level argument in the configuration file (`config.json`), explaining its purpose and role within the RLRF training pipeline.

#### Core Models

| Key | Description |
| --- | --- |
| **`model_RLRF_name_or_path`** | Path or Hugging Face name of the **policy model** to be aligned through RLRF (e.g., the base LLM used for DPO/PPO training). |
| **`RM_name_or_path`** | Path or Hugging Face name of the **reward model** or other encoder model (possibly a ranker such as E5 or Contriever) |
| **`ranker_model_name`** | Optional **ranker identifier** for dataset creation. |

---

#### Data Sources

| Key | Description |
| --- | --- |
| **`queries_path`** | Path to the file containing **search queries** used during training or dataset creation. |
| **`RM_train_data`** | Path to **reward model training data**, if fine-tuning of RM training is required. |

---

#### Reward Model Configuration

| Key | Description |
| --- | --- |
| **`add_classification_layer_to_reward_model`** | Adds a classification head to the reward model (for preference or relevance prediction). |
| **`train_full_RM`** | If `true`, fine-tunes all layers of the reward model; otherwise, only trains the added head. |
| **`train_RM`** | Enables or disables reward model training. |
| **`relative_order`** | Does the data consist of document pairs (relative order) or a document and his true score. |
| **`RM_activation_function`** | Activation function for the reward model output layer (commonly `"Sigmoid"`). |

---

#### Preferences Dataset Settings

| Key | Description |
| --- | --- |
| **`create_perfernces_dataset`** | If `true`, generates a **preference dataset** based on ranker feedback. |
| **`perfernces_dataset_path`** | Output path for the generated **preferences dataset** CSV file. |
| **`baseline_document_path`** | Path to the CSV containing **baseline documents** for preference dataset generation. |
| **`use_baseline_doc_for_perfernces_dataset`** | Whether to use a **baseline document** when forming preference pairs. |
| **`baseline_doc_column`** | Name of the column used as the **baseline document** in pair construction. |

---

#### Hyperparameter Blocks

| Key | Description |
| --- | --- |
| **`value_head_needed`** | Whether the LLM needs a value head added for the RL algorithm. |
| **`HP_RLRF`** | Configuration for RLRF alignment training (policy model fine-tuning). See the trl library documentation for more argument options. |
| **`HP_RM`** | Configuration for Reward Model (RM) training. |

---
