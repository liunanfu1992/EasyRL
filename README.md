## EasyRL

EasyRL is a from‑scratch, lightweight reinforcement learning framework for large language models (LLMs).  
It adopts a modern **train–inference decoupling** architecture that aligns with industry‑grade AI infra (e.g., verl), while keeping the implementation small enough to be easily read and modified by students and researchers.  
This repository originated as the author’s final project for the undergraduate course *Python Programming*.

### 1. Design Philosophy

- **From‑scratch implementation**  
  EasyRL avoids relying on large, monolithic RL libraries. Instead, it directly implements the key components required to apply policy‑gradient methods to LLMs, including sampling, reward computation, advantage estimation, and parameter updates.

- **Lightweight and readable**  
  The project deliberately keeps the number of modules and external dependencies small. The core training pipeline is organized around a few clearly defined components: data processing, inference engine, reward evaluator, advantage estimator, training backend, and trainer.  
  The goal is to reduce the “codebase size barrier” so that readers can understand the overall architecture in a limited amount of time.

- **Modern train–inference decoupling**  
  EasyRL follows an actor–learner style design: inference is handled by a vLLM‑based inference engine, while training is handled by a PyTorch FSDP‑based backend.  
  The two sides communicate through unified data interfaces and model checkpoints, reflecting the train–inference separation pattern commonly adopted in contemporary LLM systems.

- **Algorithm‑agnostic and extensible**  
  Although this repository currently provides a GRPO‑style example implementation, the framework itself is not tied to GRPO.  
  The reward evaluator, advantage estimator, and training backend are decoupled through clear interfaces, allowing students and researchers to implement or compare different RL algorithms and variants without rewriting the entire engineering scaffold.

### 2. Architecture Overview

At a high level, one full training loop in EasyRL can be summarized as:

> Samples from the dataset → inference engine generates multiple responses → reward evaluation (task‑specific scoring) → advantage estimation (e.g., GRPO‑style group normalization in the current implementation) → FSDP backend computes old/new policy log‑probs and losses → parameter updates and checkpoint export → updated model used in the next inference round.

Around this loop, the framework is structured into several components with distinct responsibilities:

- **Inference Engine (vLLM‑based)**  
  Executes high‑throughput sampling for given prompts, and distinguishes between training and validation sampling strategies (e.g., temperature, top‑p, top‑k, and maximum generation length).  
  The generated candidates are used both for training (policy updates) and for estimating validation metrics.

- **Reward Verifier (reward evaluation module)**  
  The current example targets mathematical problem solving. It uses `math-verify` and symbolic computation tools to strictly check mathematical equivalence between model outputs and reference solutions, yielding binary (0/1) accuracy‑style rewards.  
  At an abstract level, this module maps “model outputs + ground truth” to scalar rewards and can be replaced by task‑specific reward functions such as preference models, rule‑based systems, or human feedback.

- **Advantage Calculator (advantage estimator)**  
  Given reward signals produced by the Reward Verifier, this component computes the advantages used in policy updates. Its concrete form can vary with the chosen RL algorithm, e.g., baseline‑based advantages, GAE‑style advantages, or group‑normalized advantages.  
  The current implementation provides a GRPO‑style example based on within‑group normalization, but the module is designed as a generic insertion point for different advantage estimation schemes rather than being restricted to group‑based methods.

- **FSDP Training Backend (PyTorch FSDP‑based)**  
  Maintains model parameters in a multi‑process FSDP setup and, given old policy log‑probs, advantages, and the current model outputs, computes policy‑gradient losses (optionally with KL regularization), performs backpropagation, and updates parameters.  
  The backend can export both full checkpoints suitable for continued FSDP training and HuggingFace‑compatible weights consumable by the inference engine.

- **Trainer (training driver and orchestrator)**  
  As the high‑level orchestrator, the trainer stitches together data loading, inference, reward evaluation, advantage estimation, and FSDP updates, while managing the training/validation loop, logging, and checkpointing.  
  For readers interested in an end‑to‑end view of “RL on LLMs,” the trainer offers a clear entry point into the overall pipeline.

The overall data flow can be sketched as:

> Dataset → Inference Engine → Reward Verifier → Advantage Calculator → FSDP Training Backend → model checkpoints → Inference Engine (next iteration)

### 3. Quick Start

This section outlines a minimal working example to help readers run the full pipeline before diving into implementation details.

- **Environment setup**  
  - **Python version**: Python 3.10 or later is recommended.  
  - **Hardware**: A multi‑GPU environment is recommended to fully leverage vLLM and FSDP (the default configuration assumes several GPUs).  
  - **Dependencies**: Dependencies are specified in `pyproject.toml`. A typical editable installation can be done via:

```bash
pip install -e .
```

- **Data preparation**  
  EasyRL expects training and validation data in Parquet format. In the example configuration, the minimal required fields are:

  - **Training data**  
    - `prompt`: the input content, e.g., dialogue context with system prompts or problem statements;  
    - `ground_truth`: the reference solution, which may be written in LaTeX or wrapped in specific tags.
  - **Validation data**  
    - In addition to the training fields, it contains an `extra_info` field with metadata such as `data_source` and `pass@k`, enabling grouped evaluation across different sources and pass@k settings.

  In the default setup, data paths and field names follow the provided configuration file. To migrate to a new dataset, it is usually sufficient to preserve the same semantic fields and update the paths in the configuration.

- **Running a training example**  
  After installing dependencies and preparing the data, you can launch a training run from the repository root using the provided script:

```bash
bash easyrl/trainer/GRPO_trainer/start_trainer.sh
```

  Alternatively, you can invoke the module entry point directly and override selected Hydra configuration options (such as model path, data paths, or learning rate):

```bash
python -m easyrl.trainer.GRPO_trainer.grpo_trainer \
  paths.model_path=/path/to/your/model \
  paths.train_data=/path/to/your/train.parquet \
  paths.valid_data=/path/to/your/valid.parquet
```

  In practice, you may want to adjust the number of GPUs, batch sizes, sampling settings, and the number of training steps according to your hardware constraints and experimental goals.

- **Logging and visualization**  
  During training, EasyRL logs training losses, validation metrics, and reward‑related statistics to a TensorBoard log directory (configured via the `monitor` section in the config).  
  You can visualize training progress with:

```bash
tensorboard --logdir /path/to/your/logs
```

### 4. Extending EasyRL

One of EasyRL’s main goals is to provide a clear and easily modifiable reference implementation for exploring different RL ideas on LLMs. This section outlines several common extension directions at a conceptual level.

- **Replacing or adding RL algorithms**  
  The current repository includes a GRPO‑style example whose core idea is to exploit within‑group relative rewards to construct advantages and perform policy‑gradient updates.  
  Researchers and students can keep the inference engine and FSDP backend interfaces intact, and implement new training loops and loss functions in the trainer (e.g., other policy‑gradient methods or offline RL variants) to conduct apples‑to‑apples comparisons within the same engineering framework.

- **Custom reward functions**  
  The current Reward Verifier targets math problem solving and uses symbolic reasoning and formatting constraints to approximate an “oracle judge” that yields strict correctness rewards.  
  At the interface level, a reward module only needs to consume “model outputs + ground truth” and produce scalar rewards, making it natural to plug in preference‑model‑based scorers, rule‑based systems, or other task‑specific evaluators.

- **Porting to new tasks and models**  
  To apply EasyRL to a new task, it is typically sufficient to:  
  - Adjust the data schema and preprocessing to reflect the new task;  
  - Implement a suitable Reward Verifier for the new objective;  
  - Point the configuration to a different `model_path` and adjust inference‑related parameters.  
  Under these conditions, most of the training pipeline can be reused, allowing you to focus on the algorithm and task design itself.

### 5. License

This project is released under the **MIT License**.  
Please refer to the `LICENSE` file in the repository root for the full license text.


