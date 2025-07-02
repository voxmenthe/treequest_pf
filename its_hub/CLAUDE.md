# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

**its_hub** is a library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks. The core architecture uses abstract base classes to define clean interfaces between components.

### Key Base Classes (`its_hub/base.py`)
- `AbstractLanguageModel`: Interface for LM generation and evaluation
- `AbstractScalingAlgorithm`: Base for all scaling algorithms with unified `infer()` method
- `AbstractScalingResult`: Base for algorithm results with `the_one` property
- `AbstractOutcomeRewardModel`: Interface for outcome-based reward models
- `AbstractProcessRewardModel`: Interface for process-based reward models (step-by-step scoring)

### Main Components

#### Language Models (`its_hub/lms.py`)
- `OpenAICompatibleLanguageModel`: Primary LM implementation supporting vLLM and OpenAI APIs
- `StepGeneration`: Handles incremental generation with configurable step tokens and stop conditions
- Supports async generation with concurrency limits and backoff strategies

#### Algorithms (`its_hub/algorithms/`)
All algorithms follow the same interface: `infer(lm, prompt, budget, return_response_only=True)`

- **Self-Consistency**: Generate multiple responses, select most common answer
- **Best-of-N**: Generate N responses, select highest scoring via outcome reward model  
- **Beam Search**: Step-by-step generation with beam width, uses process reward models
- **Particle Filtering/Gibbs**: Probabilistic resampling with process reward models

#### Integration (`its_hub/integration/`)
- `LocalVllmProcessRewardModel`: Integrates with reward_hub library for process-based scoring
- `iaas.py`: Inference-as-a-Service FastAPI server providing OpenAI-compatible chat completions API with budget parameter for inference-time scaling

### Budget Interpretation
The budget parameter controls computational resources allocated to each algorithm. Different algorithms interpret budget as follows:
- **Self-Consistency/Best-of-N**: Number of parallel generations to create
- **Beam Search**: Total generations divided by beam width (controls search depth)
- **Particle Filtering**: Number of particles maintained during sampling

### Step Generation Pattern
The `StepGeneration` class enables incremental text generation:
- Configure step tokens (e.g., "\n\n" for reasoning steps)
- Set max steps and stop conditions
- Post-processing for clean output formatting

### Typical Workflow
1. Start vLLM server with instruction model
2. Initialize `OpenAICompatibleLanguageModel` pointing to server
3. Create `StepGeneration` with step/stop tokens appropriate for the task
4. Initialize reward model (e.g., `LocalVllmProcessRewardModel`)
5. Create scaling algorithm with step generation and reward model
6. Call `infer()` with prompt and budget

### Mathematical Focus
The library is optimized for mathematical reasoning:
- Predefined system prompts in `its_hub/utils.py` (SAL_STEP_BY_STEP_SYSTEM_PROMPT, QWEN_SYSTEM_PROMPT)
- Regex patterns for mathematical notation (e.g., `r"\boxed"` for final answers)
- Integration with math_verify for evaluation
- Benchmarking on MATH500 and AIME-2024 datasets