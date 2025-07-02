# its_hub API Reference

This comprehensive API reference reveals the architectural design principles and implementation details of the `its_hub` library, providing deep insights into how the library works and why it's designed this way.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Design Principles](#core-design-principles)
3. [Base Classes](#base-classes)
4. [Language Models](#language-models)
5. [Scaling Algorithms](#scaling-algorithms)
6. [Reward Models](#reward-models)
7. [Integration Components](#integration-components)
8. [Step Generation](#step-generation)
9. [Error Handling](#error-handling)

## Architecture Overview

`its_hub` follows a clean interface-based architecture using Abstract Base Classes (ABCs) to ensure extensibility and modularity. The design separates concerns into:

1. **Language Model Abstraction** - Unified interface for text generation
2. **Scaling Algorithm Abstraction** - Common interface for all inference-time scaling techniques
3. **Reward Model Abstractions** - Separate interfaces for outcome and process rewards
4. **Generation Control** - Step-based generation for incremental text production

## Core Design Principles

### 1. Interface Segregation
Each component has a minimal, focused interface. Language models only need to implement `generate()` and `evaluate()`, while algorithms only need `infer()`.

### 2. Dependency Inversion
Algorithms depend on abstractions (interfaces) not concrete implementations. This allows swapping LM providers, reward models, or algorithms without changing calling code.

### 3. Budget Abstraction
The `budget` parameter is intentionally abstract - each algorithm interprets it differently based on its computational strategy.

### 4. Result Encapsulation
The `AbstractScalingResult` allows algorithms to return rich information while maintaining a simple interface through the `the_one` property.

## Base Classes

### AbstractLanguageModel

The foundation for all language model implementations:

```python
from abc import ABC, abstractmethod
from its_hub.types import ChatMessage

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
    ) -> str | list[str]:
        """Generate text from messages (single or batched)"""
        pass
    
    @abstractmethod
    def evaluate(self, prompt: str, generation: str) -> list[float]:
        """Evaluate token-level likelihoods"""
        pass
```

**Design Rationale:**
- Supports both single and batched generation for efficiency
- Separates generation from evaluation for flexibility
- Uses structured `ChatMessage` types for clarity

### AbstractScalingAlgorithm

The unified interface for all scaling algorithms:

```python
class AbstractScalingAlgorithm(ABC):
    @abstractmethod
    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str,
        budget: int,
        return_response_only: bool = True,
    ) -> str | AbstractScalingResult:
        """Run inference with specified budget"""
        pass
```

**Design Rationale:**
- Simple interface enables algorithm interchangeability
- Budget abstraction allows algorithm-specific interpretations
- Optional rich results via `return_response_only` parameter

### AbstractScalingResult

Encapsulates algorithm results:

```python
class AbstractScalingResult(ABC):
    @property
    @abstractmethod
    def the_one(self) -> str:
        """The selected response"""
        pass
```

**Design Rationale:**
- Provides simple access to selected response
- Allows algorithms to return additional metadata
- Enables future extensibility without breaking changes

## Language Models

### OpenAICompatibleLanguageModel

The primary language model implementation supporting both vLLM and OpenAI-compatible APIs:

```python
from its_hub.lms import OpenAICompatibleLanguageModel

# Basic usage
lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8000/v1",
    api_key="NO_API_KEY",
    model_name="model-name",
    system_prompt="Optional system prompt"
)

# Advanced configuration
lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8000/v1",
    api_key="NO_API_KEY", 
    model_name="model-name",
    is_async=True,  # Enable async generation
    max_concurrency=10,  # Limit concurrent requests
    max_tries=8,  # Retry configuration
    temperature=0.7,  # Default temperature
    max_tokens=512,  # Default max tokens
    replace_error_with_message="[Error occurred]"  # Graceful error handling
)
```

**Key Features:**
- **Async Support**: Set `is_async=True` for concurrent generation
- **Concurrency Control**: Limit parallel requests with `max_concurrency`
- **Automatic Retries**: Configurable retry logic with exponential backoff
- **Error Recovery**: Optional error replacement for robustness
- **Batched Generation**: Efficiently generate multiple responses

**Implementation Details:**

```python
# Single generation
response = lm.generate([
    ChatMessage(role="user", content="Solve x^2 = 4")
])

# Batched generation (returns list)
responses = lm.generate([
    [ChatMessage(role="user", content="Solve x^2 = 4")],
    [ChatMessage(role="user", content="Solve x^2 = 9")]
])

# With generation parameters
response = lm.generate(
    messages,
    stop="\n\n",  # Stop sequence
    temperature=0.5,  # Override default
    max_tokens=256  # Override default
)
```

## Scaling Algorithms

All algorithms share the same interface but differ in their strategy and budget interpretation:

### Self-Consistency

Generates multiple responses and selects the most frequent after projection:

```python
from its_hub.algorithms import SelfConsistency
from its_hub.lms import StepGeneration

# Basic self-consistency
sc = SelfConsistency()
response = sc.infer(lm, prompt, budget=8)

# With custom consistency projection
def extract_number(response: str) -> str:
    # Extract numerical answer from response
    import re
    match = re.search(r'\\boxed{([^}]+)}', response)
    return match.group(1) if match else response

sc = SelfConsistency(consistency_space_projection_func=extract_number)

# With step generation for structured reasoning
sg = StepGeneration("\n\n", max_steps=10, stop_pattern=r"\\boxed")
sc = SelfConsistency(step_generation=sg)

# Get full results
result = sc.infer(lm, prompt, budget=16, return_response_only=False)
print(f"Selected: {result.the_one}")
print(f"All responses: {result.samples}")
print(f"Response counts: {result.response_counts}")
```

**Budget**: Number of independent samples to generate

### Best-of-N (BoN)

Generates N responses and selects the highest-scoring one:

```python
from its_hub.algorithms import BestOfN
from its_hub.base import AbstractOutcomeRewardModel

# Custom outcome reward model
class CustomOutcomeRewardModel(AbstractOutcomeRewardModel):
    def score(self, prompt: str, response: str | list[str]) -> float | list[float]:
        # Implement scoring logic
        if isinstance(response, list):
            return [self._score_single(prompt, r) for r in response]
        return self._score_single(prompt, response)
    
    def _score_single(self, prompt: str, response: str) -> float:
        # Example: score based on response length and keywords
        score = len(response) / 1000.0  # Normalize length
        if "therefore" in response.lower():
            score += 0.5
        return score

orm = CustomOutcomeRewardModel()
bon = BestOfN(reward_model=orm)
response = bon.infer(lm, prompt, budget=16)
```

**Budget**: Number of complete responses to generate and score

### Beam Search

Step-by-step generation with beam pruning:

```python
from its_hub.algorithms import BeamSearch
from its_hub.lms import StepGeneration
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

# Configure step generation
sg = StepGeneration(
    step_token="\n\n",
    max_steps=15,
    stop_pattern=r"\\boxed",
    temperature=0.7
)

# Process reward model for step evaluation
prm = LocalVllmProcessRewardModel(
    model_name="reward-model",
    device="cuda:0",
    aggregation_method="prod"  # or "mean", "min", "max"
)

# Beam search with width 4
beam = BeamSearch(
    step_generation=sg,
    process_reward_model=prm,
    beam_width=4
)

# Budget = beam_width * number_of_steps
response = beam.infer(lm, prompt, budget=32)  # 8 steps with beam width 4
```

**Budget**: Total generations (must be divisible by beam_width)

### Particle Filtering/Gibbs

Monte Carlo sampling with resampling based on scores:

```python
from its_hub.algorithms import ParticleFiltering, ParticleGibbs

# Particle Filtering (single iteration)
pf = ParticleFiltering(
    step_generation=sg,
    process_reward_model=prm
)
response = pf.infer(lm, prompt, budget=8)

# Particle Gibbs (multiple iterations)
pg = ParticleGibbs(
    step_generation=sg,
    process_reward_model=prm,
    num_iterations=4,
    selection_method="SAMPLE",  # or "ARGMAX"
    num_ref_particles=1
)
response = pg.infer(lm, prompt, budget=16)  # 4 particles × 4 iterations
```

**Budget**: Number of particles (must be divisible by num_iterations for Gibbs)

## Reward Models

### Process Reward Models

Evaluate partial sequences step-by-step:

```python
from its_hub.base import AbstractProcessRewardModel

class CustomProcessRewardModel(AbstractProcessRewardModel):
    def score(self, prompt: str, steps: list[str]) -> list[float]:
        """Score each step in the reasoning process"""
        scores = []
        for i, step in enumerate(steps):
            # Example: prefer steps with equations
            score = 1.0
            if "=" in step:
                score += 0.5
            if i > 0 and len(step) > len(steps[i-1]):
                score += 0.3  # Reward elaboration
            scores.append(score)
        return scores
```

### Outcome Reward Models  

Evaluate complete responses:

```python
from its_hub.base import AbstractOutcomeRewardModel

class MathCorrectnessRewardModel(AbstractOutcomeRewardModel):
    def score(self, prompt: str, response: str) -> float:
        """Score based on mathematical correctness indicators"""
        score = 0.0
        
        # Check for final answer
        if r"\\boxed" in response:
            score += 2.0
            
        # Check for step-by-step reasoning
        if response.count("\n") > 3:
            score += 1.0
            
        # Penalize very short responses
        if len(response) < 50:
            score -= 1.0
            
        return score
```

## Integration Components

### LocalVllmProcessRewardModel

Integrates with the `reward_hub` library for sophisticated reward models:

```python
from its_hub.integration.reward_hub import LocalVllmProcessRewardModel

prm = LocalVllmProcessRewardModel(
    model_name="Qwen/Qwen2.5-Math-PRM-7B",
    device="cuda:0",
    aggregation_method="prod"  # How to combine step scores
)

# Use with algorithms
from its_hub.algorithms import BeamSearch
beam = BeamSearch(sg, prm, beam_width=4)
```

**Aggregation Methods:**
- `"prod"`: Multiply step scores (strict - all steps must be good)
- `"mean"`: Average step scores (balanced)
- `"min"`: Minimum step score (conservative)
- `"max"`: Maximum step score (optimistic)
- `"model"`: Let the model decide aggregation

## Step Generation

Controls incremental text generation for step-by-step reasoning:

```python
from its_hub.lms import StepGeneration

# Basic configuration
sg = StepGeneration(
    step_token="\n\n",  # Split on double newlines
    max_steps=20,       # Maximum reasoning steps
    stop_pattern=r"\\boxed"  # Stop when answer is boxed
)

# Advanced configuration
sg = StepGeneration(
    step_token=["\n\n", "\n---\n"],  # Multiple possible step tokens
    max_steps=15,
    stop_pattern=r"(\\boxed|Answer:|Final answer:)",
    temperature=0.8,
    include_stop_str_in_output=True,  # Keep stop strings
    temperature_switch=(0.3, "[think]", "[/think]")  # Lower temp in think tags
)

# Use with language model
steps = []
stopped = False
while not stopped and len(steps) < sg.max_steps:
    next_step, stopped = sg.forward(lm, prompt, steps)
    steps.append(next_step)
    
final_response = sg._post_process(steps, stopped)
```

**Key Features:**
- **Flexible Step Tokens**: String or list of strings for splitting
- **Stop Patterns**: Regex patterns to detect completion
- **Temperature Control**: Dynamic temperature adjustment
- **Post-processing**: Clean output formatting

## Error Handling

The library includes sophisticated error handling with retries:

```python
from its_hub.error_handling import APIError, RETRYABLE_ERRORS

# Errors are automatically retried with exponential backoff
lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:8000/v1",
    api_key="key",
    model_name="model",
    max_tries=8,  # Maximum retry attempts
    replace_error_with_message="[Generation failed]"  # Graceful degradation
)

# Custom error handling
try:
    response = lm.generate(messages)
except APIError as e:
    if e.status_code == 429:  # Rate limit
        print("Rate limited, waiting...")
    elif e.status_code >= 500:  # Server error
        print("Server error, retrying...")
    else:
        raise
```

**Retryable Errors:**
- Connection errors
- Timeout errors  
- 5xx server errors
- Rate limiting (429)
- Specific API errors marked as retryable

## Advanced Usage Examples

### Custom Algorithm Implementation

```python
from its_hub.base import AbstractScalingAlgorithm, AbstractScalingResult
from dataclasses import dataclass

@dataclass
class CustomResult(AbstractScalingResult):
    selected_response: str
    confidence: float
    metadata: dict
    
    @property
    def the_one(self) -> str:
        return self.selected_response

class AdaptiveSamplingAlgorithm(AbstractScalingAlgorithm):
    def __init__(self, initial_samples: int = 4):
        self.initial_samples = initial_samples
    
    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str,
        budget: int,
        return_response_only: bool = True
    ) -> str | CustomResult:
        # Generate initial samples
        messages = [[ChatMessage(role="user", content=prompt)] 
                   for _ in range(self.initial_samples)]
        responses = lm.generate(messages)
        
        # Adaptive sampling based on diversity
        remaining_budget = budget - self.initial_samples
        if self._needs_more_samples(responses):
            additional = lm.generate(
                [[ChatMessage(role="user", content=prompt)] 
                 for _ in range(remaining_budget)]
            )
            responses.extend(additional)
        
        # Select best response
        selected = self._select_best(responses)
        
        if return_response_only:
            return selected
        else:
            return CustomResult(
                selected_response=selected,
                confidence=0.95,
                metadata={"total_samples": len(responses)}
            )
```

### Chaining Algorithms

```python
# Use self-consistency to generate candidates, then best-of-n to select
from its_hub.algorithms import SelfConsistency, BestOfN

# First pass: generate diverse candidates
sc = SelfConsistency()
candidates = []
for _ in range(4):
    candidate = sc.infer(lm, prompt, budget=4)
    candidates.append(candidate)

# Second pass: select best candidate
class CandidateScorer(AbstractOutcomeRewardModel):
    def score(self, prompt: str, response: str | list[str]) -> float | list[float]:
        # Score pre-generated candidates
        if isinstance(response, list):
            return [len(r) / 100.0 for r in response]
        return len(response) / 100.0

bon = BestOfN(CandidateScorer())
# Create a mock LM that returns our candidates
final_response = bon.infer(MockLM(candidates), prompt, budget=len(candidates))
```

This API reference provides a comprehensive understanding of the its_hub library's architecture, design principles, and implementation details. The clean abstraction layers make it easy to extend for domains beyond mathematical reasoning while maintaining the core benefits of inference-time scaling.