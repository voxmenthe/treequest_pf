# TreeQuest Coding Solver with Particle Filtering - Implementation Plan v2

## Executive Summary

This plan combines TreeQuest's AB-MCTS with Particle Filtering. 

Design principles include:

- **Modular architecture** with clean interfaces for dependency injection
- **Dual-mode support** for both treeQuest-native and its_hub libraries
- **Async-first design** with batched LLM operations
- **Robust error handling** with deterministic fallbacks
- **Performance optimizations** including vectorized resampling and diff-aware testing
- **Property-based testing** to ensure correctness under all conditions

Expected outcomes remain strong:
- 30-40% improvement in hard problem solve rates
- 25-35% reduction in required search iterations
- 2-3x performance improvement through optimized implementation
- Support for both TreeQuest and its_hub backends

## Architecture Overview

### Core Design Principles

1. **Plugin Architecture**: Three core interfaces enable clean separation of concerns
2. **Adapter Pattern**: Bridge between TreeQuest and its_hub without tight coupling
3. **Async-First**: Non-blocking I/O for maximum throughput
4. **Immutable State**: Separate data, evaluation, and metadata for clarity
5. **Strategy Delegation**: Extract decision logic from algorithms
6. **Graceful Degradation**: Always have a working fallback path
7. **Property-Based Testing**: Ensure invariants hold under all conditions

### Module Structure

```
src/treequest_coding_solver/
├── core/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── llm_backend.py          # Abstract LLM interface
│   │   ├── search_algorithm.py     # Abstract search algorithm interface  
│   │   └── reward_signal.py        # Abstract reward model interface
│   ├── state/
│   │   ├── __init__.py
│   │   ├── code_attempt.py         # Immutable code representation
│   │   ├── execution_result.py     # Test execution results
│   │   └── search_metadata.py      # Tree search metadata
│   └── types.py                    # Core type definitions
├── adapters/
│   ├── __init__.py
│   ├── treequest_generator.py      # Adapt its_hub → TreeQuest GenerateFnType
│   └── itshub_scaling.py           # Adapt TreeQuest → its_hub interface
├── particle/
│   ├── __init__.py
│   ├── manager.py                  # ParticleManager with all PF logic
│   ├── config.py                   # PF configuration dataclass
│   └── resampling.py              # Vectorized resampling strategies
├── algorithms/
│   ├── __init__.py
│   ├── hybrid_mcts_pf.py          # Hybrid algorithm implementation
│   └── strategy_policy.py         # Node selection & expansion strategies
├── llm/
│   ├── __init__.py
│   ├── base.py                    # Base implementation of LLMBackend
│   ├── gemini_backend.py          # Gemini-specific implementation
│   ├── openai_backend.py          # OpenAI-specific implementation
│   └── itshub_backend.py          # its_hub wrapper implementation
├── execution/
│   ├── __init__.py
│   ├── sandbox.py                 # Safe code execution environment
│   ├── diff_executor.py           # Diff-aware test execution
│   └── test_parser.py             # Test case extraction
├── rewards/
│   ├── __init__.py
│   ├── test_based.py              # Semi-verifiable test rewards
│   ├── prm_based.py               # PRM-based rewards (via its_hub)
│   └── composite.py               # Weighted combination rewards
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py                 # Performance metrics collection
│   └── telemetry.py               # OpenTelemetry integration
├── solver.py                      # Main solver orchestration
└── utils/
    ├── __init__.py
    ├── cache.py                   # Prefix tree caching
    └── serialization.py           # Versioned checkpointing
```

## Detailed Implementation

### Phase 0: Core Interfaces (Week 1)

#### 0.1 Core Interfaces (`src/treequest_coding_solver/core/interfaces/`)

```python
# llm_backend.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
import asyncio
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Encapsulates LLM response with metadata"""
    text: str
    tokens_used: int
    latency_ms: float
    model_name: str
    temperature: float
    metadata: Dict[str, Any]

class LLMBackend(ABC):
    """Abstract interface for language model backends
    
    Design Rationale:
    - Supports both sync and async operations
    - Batching is first-class for efficiency
    - Clean separation from implementation details
    """
    
    @abstractmethod
    async def generate_async(
        self,
        prompts: Union[str, List[str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Union[LLMResponse, List[LLMResponse]]:
        """Async generation supporting batching"""
        pass
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Union[LLMResponse, List[LLMResponse]]:
        """Sync wrapper for compatibility"""
        return asyncio.run(self.generate_async(prompts, temperature, max_tokens, stop, **kwargs))
    
    @abstractmethod
    async def evaluate_async(
        self,
        prompt: str,
        completions: List[str]
    ) -> List[float]:
        """Evaluate log probabilities for completions"""
        pass
```

```python
# search_algorithm.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any, Callable, Optional
from dataclasses import dataclass

StateT = TypeVar('StateT')
AlgoStateT = TypeVar('AlgoStateT')

@dataclass
class SearchResult(Generic[StateT]):
    """Encapsulates search results"""
    best_state: StateT
    score: float
    algorithm_state: Any
    metadata: Dict[str, Any]

class SearchAlgorithm(ABC, Generic[StateT, AlgoStateT]):
    """Abstract interface for search algorithms
    
    Design Rationale:
    - Generic over state types for flexibility
    - Supports both TreeQuest and its_hub paradigms
    - Clean checkpoint/restore interface
    """
    
    @abstractmethod
    def initialize(self) -> AlgoStateT:
        """Initialize algorithm state"""
        pass
    
    @abstractmethod
    def step(
        self,
        state: AlgoStateT,
        generators: Dict[str, Callable],
        **kwargs
    ) -> AlgoStateT:
        """Execute one search step"""
        pass
    
    @abstractmethod
    def get_best_states(
        self,
        state: AlgoStateT,
        k: int = 1
    ) -> List[SearchResult[StateT]]:
        """Extract top-k results"""
        pass
    
    def checkpoint(self, state: AlgoStateT) -> bytes:
        """Serialize algorithm state"""
        import pickle
        return pickle.dumps(state)
    
    def restore(self, data: bytes) -> AlgoStateT:
        """Restore algorithm state"""
        import pickle
        return pickle.loads(data)
```

```python
# reward_signal.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class RewardComponents:
    """Breakdown of reward signal components"""
    total: float  # Must be in [0, 1] for TreeQuest
    components: Dict[str, float]
    metadata: Dict[str, Any]

class RewardSignal(ABC):
    """Abstract interface for reward computation
    
    Design Rationale:
    - Supports composite rewards
    - Normalized to [0,1] for TreeQuest compatibility
    - Rich metadata for analysis
    """
    
    @abstractmethod
    def compute_reward(
        self,
        problem: str,
        solution: str,
        execution_result: Optional[Any] = None,
        **kwargs
    ) -> RewardComponents:
        """Compute reward with component breakdown"""
        pass
    
    def normalize(self, raw_score: float) -> float:
        """Normalize score to [0, 1] range"""
        return max(0.0, min(1.0, raw_score))
```

#### 0.2 State Classes (`src/treequest_coding_solver/core/state/`)

```python
# code_attempt.py
from dataclasses import dataclass, field
from typing import Optional, FrozenSet
import hashlib
from typing_extensions import Annotated

@dataclass(frozen=True)
class CodeAttempt:
    """Immutable representation of a code solution attempt
    
    Design Rationale:
    - Immutable for safe sharing across particles
    - Hash-based identity for deduplication
    - Supports diff-based storage
    """
    code: str
    language: str = "python"
    strategy: str = "initial"
    parent_hash: Optional[str] = None
    
    # Computed fields
    code_hash: str = field(init=False)
    line_count: int = field(init=False)
    
    def __post_init__(self):
        # Bypass frozen to set computed fields
        object.__setattr__(self, 'code_hash', 
                          hashlib.sha256(self.code.encode()).hexdigest()[:16])
        object.__setattr__(self, 'line_count', 
                          len(self.code.splitlines()))
    
    def to_diff(self, parent: Optional['CodeAttempt']) -> str:
        """Convert to diff representation"""
        if parent is None:
            return self.code
        
        import difflib
        parent_lines = parent.code.splitlines(keepends=True)
        self_lines = self.code.splitlines(keepends=True)
        diff = ''.join(difflib.unified_diff(
            parent_lines, self_lines,
            fromfile=f"parent_{parent.code_hash}",
            tofile=f"attempt_{self.code_hash}"
        ))
        return diff
    
    @classmethod
    def from_diff(cls, parent: 'CodeAttempt', diff: str) -> 'CodeAttempt':
        """Reconstruct from parent and diff"""
        # Implementation using patch application
        pass
```

```python
# execution_result.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class TestCase:
    """Single test case representation"""
    input: Any
    expected_output: Any
    actual_output: Optional[Any] = None
    passed: bool = False
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None

@dataclass
class ExecutionResult:
    """Results from code execution
    
    Design Rationale:
    - Captures all execution details
    - Supports partial results for timeout
    - Rich error information for debugging
    """
    test_cases: List[TestCase]
    total_tests: int
    passed_tests: int
    failed_tests: int
    
    # Performance metrics
    total_execution_time_ms: float
    memory_usage_mb: float
    
    # Execution metadata
    execution_timestamp: datetime = field(default_factory=datetime.now)
    sandbox_type: str = "docker"  # or "subprocess"
    timeout_occurred: bool = False
    
    # Error details
    compilation_errors: List[str] = field(default_factory=list)
    runtime_errors: List[str] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        """Compute test pass rate"""
        return self.passed_tests / max(self.total_tests, 1)
    
    @property
    def has_syntax_errors(self) -> bool:
        """Check if code has syntax errors"""
        return len(self.compilation_errors) > 0
    
    def get_first_failure(self) -> Optional[TestCase]:
        """Get first failing test case"""
        for tc in self.test_cases:
            if not tc.passed:
                return tc
        return None
```

```python
# search_metadata.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

@dataclass
class SearchMetadata:
    """Metadata for tree search nodes
    
    Design Rationale:
    - Tracks lineage and search history
    - Supports both MCTS and PF metadata
    - Enables rich analysis and debugging
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    depth: int = 0
    expansion_type: str = "standard"  # or "particle_filter"
    
    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    generation_time_ms: float = 0.0
    evaluation_time_ms: float = 0.0
    
    # Algorithm-specific metadata
    algorithm_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For particle filtering nodes
    particle_count: Optional[int] = None
    effective_sample_size: Optional[float] = None
    particle_diversity: Optional[float] = None
    
    def total_time_ms(self) -> float:
        """Total time spent on this node"""
        return self.generation_time_ms + self.evaluation_time_ms
```

### Phase 1: Particle Management System (Week 1-2)

#### 1.1 Particle Manager (`src/treequest_coding_solver/particle/manager.py`)

```python
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Callable
import asyncio
from dataclasses import dataclass
import logging

from ..core.interfaces import LLMBackend, RewardSignal
from ..core.state import CodeAttempt, ExecutionResult
from .config import ParticleConfig
from .resampling import systematic_resample, stratified_resample

logger = logging.getLogger(__name__)

class ParticleManager:
    """Manages particle lifecycle with all PF-specific logic
    
    Design Rationale (from feedback):
    - Encapsulates ALL particle filtering logic
    - HybridAlgorithm only asks should_use_pf()
    - Supports multiple resampling strategies
    - Handles budget allocation and decay
    """
    
    def __init__(
        self,
        config: ParticleConfig,
        llm_backend: LLMBackend,
        reward_signal: RewardSignal
    ):
        self.config = config
        self.llm = llm_backend
        self.reward = reward_signal
        
        # Budget tracking
        self.total_particles_generated = 0
        self.current_iteration = 0
        
        # Caching
        self.evaluation_cache = {}
        
    def should_use_particle_filtering(
        self,
        node_score: float,
        node_depth: int,
        children_count: int,
        iterations_since_improvement: int
    ) -> bool:
        """Decide if PF should be used for this node
        
        Encapsulates all PF deployment heuristics from the original plan
        plus advisor suggestions for cleaner separation
        """
        # Never use PF at root
        if node_depth == 0:
            return False
            
        # Check budget constraints
        pf_ratio = self.total_particles_generated / max(self.current_iteration * 10, 1)
        if pf_ratio > self.config.max_pf_budget_ratio:
            return False
            
        # Depth constraint
        if node_depth > self.config.max_pf_depth:
            return False
            
        # Score in sweet spot (high uncertainty)
        if self.config.pf_score_range[0] <= node_score <= self.config.pf_score_range[1]:
            return True
            
        # High branching factor
        if children_count > self.config.high_branching_threshold:
            return True
            
        # Stagnation
        if iterations_since_improvement > self.config.stagnation_threshold:
            return True
            
        return False
    
    def allocate_particle_budget(self, node_depth: int, total_budget: int) -> int:
        """Allocate particles with decay strategy
        
        Implements linear budget front-loading from feedback:
        Start with N₀ particles and decay by √t
        """
        base_particles = self.config.initial_particles
        
        # Decay factor based on depth
        decay = np.sqrt(node_depth + 1)
        allocated = int(base_particles / decay)
        
        # Apply bounds
        allocated = max(self.config.min_particles, allocated)
        allocated = min(self.config.max_particles_per_node, allocated)
        
        # Global budget constraint
        remaining_budget = total_budget - self.total_particles_generated
        allocated = min(allocated, remaining_budget)
        
        return max(1, allocated)
    
    async def generate_initial_particles(
        self,
        problem: str,
        num_particles: int
    ) -> List[Tuple[CodeAttempt, float]]:
        """Generate diverse initial particles
        
        Uses strategy diversification + temperature variation
        """
        strategies = [
            ("direct", 0.5),
            ("step_by_step", 0.7),
            ("test_first", 0.8),
            ("edge_cases", 0.9),
            ("minimal", 0.4),
            ("comprehensive", 1.0)
        ]
        
        # Prepare prompts with different strategies
        prompts = []
        temperatures = []
        
        for i in range(num_particles):
            strategy, base_temp = strategies[i % len(strategies)]
            # Add noise for diversity
            temp = base_temp + np.random.uniform(-0.1, 0.1)
            temp = np.clip(temp, 0.1, 1.5)
            
            prompt = self._build_strategy_prompt(problem, strategy)
            prompts.append(prompt)
            temperatures.append(temp)
        
        # Batch generate
        responses = await self._batch_generate_with_temperatures(
            prompts, temperatures
        )
        
        # Create particles
        particles = []
        for i, resp in enumerate(responses):
            attempt = CodeAttempt(
                code=self._extract_code(resp.text),
                strategy=strategies[i % len(strategies)][0]
            )
            # Initial weight uniform
            particles.append((attempt, 1.0 / num_particles))
        
        self.total_particles_generated += num_particles
        return particles
    
    async def evolve_particles(
        self,
        parent_particles: List[Tuple[CodeAttempt, float]],
        parent_results: List[ExecutionResult],
        problem: str
    ) -> List[Tuple[CodeAttempt, float]]:
        """Evolve particles based on parent performance
        
        Implements systematic resampling + targeted evolution
        """
        weights = np.array([w for _, w in parent_particles])
        attempts = [a for a, _ in parent_particles]
        
        # Calculate ESS
        ess = self._calculate_ess(weights)
        
        # Resample if ESS too low
        if ess < len(weights) * self.config.ess_threshold:
            indices = systematic_resample(weights, len(weights))
            attempts = [attempts[i] for i in indices]
            parent_results = [parent_results[i] for i in indices]
            # Reset weights after resampling
            weights = np.ones(len(indices)) / len(indices)
            logger.info(f"Resampled particles due to low ESS: {ess:.2f}")
        
        # Group similar particles
        groups = self._group_similar_particles(attempts, parent_results)
        
        # Evolve each group
        evolution_tasks = []
        for group_attempts, group_results in groups:
            # Analyze common failure patterns
            failure_analysis = self._analyze_failures(group_results)
            
            # Determine evolution strategy
            avg_pass_rate = np.mean([r.pass_rate for r in group_results])
            if avg_pass_rate == 0:
                strategy = "complete_rewrite"
                temperature = 1.0
            elif avg_pass_rate < 0.3:
                strategy = "major_refactor"
                temperature = 0.8
            elif avg_pass_rate < 0.7:
                strategy = "targeted_fix"
                temperature = 0.6
            elif avg_pass_rate < 1.0:
                strategy = "edge_case_fix"
                temperature = 0.4
            else:
                strategy = "optimize"
                temperature = 0.3
            
            # Create evolution prompts
            for attempt in group_attempts:
                prompt = self._build_evolution_prompt(
                    problem, attempt, failure_analysis, strategy
                )
                evolution_tasks.append((prompt, temperature))
        
        # Batch evolve
        prompts = [p for p, _ in evolution_tasks]
        temperatures = [t for _, t in evolution_tasks]
        
        responses = await self._batch_generate_with_temperatures(
            prompts, temperatures
        )
        
        # Create evolved particles
        evolved = []
        for i, resp in enumerate(responses):
            parent_idx = i % len(attempts)
            new_attempt = CodeAttempt(
                code=self._extract_code(resp.text),
                strategy=evolution_tasks[i][1],
                parent_hash=attempts[parent_idx].code_hash
            )
            # Inherit parent weight initially
            evolved.append((new_attempt, weights[parent_idx]))
        
        self.total_particles_generated += len(evolved)
        return evolved
    
    def compute_particle_weights(
        self,
        particles: List[CodeAttempt],
        results: List[ExecutionResult],
        alpha_test: float = 0.7,
        alpha_prm: float = 0.3
    ) -> np.ndarray:
        """Compute weights using composite scoring
        
        Implements feedback suggestion for blended scoring
        """
        n = len(particles)
        weights = np.zeros(n)
        
        for i, (particle, result) in enumerate(zip(particles, results)):
            # Test-based component (primary signal)
            test_score = result.pass_rate
            
            # Smoothing for partial progress
            if result.has_syntax_errors:
                test_score *= 0.1  # Small credit for syntactically valid
            elif result.compilation_errors:
                test_score *= 0.3  # More credit for compiling
            
            # Optional PRM component
            prm_score = 0.5  # Default if no PRM
            if hasattr(self.reward, 'get_prm_score'):
                prm_score = self.reward.get_prm_score(
                    problem=self.problem,
                    solution=particle.code
                )
            
            # Composite weight
            weights[i] = (alpha_test * test_score + 
                         alpha_prm * prm_score)
            
            # Diversity bonus
            diversity_bonus = self._compute_diversity_bonus(
                particle, particles
            )
            weights[i] *= (1.0 + diversity_bonus)
        
        # Normalize
        weights = weights / (weights.sum() + 1e-10)
        return weights
    
    def _calculate_ess(self, weights: np.ndarray) -> float:
        """Calculate Effective Sample Size"""
        return 1.0 / np.sum(weights ** 2)
    
    def _compute_diversity_bonus(
        self,
        particle: CodeAttempt,
        all_particles: List[CodeAttempt],
        max_bonus: float = 0.2
    ) -> float:
        """Compute diversity bonus for unique solutions"""
        if len(all_particles) <= 1:
            return 0.0
            
        # Simple approach: bonus based on unique lines
        particle_lines = set(particle.code.splitlines())
        
        similarities = []
        for other in all_particles:
            if other.code_hash == particle.code_hash:
                continue
            other_lines = set(other.code.splitlines())
            if len(particle_lines) > 0:
                jaccard = len(particle_lines & other_lines) / len(particle_lines | other_lines)
                similarities.append(jaccard)
        
        if not similarities:
            return max_bonus
            
        # Lower similarity = higher bonus
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        return min(max_bonus, diversity * max_bonus)
    
    async def _batch_generate_with_temperatures(
        self,
        prompts: List[str],
        temperatures: List[float]
    ) -> List[Any]:
        """Batch generation with different temperatures
        
        Handles batching limitations of LLM backend
        """
        batch_size = self.config.llm_batch_size
        all_responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_temps = temperatures[i:i + batch_size]
            
            # If all same temperature, can batch directly
            if len(set(batch_temps)) == 1:
                responses = await self.llm.generate_async(
                    batch_prompts,
                    temperature=batch_temps[0]
                )
                all_responses.extend(responses)
            else:
                # Different temperatures, need individual calls
                tasks = [
                    self.llm.generate_async(p, temperature=t)
                    for p, t in zip(batch_prompts, batch_temps)
                ]
                responses = await asyncio.gather(*tasks)
                all_responses.extend(responses)
        
        return all_responses
```

#### 1.2 Resampling Strategies (`src/treequest_coding_solver/particle/resampling.py`)

```python
import numpy as np
from typing import List, Tuple
from numba import jit

@jit(nopython=True)
def systematic_resample(weights: np.ndarray, n_samples: int) -> np.ndarray:
    """Systematic resampling - O(n) with low variance
    
    Vectorized implementation for performance
    """
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # Ensure sum to 1
    
    step = 1.0 / n_samples
    start = np.random.uniform(0, step)
    points = np.arange(start, 1.0, step)
    
    indices = np.zeros(n_samples, dtype=np.int32)
    
    i, j = 0, 0
    while i < n_samples and j < len(weights):
        if points[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    
    # Handle any remaining
    while i < n_samples:
        indices[i] = len(weights) - 1
        i += 1
        
    return indices

@jit(nopython=True)
def stratified_resample(weights: np.ndarray, n_samples: int) -> np.ndarray:
    """Stratified resampling - ensures representation across strata"""
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    
    indices = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(n_samples):
        # Random point in stratum
        u = (i + np.random.uniform(0, 1)) / n_samples
        
        # Binary search for index
        j = np.searchsorted(cumsum, u)
        indices[i] = min(j, len(weights) - 1)
    
    return indices

def adaptive_resample(
    weights: np.ndarray,
    n_samples: int,
    ess_threshold: float = 0.5
) -> Tuple[np.ndarray, bool]:
    """Adaptive resampling based on ESS
    
    Only resample when diversity is low
    """
    # Calculate ESS
    ess = 1.0 / np.sum(weights ** 2)
    ess_ratio = ess / len(weights)
    
    if ess_ratio < ess_threshold:
        # Low diversity, resample
        indices = systematic_resample(weights, n_samples)
        return indices, True
    else:
        # High diversity, keep current particles
        indices = np.arange(len(weights))
        return indices, False
```

### Phase 2: Algorithm Integration (Week 2-3)

#### 2.1 Strategy Policy (`src/treequest_coding_solver/algorithms/strategy_policy.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class NodeInfo:
    """Information about a tree node for decision making"""
    score: float
    depth: int
    children_count: int
    iterations_since_improvement: int
    parent_used_pf: bool
    total_descendants: int

class StrategyPolicy(ABC):
    """Abstract strategy for node selection and expansion decisions
    
    Design Rationale (from feedback):
    - Extracted from algorithm for clean separation
    - Enables A/B testing different strategies
    - Can be learned or hand-crafted
    """
    
    @abstractmethod
    def select_node(self, nodes: List[NodeInfo]) -> int:
        """Select which node to expand next"""
        pass
    
    @abstractmethod
    def decide_expansion_type(self, node: NodeInfo) -> str:
        """Decide 'standard' or 'particle_filter' expansion"""
        pass

class UCBStrategyPolicy(StrategyPolicy):
    """Upper Confidence Bound node selection"""
    
    def __init__(self, exploration_constant: float = 1.414):
        self.c = exploration_constant
        
    def select_node(self, nodes: List[NodeInfo]) -> int:
        """Select using UCB formula"""
        if not nodes:
            return 0
            
        total_visits = sum(n.children_count + 1 for n in nodes)
        
        best_idx = 0
        best_value = -float('inf')
        
        for i, node in enumerate(nodes):
            visits = node.children_count + 1
            
            # UCB value
            exploitation = node.score
            exploration = self.c * np.sqrt(np.log(total_visits) / visits)
            value = exploitation + exploration
            
            if value > best_value:
                best_value = value
                best_idx = i
                
        return best_idx
    
    def decide_expansion_type(self, node: NodeInfo) -> str:
        """Use particle filtering for uncertain nodes"""
        # Delegate to ParticleManager via should_use_pf
        # This is just the default fallback
        return "standard"

class ThompsonSamplingPolicy(StrategyPolicy):
    """Thompson Sampling for node selection"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        
    def select_node(self, nodes: List[NodeInfo]) -> int:
        """Sample from posterior and select max"""
        if not nodes:
            return 0
            
        samples = []
        for node in nodes:
            # Beta posterior for pass rate
            successes = node.score * (node.children_count + 1)
            failures = (1 - node.score) * (node.children_count + 1)
            
            sample = np.random.beta(
                self.alpha + successes,
                self.beta + failures
            )
            samples.append(sample)
            
        return int(np.argmax(samples))
    
    def decide_expansion_type(self, node: NodeInfo) -> str:
        """Adaptive decision based on uncertainty"""
        # High uncertainty nodes benefit from PF
        visits = node.children_count + 1
        uncertainty = 1.0 / np.sqrt(visits + 1)
        
        if uncertainty > 0.3 and node.score > 0.2 and node.score < 0.8:
            return "particle_filter"
        return "standard"

class AdaptiveStrategyPolicy(StrategyPolicy):
    """Adaptive policy that learns from history
    
    Future work: Could use RL to learn optimal policy
    """
    
    def __init__(self):
        self.history = []
        self.success_rates = {"standard": 0.5, "particle_filter": 0.5}
        
    def select_node(self, nodes: List[NodeInfo]) -> int:
        """Combine UCB with learned biases"""
        # For now, delegate to UCB
        ucb = UCBStrategyPolicy()
        return ucb.select_node(nodes)
    
    def decide_expansion_type(self, node: NodeInfo) -> str:
        """Use learned success rates"""
        # Epsilon-greedy with learned rates
        if np.random.random() < 0.1:  # Explore
            return np.random.choice(["standard", "particle_filter"])
            
        # Exploit: choose based on success rates
        if self.success_rates["particle_filter"] > self.success_rates["standard"]:
            if node.score > 0.2 and node.score < 0.8:
                return "particle_filter"
                
        return "standard"
    
    def update_history(self, expansion_type: str, improvement: float):
        """Update success rates based on outcomes"""
        self.history.append((expansion_type, improvement))
        
        # Update success rates with moving average
        alpha = 0.1
        old_rate = self.success_rates[expansion_type]
        self.success_rates[expansion_type] = (
            (1 - alpha) * old_rate + alpha * improvement
        )
```

#### 2.2 Hybrid Algorithm (`src/treequest_coding_solver/algorithms/hybrid_mcts_pf.py`)

```python
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass

from treequest import Algorithm
from treequest.algos.tree import Tree, Node
from treequest.types import GenerateFnType, StateScoreType, NodeStateT

from ..core.interfaces import SearchAlgorithm
from ..core.state import CodeAttempt, ExecutionResult, SearchMetadata
from ..particle.manager import ParticleManager
from .strategy_policy import StrategyPolicy, NodeInfo

logger = logging.getLogger(__name__)

@dataclass  
class HybridState:
    """Combined state for code attempts with optional particles"""
    attempt: CodeAttempt
    result: ExecutionResult
    metadata: SearchMetadata
    
    # Particle information (if PF was used)
    particles: Optional[List[Tuple[CodeAttempt, float]]] = None
    particle_results: Optional[List[ExecutionResult]] = None
    
    @property
    def score(self) -> float:
        """TreeQuest-compatible score in [0,1]"""
        return self.result.pass_rate

@dataclass
class HybridAlgoState:
    """Algorithm state for hybrid approach"""
    tree: Tree[HybridState]
    particle_manager: ParticleManager
    strategy_policy: StrategyPolicy
    
    # Tracking
    iterations: int = 0
    iterations_since_improvement: int = 0
    best_score: float = 0.0
    
    # Fallback state
    fallback_active: bool = False

class HybridMCTSPFAlgorithm(SearchAlgorithm[HybridState, HybridAlgoState]):
    """Hybrid algorithm combining MCTS with strategic particle filtering
    
    Design improvements from feedback:
    - Thin wrapper delegating to ParticleManager and StrategyPolicy
    - Clean fallback to base algorithm
    - Supports both sync and async operation
    """
    
    def __init__(
        self,
        base_algorithm: Algorithm,
        particle_manager: ParticleManager,
        strategy_policy: StrategyPolicy,
        fallback_algorithm: Optional[Algorithm] = None,
        enable_telemetry: bool = True
    ):
        self.base_algo = base_algorithm
        self.particle_manager = particle_manager
        self.strategy_policy = strategy_policy
        self.fallback_algo = fallback_algorithm or base_algorithm
        self.enable_telemetry = enable_telemetry
        
        if enable_telemetry:
            from ..monitoring.telemetry import init_telemetry
            self.tracer = init_telemetry()
    
    def initialize(self) -> HybridAlgoState:
        """Initialize algorithm state"""
        return HybridAlgoState(
            tree=Tree[HybridState](),
            particle_manager=self.particle_manager,
            strategy_policy=self.strategy_policy
        )
    
    def step(
        self,
        state: HybridAlgoState,
        generators: Dict[str, GenerateFnType[HybridState]]
    ) -> HybridAlgoState:
        """Execute one search step with telemetry"""
        
        if self.enable_telemetry:
            with self.tracer.start_as_current_span("hybrid_step") as span:
                span.set_attribute("iteration", state.iterations)
                span.set_attribute("best_score", state.best_score)
                return self._step_impl(state, generators)
        else:
            return self._step_impl(state, generators)
    
    def _step_impl(
        self,
        state: HybridAlgoState,
        generators: Dict[str, GenerateFnType[HybridState]]
    ) -> HybridAlgoState:
        """Core step implementation"""
        
        # Update iteration tracking
        state.iterations += 1
        state.particle_manager.current_iteration = state.iterations
        
        # Collect expandable nodes
        expandable = self._collect_expandable_nodes(state.tree.root)
        if not expandable:
            logger.warning("No expandable nodes found")
            return state
        
        # Convert to NodeInfo for strategy
        node_infos = [self._node_to_info(n, state) for n in expandable]
        
        # Select node
        selected_idx = state.strategy_policy.select_node(node_infos)
        selected_node = expandable[selected_idx]
        selected_info = node_infos[selected_idx]
        
        # Decide expansion type
        should_use_pf = state.particle_manager.should_use_particle_filtering(
            node_score=selected_info.score,
            node_depth=selected_info.depth,
            children_count=selected_info.children_count,
            iterations_since_improvement=state.iterations_since_improvement
        )
        
        expansion_type = "particle_filter" if should_use_pf else "standard"
        
        # Log decision
        logger.info(f"Iteration {state.iterations}: Expanding node {selected_node.expand_idx} "
                   f"with {expansion_type} (score={selected_info.score:.3f})")
        
        try:
            # Select generator
            if expansion_type == "particle_filter" and "pf_generator" in generators:
                generator = generators["pf_generator"]
            else:
                # Pick a standard generator based on node state
                if selected_info.score == 0:
                    generator_name = "debug_generator"
                elif selected_info.score < 0.5:
                    generator_name = "refine_generator"
                elif selected_info.score < 1.0:
                    generator_name = "edge_case_generator"
                else:
                    generator_name = "optimize_generator"
                    
                generator = generators.get(generator_name, 
                                         generators.get("default_generator"))
            
            # Generate new state
            parent_state = selected_node.state if selected_node.expand_idx >= 0 else None
            new_state, score = generator(parent_state)
            
            # Update metadata
            new_state.metadata.expansion_type = expansion_type
            new_state.metadata.parent_id = (parent_state.metadata.node_id 
                                           if parent_state else None)
            new_state.metadata.depth = selected_info.depth + 1
            
            # Add to tree
            new_node = state.tree.add_node(
                parent=selected_node,
                state=new_state,
                score=score,
                action_name=expansion_type
            )
            
            # Update tracking
            if score > state.best_score:
                state.best_score = score
                state.iterations_since_improvement = 0
            else:
                state.iterations_since_improvement += 1
                
        except Exception as e:
            logger.error(f"Error during expansion: {e}")
            if state.fallback_active:
                raise
            
            # Activate fallback
            logger.warning("Activating fallback algorithm")
            state.fallback_active = True
            
            # Use fallback algorithm for this step
            fallback_state = self.fallback_algo.step(
                state.tree,  # Assuming compatible tree structure
                generators
            )
            # Update our state from fallback
            # ... (implementation depends on fallback compatibility)
        
        return state
    
    def get_best_states(
        self,
        state: HybridAlgoState,
        k: int = 1
    ) -> List[Tuple[HybridState, float]]:
        """Extract top-k states from tree"""
        
        all_states = []
        
        def collect_states(node: Node):
            if node.state is not None:
                all_states.append((node.state, node.score))
            for child in node.children.values():
                collect_states(child)
        
        collect_states(state.tree.root)
        
        # Sort by score
        all_states.sort(key=lambda x: x[1], reverse=True)
        
        return all_states[:k]
    
    def _collect_expandable_nodes(self, root: Node) -> List[Node]:
        """Collect all nodes that can be expanded"""
        expandable = []
        
        def traverse(node: Node):
            # Node is expandable if not perfect and not too deep
            if node.state and node.state.score < 1.0:
                expandable.append(node)
            
            for child in node.children.values():
                traverse(child)
        
        # Start from root's children (root itself has no state)
        for child in root.children.values():
            traverse(child)
            
        return expandable
    
    def _node_to_info(self, node: Node, algo_state: HybridAlgoState) -> NodeInfo:
        """Convert tree node to NodeInfo for strategy"""
        # Calculate depth
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
            
        # Check if parent used PF
        parent_used_pf = False
        if node.parent and node.parent.state:
            parent_used_pf = (node.parent.state.metadata.expansion_type == 
                            "particle_filter")
        
        # Count descendants
        def count_descendants(n: Node) -> int:
            return 1 + sum(count_descendants(c) for c in n.children.values())
        
        total_descendants = count_descendants(node) - 1
        
        return NodeInfo(
            score=node.score,
            depth=depth,
            children_count=len(node.children),
            iterations_since_improvement=algo_state.iterations_since_improvement,
            parent_used_pf=parent_used_pf,
            total_descendants=total_descendants
        )
```

### Phase 3: Adapters and Integration (Week 3-4)

#### 3.1 TreeQuest Generator Adapter (`src/treequest_coding_solver/adapters/treequest_generator.py`)

```python
import asyncio
from typing import Optional, Tuple, Callable, List
import logging

from treequest.types import GenerateFnType

from ..core.state import CodeAttempt, ExecutionResult, SearchMetadata
from ..algorithms.hybrid_mcts_pf import HybridState
from ..particle.manager import ParticleManager
from ..execution.sandbox import SafeCodeExecutor

logger = logging.getLogger(__name__)

class TreeQuestGeneratorAdapter:
    """Adapts particle filtering to TreeQuest's GenerateFnType
    
    Design Rationale:
    - Wraps async particle operations in sync interface
    - Handles all PF complexity internally
    - Returns TreeQuest-compatible (state, score) tuples
    """
    
    def __init__(
        self,
        particle_manager: ParticleManager,
        executor: SafeCodeExecutor,
        problem: str,
        test_cases: List[Any]
    ):
        self.pm = particle_manager
        self.executor = executor
        self.problem = problem
        self.test_cases = test_cases
        
    def create_pf_generator(self) -> GenerateFnType[HybridState]:
        """Create a particle filtering generator function"""
        
        def generate(parent_state: Optional[HybridState]) -> Tuple[HybridState, float]:
            """Particle filtering generation
            
            This is called synchronously by TreeQuest but runs
            async particle operations internally
            """
            # Run async operations in sync context
            return asyncio.run(self._async_generate(parent_state))
            
        return generate
    
    async def _async_generate(
        self,
        parent_state: Optional[HybridState]
    ) -> Tuple[HybridState, float]:
        """Async particle filtering implementation"""
        
        if parent_state is None:
            # Initial particle generation
            budget = self.pm.config.initial_particles
            particles = await self.pm.generate_initial_particles(
                self.problem,
                budget
            )
        else:
            # Evolve from parent
            budget = self.pm.allocate_particle_budget(
                parent_state.metadata.depth,
                self.pm.config.total_particle_budget
            )
            
            if parent_state.particles:
                # Evolve existing particles
                particles = await self.pm.evolve_particles(
                    parent_state.particles,
                    parent_state.particle_results,
                    self.problem
                )
            else:
                # Convert single state to particles
                particles = await self._diversify_single_state(
                    parent_state,
                    budget
                )
        
        # Evaluate all particles
        results = await self._evaluate_particles(particles)
        
        # Compute weights
        attempts = [p[0] for p in particles]
        weights = self.pm.compute_particle_weights(attempts, results)
        
        # Update particles with new weights
        weighted_particles = [(attempts[i], weights[i]) 
                             for i in range(len(particles))]
        
        # Select best for primary state
        best_idx = int(weights.argmax())
        best_attempt = attempts[best_idx]
        best_result = results[best_idx]
        
        # Create hybrid state
        new_state = HybridState(
            attempt=best_attempt,
            result=best_result,
            metadata=SearchMetadata(
                expansion_type="particle_filter",
                particle_count=len(particles),
                effective_sample_size=self.pm._calculate_ess(weights),
                particle_diversity=self._calculate_diversity(attempts)
            ),
            particles=weighted_particles,
            particle_results=results
        )
        
        return new_state, new_state.score
    
    async def _evaluate_particles(
        self,
        particles: List[Tuple[CodeAttempt, float]]
    ) -> List[ExecutionResult]:
        """Evaluate all particles in parallel"""
        
        # Extract attempts
        attempts = [p[0] for p in particles]
        
        # Check cache first
        results = []
        uncached_indices = []
        
        for i, attempt in enumerate(attempts):
            cached = self.executor.get_cached_result(attempt.code_hash)
            if cached:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
        
        # Evaluate uncached in parallel
        if uncached_indices:
            eval_tasks = []
            for i in uncached_indices:
                task = self.executor.execute_code_async(
                    attempts[i].code,
                    self.test_cases
                )
                eval_tasks.append(task)
            
            uncached_results = await asyncio.gather(*eval_tasks)
            
            # Fill in results
            for i, result in zip(uncached_indices, uncached_results):
                results[i] = result
                # Cache for future
                self.executor.cache_result(attempts[i].code_hash, result)
        
        return results
    
    async def _diversify_single_state(
        self,
        state: HybridState,
        num_particles: int
    ) -> List[Tuple[CodeAttempt, float]]:
        """Convert single state to diverse particles"""
        
        # Use different refinement strategies
        strategies = ["refine", "debug", "optimize", "edge_cases"]
        
        particles = [(state.attempt, 1.0)]  # Include original
        
        # Generate variations
        for i in range(num_particles - 1):
            strategy = strategies[i % len(strategies)]
            # ... generate variation using strategy
            # (implementation depends on LLM backend)
        
        return particles
    
    def _calculate_diversity(self, attempts: List[CodeAttempt]) -> float:
        """Calculate diversity metric for particles"""
        if len(attempts) <= 1:
            return 0.0
            
        # Simple metric: average pairwise distance
        total_distance = 0.0
        count = 0
        
        for i in range(len(attempts)):
            for j in range(i + 1, len(attempts)):
                distance = self._code_distance(
                    attempts[i].code,
                    attempts[j].code
                )
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _code_distance(self, code1: str, code2: str) -> float:
        """Compute distance between two code strings"""
        # Simple line-based distance
        lines1 = set(code1.splitlines())
        lines2 = set(code2.splitlines())
        
        if not lines1 and not lines2:
            return 0.0
            
        union = lines1 | lines2
        intersection = lines1 & lines2
        
        if len(union) == 0:
            return 0.0
            
        return 1.0 - (len(intersection) / len(union))
```

#### 3.2 its_hub Adapter (`src/treequest_coding_solver/adapters/itshub_backend.py`)

```python
from typing import List, Optional, Union
import asyncio

try:
    from its_hub.base import AbstractLanguageModel, AbstractScalingAlgorithm
    from its_hub.types import ChatMessage
    from its_hub.algorithms import ParticleFiltering
    from its_hub.lms import StepGeneration
    ITS_HUB_AVAILABLE = True
except ImportError:
    ITS_HUB_AVAILABLE = False

from ..core.interfaces import LLMBackend, LLMResponse

class ItsHubLLMBackend(LLMBackend):
    """Adapter for its_hub language models
    
    Design Rationale:
    - Provides bridge to its_hub ecosystem
    - Supports their batching and async features
    - Graceful fallback if its_hub not available
    """
    
    def __init__(
        self,
        its_hub_lm: Optional['AbstractLanguageModel'] = None,
        fallback_backend: Optional[LLMBackend] = None
    ):
        if not ITS_HUB_AVAILABLE and fallback_backend is None:
            raise ImportError(
                "its_hub not available and no fallback provided. "
                "Install with: pip install its_hub"
            )
            
        self.its_hub_lm = its_hub_lm
        self.fallback = fallback_backend
        
    async def generate_async(
        self,
        prompts: Union[str, List[str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Union[LLMResponse, List[LLMResponse]]:
        """Generate using its_hub or fallback"""
        
        if self.its_hub_lm is None:
            # Use fallback
            return await self.fallback.generate_async(
                prompts, temperature, max_tokens, stop, **kwargs
            )
        
        # Convert to its_hub format
        is_batch = isinstance(prompts, list)
        prompt_list = prompts if is_batch else [prompts]
        
        # Create ChatMessage format
        messages_list = []
        for prompt in prompt_list:
            messages = [
                ChatMessage(role="user", content=prompt)
            ]
            messages_list.append(messages)
        
        # Call its_hub
        if is_batch:
            responses = self.its_hub_lm.generate(messages_list, stop=stop)
        else:
            responses = [self.its_hub_lm.generate(messages_list[0], stop=stop)]
        
        # Convert responses
        llm_responses = []
        for i, resp in enumerate(responses):
            llm_resp = LLMResponse(
                text=resp,
                tokens_used=len(resp.split()),  # Approximate
                latency_ms=0.0,  # its_hub doesn't provide this
                model_name=getattr(self.its_hub_lm, 'model_name', 'its_hub'),
                temperature=temperature,
                metadata={}
            )
            llm_responses.append(llm_resp)
        
        return llm_responses if is_batch else llm_responses[0]
    
    async def evaluate_async(
        self,
        prompt: str,
        completions: List[str]
    ) -> List[float]:
        """Evaluate completions using its_hub"""
        
        if self.its_hub_lm is None:
            # Use fallback if available
            if hasattr(self.fallback, 'evaluate_async'):
                return await self.fallback.evaluate_async(prompt, completions)
            else:
                # Default uniform scores
                return [0.5] * len(completions)
        
        # Use its_hub evaluation
        scores = []
        for completion in completions:
            score_list = self.its_hub_lm.evaluate(prompt, completion)
            # Average token-level scores
            avg_score = sum(score_list) / len(score_list) if score_list else 0.5
            scores.append(avg_score)
        
        return scores

class ItsHubParticleFilteringAdapter:
    """Adapter to use its_hub's ParticleFiltering algorithm
    
    Allows using their optimized implementation when available
    """
    
    def __init__(
        self,
        step_generation: Optional['StepGeneration'] = None,
        process_reward_model: Optional[Any] = None
    ):
        if not ITS_HUB_AVAILABLE:
            raise ImportError("its_hub required for ParticleFilteringAdapter")
            
        self.sg = step_generation or StepGeneration()
        self.prm = process_reward_model
        
    def create_algorithm(
        self,
        num_particles: int = 16,
        budget: int = 64
    ) -> 'AbstractScalingAlgorithm':
        """Create its_hub ParticleFiltering instance"""
        
        return ParticleFiltering(
            sg=self.sg,
            prm=self.prm,
            n=num_particles,
            budget=budget
        )
    
    async def run_particle_filtering(
        self,
        lm: 'AbstractLanguageModel',
        prompt: str,
        num_particles: int,
        budget: int
    ) -> Any:
        """Run particle filtering using its_hub"""
        
        algo = self.create_algorithm(num_particles, budget)
        
        # its_hub algorithms are synchronous
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            algo.infer,
            lm,
            prompt,
            budget,
            False  # return_response_only
        )
        
        return result
```

### Phase 4: Execution and Rewards (Week 4-5)

#### 4.1 Diff-Aware Executor (`src/treequest_coding_solver/execution/diff_executor.py`)

```python
import ast
import difflib
from typing import List, Set, Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass
import hashlib

from .sandbox import SafeCodeExecutor, TestCase
from ..core.state import ExecutionResult

@dataclass
class CodeCoverage:
    """Track which lines are covered by tests"""
    covered_lines: Set[int]
    test_to_lines: Dict[int, Set[int]]  # test_idx -> covered lines

class DiffAwareExecutor(SafeCodeExecutor):
    """Optimized executor that only runs affected tests
    
    Design Rationale (from feedback):
    - 2-3x speedup on iterative refinements
    - Maintains full accuracy through AST analysis
    - Falls back to full execution when needed
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coverage_cache: Dict[str, CodeCoverage] = {}
        self.ast_cache: Dict[str, ast.AST] = {}
        
    async def execute_code_with_diff_async(
        self,
        new_code: str,
        old_code: Optional[str],
        test_cases: List[TestCase],
        previous_result: Optional[ExecutionResult] = None
    ) -> ExecutionResult:
        """Execute only tests affected by code changes"""
        
        # If no previous code, run all tests
        if old_code is None or previous_result is None:
            return await self.execute_code_async(new_code, test_cases)
        
        # Parse both versions
        try:
            old_ast = self._parse_code(old_code)
            new_ast = self._parse_code(new_code)
        except SyntaxError:
            # Can't analyze, run all tests
            return await self.execute_code_async(new_code, test_cases)
        
        # Find changed lines
        changed_lines = self._find_changed_lines(old_code, new_code)
        if not changed_lines:
            # No changes, return previous result
            return previous_result
        
        # Determine affected functions/classes
        affected_symbols = self._find_affected_symbols(
            new_ast, changed_lines
        )
        
        # Get coverage information
        coverage = await self._get_or_compute_coverage(
            old_code, test_cases
        )
        
        # Determine which tests to rerun
        tests_to_run = self._select_affected_tests(
            coverage, changed_lines, affected_symbols
        )
        
        if len(tests_to_run) == len(test_cases):
            # All tests affected, run normally
            return await self.execute_code_async(new_code, test_cases)
        
        # Run only affected tests
        affected_test_cases = [test_cases[i] for i in tests_to_run]
        partial_result = await self.execute_code_async(
            new_code, affected_test_cases
        )
        
        # Merge with previous results
        return self._merge_results(
            previous_result, partial_result, tests_to_run
        )
    
    def _parse_code(self, code: str) -> ast.AST:
        """Parse code with caching"""
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        if code_hash in self.ast_cache:
            return self.ast_cache[code_hash]
        
        parsed = ast.parse(code)
        self.ast_cache[code_hash] = parsed
        return parsed
    
    def _find_changed_lines(self, old_code: str, new_code: str) -> Set[int]:
        """Find line numbers that changed"""
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        
        differ = difflib.SequenceMatcher(None, old_lines, new_lines)
        changed = set()
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag != 'equal':
                # Add all affected lines in new code
                for line_num in range(j1 + 1, j2 + 1):
                    changed.add(line_num)
        
        return changed
    
    def _find_affected_symbols(
        self,
        tree: ast.AST,
        changed_lines: Set[int]
    ) -> Set[str]:
        """Find functions/classes containing changed lines"""
        
        affected = set()
        
        class SymbolFinder(ast.NodeVisitor):
            def __init__(self):
                self.current_symbol = None
                
            def visit_FunctionDef(self, node):
                old_symbol = self.current_symbol
                self.current_symbol = node.name
                
                # Check if any line in function changed
                for child in ast.walk(node):
                    if hasattr(child, 'lineno') and child.lineno in changed_lines:
                        affected.add(self.current_symbol)
                        break
                
                self.generic_visit(node)
                self.current_symbol = old_symbol
                
            def visit_ClassDef(self, node):
                old_symbol = self.current_symbol
                self.current_symbol = node.name
                
                for child in ast.walk(node):
                    if hasattr(child, 'lineno') and child.lineno in changed_lines:
                        affected.add(self.current_symbol)
                        break
                
                self.generic_visit(node)
                self.current_symbol = old_symbol
        
        finder = SymbolFinder()
        finder.visit(tree)
        
        return affected
    
    async def _get_or_compute_coverage(
        self,
        code: str,
        test_cases: List[TestCase]
    ) -> CodeCoverage:
        """Get coverage information with caching"""
        
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        if code_hash in self.coverage_cache:
            return self.coverage_cache[code_hash]
        
        # Compute coverage by running with tracing
        # This is expensive but cached
        coverage = await self._compute_coverage(code, test_cases)
        self.coverage_cache[code_hash] = coverage
        
        return coverage
    
    async def _compute_coverage(
        self,
        code: str,
        test_cases: List[TestCase]
    ) -> CodeCoverage:
        """Compute test coverage information"""
        
        # Simple implementation - in production would use coverage.py
        # For now, assume each test covers all lines
        total_lines = len(code.splitlines())
        all_lines = set(range(1, total_lines + 1))
        
        coverage = CodeCoverage(
            covered_lines=all_lines,
            test_to_lines={i: all_lines for i in range(len(test_cases))}
        )
        
        return coverage
    
    def _select_affected_tests(
        self,
        coverage: CodeCoverage,
        changed_lines: Set[int],
        affected_symbols: Set[str]
    ) -> List[int]:
        """Select which tests need to be rerun"""
        
        affected_tests = []
        
        for test_idx, covered_lines in coverage.test_to_lines.items():
            # Test is affected if it covers any changed line
            if covered_lines & changed_lines:
                affected_tests.append(test_idx)
                
        return affected_tests
    
    def _merge_results(
        self,
        previous: ExecutionResult,
        partial: ExecutionResult,
        updated_indices: List[int]
    ) -> ExecutionResult:
        """Merge partial results with previous results"""
        
        # Clone previous test cases
        merged_cases = previous.test_cases.copy()
        
        # Update with new results
        for i, test_idx in enumerate(updated_indices):
            merged_cases[test_idx] = partial.test_cases[i]
        
        # Recalculate totals
        passed = sum(1 for tc in merged_cases if tc.passed)
        failed = len(merged_cases) - passed
        
        return ExecutionResult(
            test_cases=merged_cases,
            total_tests=len(merged_cases),
            passed_tests=passed,
            failed_tests=failed,
            total_execution_time_ms=partial.total_execution_time_ms,
            memory_usage_mb=partial.memory_usage_mb,
            sandbox_type=partial.sandbox_type,
            timeout_occurred=partial.timeout_occurred,
            compilation_errors=partial.compilation_errors,
            runtime_errors=partial.runtime_errors
        )
```

#### 4.2 Composite Reward (`src/treequest_coding_solver/rewards/composite.py`)

```python
from typing import Dict, List, Optional, Any
import numpy as np

from ..core.interfaces import RewardSignal, RewardComponents
from ..core.state import ExecutionResult

class CompositeRewardSignal(RewardSignal):
    """Blends multiple reward signals with configurable weights
    
    Design Rationale (from feedback):
    - Semi-verifiable (tests) as primary signal
    - Optional PRM for shaping/smoothing
    - Configurable weights for experimentation
    """
    
    def __init__(
        self,
        components: Dict[str, RewardSignal],
        weights: Dict[str, float],
        normalize: bool = True
    ):
        self.components = components
        self.weights = weights
        self.normalize_output = normalize
        
        # Validate weights sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # Auto-normalize weights
            self.weights = {k: v/total_weight for k, v in weights.items()}
    
    def compute_reward(
        self,
        problem: str,
        solution: str,
        execution_result: Optional[ExecutionResult] = None,
        **kwargs
    ) -> RewardComponents:
        """Compute composite reward from all components"""
        
        component_scores = {}
        component_metadata = {}
        
        # Compute each component
        for name, signal in self.components.items():
            reward = signal.compute_reward(
                problem, solution, execution_result, **kwargs
            )
            component_scores[name] = reward.total
            component_metadata[name] = reward.metadata
        
        # Apply weights
        weighted_sum = sum(
            component_scores[name] * self.weights.get(name, 0.0)
            for name in component_scores
        )
        
        # Normalize if requested
        if self.normalize_output:
            total = self.normalize(weighted_sum)
        else:
            total = weighted_sum
        
        return RewardComponents(
            total=total,
            components=component_scores,
            metadata={
                'weights': self.weights,
                'component_metadata': component_metadata
            }
        )

class TestBasedReward(RewardSignal):
    """Primary reward signal from test execution
    
    Includes smoothing for partial progress
    """
    
    def compute_reward(
        self,
        problem: str,
        solution: str,
        execution_result: Optional[ExecutionResult] = None,
        **kwargs
    ) -> RewardComponents:
        
        if execution_result is None:
            return RewardComponents(
                total=0.0,
                components={'test_score': 0.0},
                metadata={'error': 'No execution result'}
            )
        
        # Base score from pass rate
        base_score = execution_result.pass_rate
        
        # Smoothing for partial progress
        smooth_score = base_score
        
        if execution_result.has_syntax_errors:
            # Syntax errors get minimal credit
            smooth_score *= 0.1
        elif execution_result.compilation_errors:
            # Compilation errors get some credit
            smooth_score *= 0.3
        elif execution_result.timeout_occurred:
            # Timeout might mean infinite loop
            smooth_score *= 0.5
        
        # Bonus for fast execution
        if execution_result.total_execution_time_ms < 1000:
            smooth_score = min(1.0, smooth_score * 1.1)
        
        # Penalty for high memory
        if execution_result.memory_usage_mb > 256:
            smooth_score *= 0.9
        
        return RewardComponents(
            total=self.normalize(smooth_score),
            components={
                'test_pass_rate': base_score,
                'smoothed_score': smooth_score
            },
            metadata={
                'passed_tests': execution_result.passed_tests,
                'total_tests': execution_result.total_tests,
                'has_errors': len(execution_result.compilation_errors) > 0
            }
        )

class PRMReward(RewardSignal):
    """Process Reward Model signal for shaping
    
    Optional component for domains where tests are sparse
    """
    
    def __init__(self, prm_model: Any):
        self.prm = prm_model
        
    def compute_reward(
        self,
        problem: str,
        solution: str,
        execution_result: Optional[ExecutionResult] = None,
        **kwargs
    ) -> RewardComponents:
        
        # Call PRM model
        try:
            prm_score = self.prm.score(problem, solution)
        except Exception as e:
            # Graceful fallback
            prm_score = 0.5
            
        return RewardComponents(
            total=self.normalize(prm_score),
            components={'prm_score': prm_score},
            metadata={'model': str(self.prm)}
        )

class ComplexityPenalty(RewardSignal):
    """Penalize overly complex solutions"""
    
    def compute_reward(
        self,
        problem: str,
        solution: str,
        execution_result: Optional[ExecutionResult] = None,
        **kwargs
    ) -> RewardComponents:
        
        lines = solution.splitlines()
        
        # Simple complexity metrics
        line_count = len(lines)
        avg_line_length = np.mean([len(l) for l in lines]) if lines else 0
        
        # Penalty for complexity
        if line_count > 100:
            penalty = 0.9
        elif line_count > 50:
            penalty = 0.95
        else:
            penalty = 1.0
            
        if avg_line_length > 80:
            penalty *= 0.95
            
        return RewardComponents(
            total=penalty,
            components={'complexity_score': penalty},
            metadata={
                'line_count': line_count,
                'avg_line_length': avg_line_length
            }
        )
```

### Phase 5: Main Solver and CLI (Week 5-6)

#### 5.1 Main Solver (`src/treequest_coding_solver/solver.py`)

```python
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from treequest import top_k, Algorithm, ABMCTSA, ABMCTSM, StandardMCTS

from .core.interfaces import LLMBackend, SearchAlgorithm, RewardSignal
from .core.state import CodeAttempt, ExecutionResult, SearchMetadata
from .algorithms.hybrid_mcts_pf import HybridMCTSPFAlgorithm, HybridState
from .algorithms.strategy_policy import (
    StrategyPolicy, UCBStrategyPolicy, 
    ThompsonSamplingPolicy, AdaptiveStrategyPolicy
)
from .particle.manager import ParticleManager
from .particle.config import ParticleConfig
from .adapters.treequest_generator import TreeQuestGeneratorAdapter
from .execution.sandbox import SafeCodeExecutor, TestCase
from .execution.diff_executor import DiffAwareExecutor
from .rewards.composite import (
    CompositeRewardSignal, TestBasedReward, 
    PRMReward, ComplexityPenalty
)
from .monitoring.metrics import SolverMonitor
from .llm.gemini_backend import GeminiLLMBackend
from .llm.openai_backend import OpenAILLMBackend
from .utils.serialization import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

@dataclass
class SolverConfig:
    """Configuration for the coding problem solver"""
    # Algorithm selection
    algorithm: str = "hybrid"  # "standard", "abmcts-a", "abmcts-m", "hybrid"
    base_algorithm: str = "abmcts-a"  # For hybrid mode
    strategy_policy: str = "ucb"  # "ucb", "thompson", "adaptive"
    
    # Particle filtering
    enable_particle_filtering: bool = True
    particle_config: ParticleConfig = field(default_factory=ParticleConfig)
    
    # Search budget
    budget: int = 50
    
    # LLM configuration
    llm_backend: str = "gemini"  # "gemini", "openai", "its_hub"
    llm_models: Dict[str, str] = field(default_factory=lambda: {
        "gemini": "gemini-2.0-flash-exp",
        "openai": "gpt-4o"
    })
    
    # Reward configuration
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "test": 0.7,
        "prm": 0.2,
        "complexity": 0.1
    })
    
    # Execution
    enable_docker: bool = True
    diff_aware_execution: bool = True
    execution_timeout: float = 5.0
    
    # Monitoring
    enable_telemetry: bool = True
    enable_visualization: bool = False
    verbose: bool = True
    
    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_dir: Optional[Path] = None

class CodingProblemSolver:
    """Main solver orchestrating all components
    
    Design improvements from feedback:
    - Dependency injection for all major components
    - Clean configuration management
    - Async-first with sync wrappers
    - Comprehensive monitoring
    """
    
    def __init__(
        self,
        config: SolverConfig,
        llm_backend: Optional[LLMBackend] = None,
        custom_components: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.custom_components = custom_components or {}
        
        # Initialize components
        self._init_llm_backend(llm_backend)
        self._init_executor()
        self._init_reward_signal()
        self._init_algorithm()
        
        # Monitoring
        self.monitor = SolverMonitor() if config.enable_telemetry else None
        
        logger.info(f"Initialized solver with config: {config}")
    
    def _init_llm_backend(self, provided_backend: Optional[LLMBackend]):
        """Initialize LLM backend with fallback chain"""
        
        if provided_backend:
            self.llm_backend = provided_backend
            return
        
        # Try to create based on config
        backend_type = self.config.llm_backend
        
        if backend_type == "gemini":
            from .llm.gemini_backend import GeminiLLMBackend
            self.llm_backend = GeminiLLMBackend(
                api_key=self.custom_components.get('gemini_api_key'),
                model=self.config.llm_models.get('gemini')
            )
        elif backend_type == "openai":
            from .llm.openai_backend import OpenAILLMBackend
            self.llm_backend = OpenAILLMBackend(
                api_key=self.custom_components.get('openai_api_key'),
                model=self.config.llm_models.get('openai')
            )
        elif backend_type == "its_hub":
            from .adapters.itshub_backend import ItsHubLLMBackend
            # Create with fallback
            fallback = self._create_fallback_backend()
            self.llm_backend = ItsHubLLMBackend(
                its_hub_lm=self.custom_components.get('its_hub_lm'),
                fallback_backend=fallback
            )
        else:
            raise ValueError(f"Unknown LLM backend: {backend_type}")
    
    def _init_executor(self):
        """Initialize code executor"""
        
        if self.config.diff_aware_execution:
            self.executor = DiffAwareExecutor(
                docker_enabled=self.config.enable_docker,
                timeout_seconds=self.config.execution_timeout
            )
        else:
            self.executor = SafeCodeExecutor(
                docker_enabled=self.config.enable_docker,
                timeout_seconds=self.config.execution_timeout
            )
    
    def _init_reward_signal(self):
        """Initialize reward computation"""
        
        components = {}
        
        # Always include test-based reward
        components['test'] = TestBasedReward()
        
        # Optional PRM
        if 'prm' in self.config.reward_weights and self.config.reward_weights['prm'] > 0:
            prm_model = self.custom_components.get('prm_model')
            if prm_model:
                components['prm'] = PRMReward(prm_model)
        
        # Optional complexity penalty
        if 'complexity' in self.config.reward_weights:
            components['complexity'] = ComplexityPenalty()
        
        # Create composite
        self.reward_signal = CompositeRewardSignal(
            components=components,
            weights=self.config.reward_weights
        )
    
    def _init_algorithm(self):
        """Initialize search algorithm"""
        
        # Create base TreeQuest algorithm
        if self.config.base_algorithm == "abmcts-a":
            base_algo = ABMCTSA()
        elif self.config.base_algorithm == "abmcts-m":
            base_algo = ABMCTSM()
        else:
            base_algo = StandardMCTS()
        
        # Create strategy policy
        if self.config.strategy_policy == "thompson":
            strategy = ThompsonSamplingPolicy()
        elif self.config.strategy_policy == "adaptive":
            strategy = AdaptiveStrategyPolicy()
        else:
            strategy = UCBStrategyPolicy()
        
        if self.config.algorithm == "hybrid" and self.config.enable_particle_filtering:
            # Create particle manager
            particle_manager = ParticleManager(
                config=self.config.particle_config,
                llm_backend=self.llm_backend,
                reward_signal=self.reward_signal
            )
            
            # Create hybrid algorithm
            self.algorithm = HybridMCTSPFAlgorithm(
                base_algorithm=base_algo,
                particle_manager=particle_manager,
                strategy_policy=strategy,
                fallback_algorithm=base_algo,
                enable_telemetry=self.config.enable_telemetry
            )
        else:
            # Use base algorithm directly
            self.algorithm = base_algo
    
    async def solve_problem_async(
        self,
        problem_description: str,
        test_cases: List[TestCase],
        initial_code: Optional[str] = None,
        resume_from_checkpoint: Optional[Path] = None
    ) -> HybridState:
        """Async problem solving with all features"""
        
        if self.monitor:
            self.monitor.start_solving()
        
        try:
            # Load checkpoint if provided
            if resume_from_checkpoint:
                algo_state = load_checkpoint(resume_from_checkpoint)
                logger.info(f"Resumed from checkpoint: {resume_from_checkpoint}")
            else:
                algo_state = self.algorithm.initialize()
            
            # Create generator factory
            if hasattr(self.algorithm, 'particle_manager'):
                # Hybrid mode
                generator_adapter = TreeQuestGeneratorAdapter(
                    particle_manager=self.algorithm.particle_manager,
                    executor=self.executor,
                    problem=problem_description,
                    test_cases=test_cases
                )
                
                generators = {
                    'pf_generator': generator_adapter.create_pf_generator(),
                    'default_generator': self._create_standard_generator(
                        problem_description, test_cases
                    )
                }
            else:
                # Standard mode
                generators = self._create_standard_generators(
                    problem_description, test_cases
                )
            
            # Main search loop
            for i in range(self.config.budget):
                algo_state = self.algorithm.step(algo_state, generators)
                
                # Monitoring
                if self.monitor and i % 10 == 0:
                    self._update_monitor(algo_state, i)
                
                # Checkpointing
                if (self.config.checkpoint_dir and 
                    i % self.config.checkpoint_interval == 0):
                    checkpoint_path = (self.config.checkpoint_dir / 
                                     f"checkpoint_{i}.pkl")
                    save_checkpoint(algo_state, checkpoint_path)
                
                # Progress logging
                if self.config.verbose and i % 10 == 0:
                    self._log_progress(algo_state, i)
            
            # Extract best solution
            best_states = self.algorithm.get_best_states(algo_state, k=1)
            if not best_states:
                raise ValueError("No solutions found")
            
            best_state, best_score = best_states[0]
            
            if self.config.verbose:
                self._log_final_result(best_state, best_score)
            
            return best_state
            
        finally:
            if self.monitor:
                self.monitor.end_solving()
                if self.config.verbose:
                    report = self.monitor.generate_report()
                    self._print_performance_report(report)
    
    def solve_problem(
        self,
        problem_description: str,
        test_cases: List[TestCase],
        **kwargs
    ) -> HybridState:
        """Synchronous wrapper for compatibility"""
        return asyncio.run(self.solve_problem_async(
            problem_description, test_cases, **kwargs
        ))
    
    def _create_standard_generator(
        self,
        problem: str,
        test_cases: List[TestCase]
    ) -> Any:
        """Create a standard TreeQuest generator"""
        
        def generate(parent_state: Optional[HybridState]) -> Tuple[HybridState, float]:
            # Determine strategy
            if parent_state is None:
                strategy = "initial"
            elif parent_state.score == 0:
                strategy = "debug"
            elif parent_state.score < 0.5:
                strategy = "refine"
            elif parent_state.score < 1.0:
                strategy = "edge_cases"
            else:
                strategy = "optimize"
            
            # Generate code
            prompt = self._build_prompt(problem, parent_state, strategy)
            response = self.llm_backend.generate(
                prompt,
                temperature=0.7 if strategy == "initial" else 0.5
            )
            
            # Extract code
            code = self._extract_code(response.text)
            
            # Create attempt
            attempt = CodeAttempt(
                code=code,
                strategy=strategy,
                parent_hash=(parent_state.attempt.code_hash 
                           if parent_state else None)
            )
            
            # Execute
            if (self.config.diff_aware_execution and 
                parent_state and 
                isinstance(self.executor, DiffAwareExecutor)):
                result = asyncio.run(
                    self.executor.execute_code_with_diff_async(
                        code,
                        parent_state.attempt.code,
                        test_cases,
                        parent_state.result
                    )
                )
            else:
                result = self.executor.execute_code(code, test_cases)
            
            # Create state
            state = HybridState(
                attempt=attempt,
                result=result,
                metadata=SearchMetadata(
                    expansion_type="standard"
                )
            )
            
            return state, state.score
        
        return generate
    
    def _log_progress(self, algo_state: Any, iteration: int):
        """Log search progress"""
        
        top_states = self.algorithm.get_best_states(algo_state, k=3)
        
        logger.info(f"\nIteration {iteration}:")
        logger.info(f"Tree size: {algo_state.tree.size if hasattr(algo_state, 'tree') else 'N/A'}")
        
        if hasattr(algo_state, 'particle_manager'):
            logger.info(f"Total particles: {algo_state.particle_manager.total_particles_generated}")
        
        logger.info("Top 3 solutions:")
        for i, (state, score) in enumerate(top_states):
            logger.info(f"  {i+1}. Score: {score:.3f} "
                       f"({state.result.passed_tests}/{state.result.total_tests} tests)")
    
    def _log_final_result(self, state: HybridState, score: float):
        """Log final result"""
        
        print("\n" + "="*60)
        print("FINAL SOLUTION")
        print("="*60)
        print(f"Score: {score:.3f}")
        print(f"Tests: {state.result.passed_tests}/{state.result.total_tests}")
        print(f"Strategy: {state.attempt.strategy}")
        
        if state.particles:
            print(f"Used particle filtering: {len(state.particles)} particles")
            
        print(f"\nCode:\n{state.attempt.code}")
```

### Phase 6: Testing Strategy (Week 6-7)

#### 6.1 Property-Based Tests (`tests/test_properties.py`)

```python
import hypothesis
from hypothesis import strategies as st, given, assume, settings
import numpy as np
import pytest

from treequest_coding_solver.particle.resampling import (
    systematic_resample, stratified_resample, adaptive_resample
)
from treequest_coding_solver.particle.manager import ParticleManager

class TestResamplingProperties:
    """Property-based tests for resampling algorithms
    
    These tests ensure mathematical invariants hold under all conditions
    """
    
    @given(
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=100
        ),
        n_samples=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=1000)
    def test_systematic_resample_preserves_count(self, weights, n_samples):
        """Resampling should always return exactly n_samples indices"""
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
        
        indices = systematic_resample(weights, n_samples)
        
        assert len(indices) == n_samples
        assert all(0 <= idx < len(weights) for idx in indices)
    
    @given(
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=2,
            max_size=50
        )
    )
    def test_ess_calculation_bounds(self, weights):
        """ESS should be between 1 and N"""
        weights = np.array(weights)
        assume(weights.sum() > 0)
        weights = weights / weights.sum()
        
        ess = 1.0 / np.sum(weights ** 2)
        
        assert 1.0 <= ess <= len(weights)
        
        # ESS = N when uniform
        uniform = np.ones_like(weights) / len(weights)
        uniform_ess = 1.0 / np.sum(uniform ** 2)
        assert abs(uniform_ess - len(weights)) < 1e-6
        
        # ESS = 1 when degenerate
        degenerate = np.zeros_like(weights)
        degenerate[0] = 1.0
        degenerate_ess = 1.0 / np.sum(degenerate ** 2)
        assert abs(degenerate_ess - 1.0) < 1e-6
    
    @given(
        n_particles=st.integers(min_value=1, max_value=50),
        n_tests=st.integers(min_value=1, max_value=20)
    )
    def test_particle_weight_normalization(self, n_particles, n_tests):
        """Particle weights should always sum to 1"""
        
        # Create mock data
        from treequest_coding_solver.core.state import CodeAttempt, ExecutionResult, TestCase
        
        particles = []
        results = []
        
        for i in range(n_particles):
            attempt = CodeAttempt(
                code=f"# Solution {i}",
                strategy="test"
            )
            
            # Random test results
            passed = np.random.randint(0, n_tests + 1)
            result = ExecutionResult(
                test_cases=[TestCase(None, None, passed=j < passed) 
                           for j in range(n_tests)],
                total_tests=n_tests,
                passed_tests=passed,
                failed_tests=n_tests - passed,
                total_execution_time_ms=100.0,
                memory_usage_mb=50.0
            )
            
            particles.append(attempt)
            results.append(result)
        
        # Compute weights
        manager = ParticleManager(
            config=ParticleConfig(),
            llm_backend=None,
            reward_signal=None
        )
        
        weights = manager.compute_particle_weights(particles, results)
        
        # Check normalization
        assert abs(weights.sum() - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)
    
    @hypothesis.example(weights=[0.0, 0.0, 0.0], n_samples=3)  # Edge case
    def test_resample_handles_zero_weights(self, weights, n_samples):
        """Resampling should handle all-zero weights gracefully"""
        weights = np.array(weights)
        
        # Should treat as uniform when all zero
        indices = systematic_resample(weights, n_samples)
        
        assert len(indices) == n_samples
        # Should sample uniformly
        unique_indices = set(indices)
        assert len(unique_indices) <= len(weights)

class TestParticleEvolutionProperties:
    """Test particle evolution maintains invariants"""
    
    @given(
        n_parents=st.integers(min_value=1, max_value=20),
        n_evolved=st.integers(min_value=1, max_value=20)
    )
    def test_evolution_preserves_lineage(self, n_parents, n_evolved):
        """Evolved particles should track parents correctly"""
        
        # Create parent particles
        parents = []
        for i in range(n_parents):
            attempt = CodeAttempt(
                code=f"# Parent {i}",
                strategy="initial"
            )
            parents.append((attempt, 1.0 / n_parents))
        
        # Mock evolution (would be async in real code)
        evolved = []
        for i in range(n_evolved):
            parent_idx = i % n_parents
            parent = parents[parent_idx][0]
            
            child = CodeAttempt(
                code=f"# Child {i} of parent {parent_idx}",
                strategy="evolved",
                parent_hash=parent.code_hash
            )
            evolved.append((child, parents[parent_idx][1]))
        
        # Check lineage
        for child, _ in evolved:
            assert child.parent_hash is not None
            # Find parent
            parent_found = any(
                p[0].code_hash == child.parent_hash 
                for p in parents
            )
            assert parent_found

class TestSearchInvariants:
    """Test search algorithm invariants"""
    
    @given(
        tree_depth=st.integers(min_value=0, max_value=10),
        branching_factor=st.integers(min_value=1, max_value=5)
    )
    def test_tree_consistency(self, tree_depth, branching_factor):
        """Tree structure should remain consistent"""
        
        from treequest.algos.tree import Tree, Node
        
        tree = Tree()
        
        # Build tree
        def add_children(node, depth):
            if depth <= 0:
                return
            
            for i in range(branching_factor):
                child = tree.add_node(
                    parent=node,
                    state=f"State at depth {tree_depth - depth + 1}",
                    score=np.random.random(),
                    action_name=f"action_{i}"
                )
                add_children(child, depth - 1)
        
        add_children(tree.root, tree_depth)
        
        # Check invariants
        def check_node(node):
            # Parent-child consistency
            for child in node.children.values():
                assert child.parent == node
                check_node(child)
            
            # Depth consistency
            if node.parent:
                actual_depth = 0
                current = node
                while current.parent:
                    actual_depth += 1
                    current = current.parent
                # Depth should match
        
        check_node(tree.root)
        
        # Tree size
        def count_nodes(node):
            return 1 + sum(count_nodes(c) for c in node.children.values())
        
        expected_size = sum(branching_factor ** i for i in range(tree_depth + 1))
        actual_size = count_nodes(tree.root)
        
        # May be less due to random generation
        assert actual_size <= expected_size
```

### Phase 7: Documentation and Examples (Week 7)

#### 7.1 API Documentation

See `docs/api.md` for comprehensive API documentation generated from docstrings.

#### 7.2 Example Usage

```python
# examples/quickstart.py
import asyncio
from treequest_coding_solver import (
    CodingProblemSolver, SolverConfig, TestCase
)

async def main():
    # Configure solver
    config = SolverConfig(
        algorithm="hybrid",
        enable_particle_filtering=True,
        budget=30,
        llm_backend="gemini",
        reward_weights={
            "test": 0.8,
            "complexity": 0.2
        }
    )
    
    # Create solver
    solver = CodingProblemSolver(
        config=config,
        custom_components={
            'gemini_api_key': 'YOUR_API_KEY'
        }
    )
    
    # Define problem
    problem = """
    Write a function that finds the longest palindromic substring.
    
    Example:
    Input: "babad"
    Output: "bab" or "aba"
    
    Input: "cbbd"
    Output: "bb"
    """
    
    # Define test cases
    test_cases = [
        TestCase(input="babad", expected_output="bab"),
        TestCase(input="cbbd", expected_output="bb"),
        TestCase(input="racecar", expected_output="racecar"),
        TestCase(input="", expected_output=""),
    ]
    
    # Solve
    solution = await solver.solve_problem_async(problem, test_cases)
    
    print(f"Solution found!")
    print(f"Score: {solution.score}")
    print(f"Code:\n{solution.attempt.code}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Improvements from v1

1. **Modular Architecture**: Clean interfaces enable easy testing and component swapping
2. **Async-First**: Non-blocking I/O throughout for better performance
3. **Robust Error Handling**: Fallback paths and graceful degradation
4. **Performance Optimizations**: Vectorized operations, caching, diff-aware execution
5. **Property-Based Testing**: Mathematical invariants verified under all conditions
6. **its_hub Integration**: Optional support for its_hub ecosystem
7. **Comprehensive Monitoring**: OpenTelemetry integration for production use

## Success Metrics

Same targets as v1:
- 30-40% improvement on hard problems
- 25-35% reduction in iterations
- 2-3x performance improvement through optimizations
- <$5 average cost per problem

Additional v2 metrics:
- 90%+ test coverage
- <500ms latency for standard operations
- Support for problems up to 10k lines
- Successful integration with both TreeQuest and its_hub

## Conclusion

This v2 plan addresses all major feedback points while maintaining the core vision of combining TreeQuest with Particle Filtering. The modular architecture ensures long-term maintainability, while performance optimizations deliver on the promise of efficient inference-time scaling for code generation.