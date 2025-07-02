# TreeQuest Coding Problem Solver - Comprehensive Implementation Plan

## Executive Summary

This document outlines a detailed implementation plan for extending TreeQuest to solve coding problems using test-driven development (TDD) with LLM APIs (Gemini and OpenAI). The system will leverage AB-MCTS algorithms to efficiently search through the solution space, using test feedback as the scoring mechanism.

## Architecture Overview

### TreeQuest Components We'll Use

1. **From `treequest` (top-level exports):**
   - `Algorithm` - Base class for search algorithms
   - `StandardMCTS` - Standard Monte Carlo Tree Search
   - `ABMCTSA` - Adaptive Branching MCTS with aggregation
   - `ABMCTSM` - Adaptive Branching MCTS with mixed models
   - `TreeOfThoughtsBFSAlgo` - Tree of Thoughts BFS algorithm
   - `top_k` - Extract best solutions from tree

2. **From `treequest.algos.tree` (direct import needed):**
   - `Tree` - The search tree structure
   - `Node` - Individual nodes in the tree

3. **From `treequest.visualization` (for debugging):**
   - `visualize_tree_graphviz` - Visualize search tree

4. **From `treequest.types` (for type hints):**
   - `NodeStateT` - Type variable for node states
   - `StateScoreType` - Type for (state, score) tuples
   - `GenerateFnType` - Type for generation functions

### Components We Need to Implement

1. **CodingProblemState** - Represents a node in the search tree containing:
   - Generated code solution
   - Test results and feedback
   - Score (based on test pass rate)
   - Prompt used for generation
   - LLM metadata (model, temperature, etc.)

2. **TestExecutionFramework** - Handles:
   - Test case parsing from problem description
   - Safe code execution in sandboxed environment
   - Result collection and scoring
   - Error handling and timeout management

3. **LLMIntegrationLayer** - Manages:
   - API connections to Gemini and OpenAI
   - Prompt engineering for code generation
   - Response parsing and validation
   - Rate limiting and error handling

4. **GenerationFunctions** - TreeQuest-compatible functions that:
   - Generate initial solutions from problem description
   - Refine existing solutions based on test feedback
   - Handle different coding strategies (algorithmic, optimization, debugging)

## Detailed Implementation Plan

### Phase 1: Foundation (Week 1)

#### 1.1 State Classes Design (`src/treequest/coding_solver/state.py`)

```python
@dataclasses.dataclass
class CodingProblemState:
    """State representing a code solution attempt"""
    problem_description: str
    generated_code: str
    test_results: TestExecutionResults
    score: float  # 0.0 to 1.0, based on test pass rate
    prompt_used: str
    llm_metadata: LLMMetadata
    generation_strategy: str  # 'initial', 'refine', 'debug', 'optimize'
    parent_state_id: Optional[str] = None
    
@dataclasses.dataclass
class TestExecutionResults:
    """Results from executing tests against generated code"""
    total_tests: int
    passed_tests: int
    failed_tests: List[TestFailure]
    execution_time_ms: float
    memory_usage_mb: float
    error_messages: List[str]
    
@dataclasses.dataclass
class LLMMetadata:
    """Metadata about the LLM call"""
    model_name: str  # 'gemini-2.0-flash', 'gpt-4o', etc.
    temperature: float
    max_tokens: int
    api_call_duration_ms: float
    token_count: TokenCount
```

#### 1.2 Test Execution Framework (`src/treequest/coding_solver/test_executor.py`)

```python
class SafeCodeExecutor:
    """Execute code safely with resource limits"""
    
    def execute_code_with_tests(
        self,
        code: str,
        test_cases: List[TestCase],
        timeout_seconds: float = 5.0,
        memory_limit_mb: int = 512
    ) -> TestExecutionResults:
        """Execute code against test cases with safety constraints"""
        
class TestCaseParser:
    """Parse test cases from various formats"""
    
    def parse_from_problem_description(
        self,
        problem_description: str
    ) -> List[TestCase]:
        """Extract test cases from problem description"""
        
    def parse_from_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[TestCase]:
        """Convert examples to executable test cases"""
```

### Phase 2: LLM Integration (Week 1-2)

#### 2.1 Base LLM Client (`src/treequest/coding_solver/llm/base.py`)

```python
class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate_code_solution(
        self,
        problem_description: str,
        previous_attempt: Optional[CodingProblemState] = None,
        strategy: str = "initial"
    ) -> Tuple[str, LLMMetadata]:
        """Generate a code solution"""
        
    @abstractmethod
    async def analyze_test_failures(
        self,
        code: str,
        test_failures: List[TestFailure]
    ) -> str:
        """Analyze test failures and suggest improvements"""
```

#### 2.2 Gemini Client (`src/treequest/coding_solver/llm/gemini_client.py`)

```python
class GeminiCodeSolverClient(BaseLLMClient):
    """Gemini API client for code generation"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.client = genai.GenerativeModel(model)
        genai.configure(api_key=api_key)
        
    async def generate_code_solution(
        self,
        problem_description: str,
        previous_attempt: Optional[CodingProblemState] = None,
        strategy: str = "initial"
    ) -> Tuple[str, LLMMetadata]:
        """Generate code using Gemini API"""
        prompt = self._build_prompt(problem_description, previous_attempt, strategy)
        # Implementation details...
```

#### 2.3 OpenAI Client (`src/treequest/coding_solver/llm/openai_client.py`)

```python
class OpenAICodeSolverClient(BaseLLMClient):
    """OpenAI API client for code generation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        
    async def generate_code_solution(
        self,
        problem_description: str,
        previous_attempt: Optional[CodingProblemState] = None,
        strategy: str = "initial"
    ) -> Tuple[str, LLMMetadata]:
        """Generate code using OpenAI API"""
        # Implementation details...
```

### Phase 3: TreeQuest Integration (Week 2)

#### 3.1 Generation Functions (`src/treequest/coding_solver/generators.py`)

```python
from typing import Callable, Dict, List, Optional, Tuple
import asyncio

# TreeQuest imports
from treequest.algos.tree import Node  # Not exported at top level
from treequest.types import GenerateFnType, StateScoreType

class CodingProblemGeneratorFactory:
    """Factory for creating TreeQuest-compatible generation functions"""
    
    def __init__(
        self,
        problem_description: str,
        test_cases: List[TestCase],
        llm_clients: Dict[str, BaseLLMClient],
        executor: SafeCodeExecutor
    ):
        self.problem_description = problem_description
        self.test_cases = test_cases
        self.llm_clients = llm_clients
        self.executor = executor
        
    def create_initial_solution_generator(
        self,
        llm_name: str
    ) -> Callable[[Optional[CodingProblemState]], Tuple[CodingProblemState, float]]:
        """Create generator for initial solutions
        
        NOTE: TreeQuest expects synchronous generation functions.
        We'll use asyncio.run() to bridge async LLM calls.
        """
        
        def generate(parent_state: Optional[CodingProblemState]) -> Tuple[CodingProblemState, float]:
            if parent_state is not None:
                raise ValueError("Initial generator expects no parent state")
                
            # Generate code (bridge async to sync)
            code, llm_metadata = asyncio.run(
                self.llm_clients[llm_name].generate_code_solution(
                self.problem_description,
                    strategy="initial"
                )
            )
            
            # Execute tests
            test_results = self.executor.execute_code_with_tests(code, self.test_cases)
            
            # Calculate score
            score = test_results.passed_tests / test_results.total_tests
            
            # Create state
            state = CodingProblemState(
                problem_description=self.problem_description,
                generated_code=code,
                test_results=test_results,
                score=score,
                prompt_used="initial_generation",
                llm_metadata=llm_metadata,
                generation_strategy="initial"
            )
            
            return state, score
            
        return generate
        
    def create_refinement_generator(
        self,
        llm_name: str,
        strategy: str = "refine"
    ) -> Callable[[Optional[CodingProblemState]], Tuple[CodingProblemState, float]]:
        """Create generator for refining existing solutions"""
        # Similar implementation...
```

#### 3.2 Main Solver (`src/treequest/coding_solver/solver.py`)

```python
from typing import Dict, List, Optional, Union
import asyncio

# TreeQuest imports - use top-level exports where possible
from treequest import (
    Algorithm,
    StandardMCTS,
    ABMCTSA,
    ABMCTSM,
    TreeOfThoughtsBFSAlgo,
    top_k,
)
from treequest.algos.tree import Tree, Node  # These aren't exported at top level
from treequest.visualization import visualize_tree_graphviz  # For debugging

# Our imports (these are modules we'll create)
from .state import CodingProblemState
from .test_executor import SafeCodeExecutor, TestCase
from .generators import CodingProblemGeneratorFactory
from .llm.gemini_client import GeminiCodeSolverClient
from .llm.openai_client import OpenAICodeSolverClient

class CodingProblemSolver:
    """Main solver using TreeQuest AB-MCTS"""
    
    def __init__(
        self,
        gemini_api_key: str,
        openai_api_key: str,
        algorithm: str = "abmcts-m"  # or "abmcts-a"
    ):
        self.llm_clients = {
            "gemini": GeminiCodeSolverClient(gemini_api_key),
            "gpt-4o": OpenAICodeSolverClient(openai_api_key)
        }
        self.executor = SafeCodeExecutor()
        self.algorithm = self._create_algorithm(algorithm)
    
    def _create_algorithm(self, algorithm_name: str) -> Algorithm:
        """Create the specified TreeQuest algorithm"""
        if algorithm_name == "abmcts-m":
            return ABMCTSM()  # Uses PyMC for mixed modeling
        elif algorithm_name == "abmcts-a":
            return ABMCTSA()  # Uses aggregation approach
        elif algorithm_name == "standard-mcts":
            return StandardMCTS()  # Traditional MCTS
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
    async def solve_problem(
        self,
        problem_description: str,
        test_cases: List[TestCase],
        budget: int = 50,
        strategies: List[str] = ["initial", "refine", "debug"]
    ) -> CodingProblemState:
        """Solve a coding problem using AB-MCTS"""
        
        # Create generator factory
        factory = CodingProblemGeneratorFactory(
            problem_description,
            test_cases,
            self.llm_clients,
            self.executor
        )
        
        # Create generation functions for different strategies and LLMs
        generate_fns = {}
        for llm_name in self.llm_clients:
            for strategy in strategies:
                key = f"{llm_name}_{strategy}"
                if strategy == "initial":
                    generate_fns[key] = factory.create_initial_solution_generator(llm_name)
                else:
                    generate_fns[key] = factory.create_refinement_generator(llm_name, strategy)
                    
        # Initialize search tree using TreeQuest's init_tree
        # Note: init_tree returns the algorithm-specific state, not AlgorithmState
        tree_state = self.algorithm.init_tree()
        
        # Run search using TreeQuest's step function
        for i in range(budget):
            # algorithm.step expects:
            # - tree_state: algorithm-specific state
            # - generate_fns: Dict[str, Callable]
            tree_state = self.algorithm.step(tree_state, generate_fns)
            
            # Log progress using TreeQuest's top_k utility
            if (i + 1) % 10 == 0:
                # top_k returns List[Tuple[state, score]]
                top_solutions = top_k(tree_state, self.algorithm, k=1)
                if top_solutions:
                    best_state, score = top_solutions[0]
                    print(f"Iteration {i+1}: Best score = {score:.3f}")
                
        # Return best solution using TreeQuest's top_k
        top_solutions = top_k(tree_state, self.algorithm, k=1)
        if not top_solutions:
            raise ValueError("No solutions found")
        best_state, _ = top_solutions[0]
        return best_state
```

### Phase 4: Testing Framework (Week 3)

#### 4.1 Unit Tests (`tests/test_coding_solver/`)

```python
# test_treequest_integration.py
import pytest
from unittest.mock import Mock, patch
from treequest import ABMCTSA, top_k
from treequest.algos.tree import Tree, Node

def test_generation_function_compatibility():
    """Test our generation functions work with TreeQuest"""
    # Create a mock generation function
    def mock_generate(parent_state):
        return "test_state", 0.8
    
    # Test with ABMCTSA
    algo = ABMCTSA()
    algo_state = algo.init_tree()
    
    # Should be able to call step without errors
    algo_state = algo.step(algo_state, {"test_action": mock_generate})
    
    # Should have created nodes
    # Check that nodes were created
    # Note: The exact structure depends on the algorithm implementation
    assert tree_state is not None

def test_state_extraction_from_tree():
    """Test we can extract our custom states from TreeQuest tree"""
    # Implementation...
```

```python
# test_state.py
def test_coding_problem_state_initialization():
    """Test state initialization and score calculation"""
    
# test_executor.py
def test_safe_code_execution_with_timeout():
    """Test code execution with timeout constraints"""
    
def test_memory_limit_enforcement():
    """Test memory limit enforcement during execution"""
    
# test_llm_clients.py
@pytest.mark.asyncio
async def test_gemini_code_generation():
    """Test Gemini client code generation (with mocks)"""
    
# test_generators.py
def test_generation_function_scoring():
    """Test that generation functions produce correct scores"""
    
# test_solver.py
@pytest.mark.integration
async def test_end_to_end_problem_solving():
    """Test complete problem solving pipeline"""
```

#### 4.2 Integration Tests (`tests/integration/`)

```python
@pytest.mark.slow
@pytest.mark.requires_api_keys
async def test_real_leetcode_easy_problem():
    """Test with real LeetCode easy problem"""
    
@pytest.mark.slow
@pytest.mark.requires_api_keys
async def test_multi_llm_collaboration():
    """Test that multiple LLMs work together effectively"""
```

### Phase 5: Example Scripts and Documentation (Week 3-4)

#### 5.1 Example: LeetCode Problem Solver (`examples/leetcode_solver.py`)

```python
import os
import asyncio
# Note: We're creating the coding_solver module, it's not part of TreeQuest
from coding_solver import CodingProblemSolver, TestCase

async def solve_two_sum():
    """Example: Solve LeetCode Two Sum problem"""
    problem = """
    Given an array of integers nums and an integer target, 
    return indices of the two numbers such that they add up to target.
    
    Example 1:
    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
    """
    
    test_cases = [
        TestCase(input={"nums": [2,7,11,15], "target": 9}, expected=[0,1]),
        TestCase(input={"nums": [3,2,4], "target": 6}, expected=[1,2]),
        TestCase(input={"nums": [3,3], "target": 6}, expected=[0,1])
    ]
    
    solver = CodingProblemSolver(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    solution = await solver.solve_problem(problem, test_cases, budget=30)
    print(f"Best solution (score: {solution.score}):")
    print(solution.generated_code)
```

#### 5.2 Example: Competitive Programming (`examples/competitive_programming.py`)

```python
async def solve_codeforces_problem():
    """Example: Solve a Codeforces-style problem"""
    # Implementation...
```

### Phase 6: Advanced Features (Week 4+)

#### 6.1 Enhanced Scoring System

```python
class MultiCriteriaScorer:
    """Score solutions based on multiple criteria"""
    
    def calculate_score(
        self,
        test_results: TestExecutionResults,
        code: str,
        problem_type: str
    ) -> float:
        """Calculate composite score"""
        # Consider:
        # - Test pass rate (primary)
        # - Execution time (for optimization problems)
        # - Code complexity (cyclomatic complexity)
        # - Memory usage
        # - Code style/readability
```

#### 6.2 Problem Type Detection

```python
class ProblemTypeClassifier:
    """Classify problem types to choose appropriate strategies"""
    
    def classify(self, problem_description: str) -> ProblemType:
        """Detect if problem is algorithmic, DP, graph, etc."""
```

#### 6.3 Learning from History

```python
class HistoricalLearner:
    """Learn from past solutions to improve future performance"""
    
    def extract_patterns(self, successful_solutions: List[CodingProblemState]):
        """Extract successful patterns from past solutions"""
        
    def suggest_strategies(self, problem: str) -> List[str]:
        """Suggest strategies based on similar past problems"""
```

## TreeQuest-Specific Integration Notes

### Key Integration Points

1. **State Storage in Nodes:**
   - TreeQuest stores our `CodingProblemState` objects in `Node.state`
   - We access them via `node.state` when traversing the tree

2. **Generation Function Signature:**
   - TreeQuest expects: `Callable[[Optional[State]], Tuple[State, float]]`
   - Must be synchronous (we use `asyncio.run()` for async LLM calls)
   - Parent state can be `None` for root expansions

3. **Algorithm State Management:**
   - Each algorithm returns its own state type from `init_tree()`
   - ABMCTSA returns a state with the tree and aggregation data
   - ABMCTSM returns a state with the tree and PyMC mixed model data
   - StandardMCTS returns a state with the tree and visit counts

4. **Action Types:**
   - TreeQuest uses action strings (keys in `generate_fns` dict)
   - We encode strategy and LLM as: `"{llm_name}_{strategy}"`
   - Example: `"gemini_refine"`, `"gpt-4o_debug"`

5. **Score Normalization:**
   - TreeQuest expects scores in [0, 1] range
   - We use test pass rate as primary score
   - Can add weighted factors for other metrics

### TreeQuest Configuration

```python
# Example: Using TreeQuest algorithms
from treequest import ABMCTSA, ABMCTSM, StandardMCTS

# ABMCTSA - uses aggregation approach
algo_a = ABMCTSA()

# ABMCTSM - uses PyMC mixed models (requires treequest[abmcts-m])
algo_m = ABMCTSM()

# StandardMCTS - traditional MCTS
algo_std = StandardMCTS()

# Note: Algorithm parameters are typically set internally
# Check the source code for specific configuration options
```

## Testing Strategy

### Unit Testing
- Test each component in isolation with mocked dependencies
- Focus on edge cases and error handling
- Achieve >90% code coverage

### Integration Testing
- Test LLM client integration with rate limiting
- Test code execution with various edge cases
- Test full pipeline with simple problems

### End-to-End Testing
- Use real coding problems from:
  - LeetCode (Easy, Medium, Hard)
  - Codeforces
  - Project Euler
- Measure success rates and API usage
- Compare against baselines (repeated sampling, sequential refinement)

### Performance Testing
- Benchmark execution speed
- Monitor memory usage
- Track API costs

## Deployment Considerations

### API Key Management
- Use environment variables for API keys
- Implement key rotation mechanism
- Add usage tracking and alerts

### Rate Limiting
- Implement exponential backoff
- Queue requests appropriately
- Monitor API quotas

### Monitoring
- Log all LLM calls with costs
- Track success rates by problem type
- Monitor execution times

### Security
- Sandbox all code execution
- Validate generated code before execution
- Implement resource limits

## Success Metrics

1. **Primary Metrics**
   - Problem solve rate (% of problems solved correctly)
   - API efficiency (problems solved per API dollar)
   - Time to solution

2. **Secondary Metrics**
   - Code quality scores
   - Execution efficiency of generated code
   - Diversity of solution approaches

3. **Comparison Baselines**
   - Repeated sampling with same budget
   - Sequential refinement
   - Single-shot generation

## Timeline

- **Week 1**: Foundation - State classes, test execution framework
- **Week 2**: LLM Integration - Implement Gemini and OpenAI clients
- **Week 3**: TreeQuest Integration - Generation functions, main solver
- **Week 4**: Testing & Documentation - Comprehensive tests, examples
- **Week 5+**: Advanced Features - Multi-criteria scoring, learning

## Risks and Mitigations

1. **Risk**: LLM API costs exceed budget
   - **Mitigation**: Implement strict budgeting, use cheaper models for exploration

2. **Risk**: Code execution security vulnerabilities
   - **Mitigation**: Use proper sandboxing, limit resources, validate inputs

3. **Risk**: Poor initial performance
   - **Mitigation**: Start with easy problems, iterate on prompts, analyze failures

4. **Risk**: Integration complexity
   - **Mitigation**: Modular design, comprehensive testing, gradual feature addition

## Implementation Checklist

### What TreeQuest Provides:
- [x] Search algorithms (ABMCTSA, ABMCTSM, StandardMCTS)
- [x] Tree structure and node management
- [x] Thompson sampling for selection
- [x] Posterior updates and Bayesian inference
- [x] Utilities for extracting best solutions
- [x] Visualization support (GraphViz)

### What We Need to Implement:
- [ ] CodingProblemState dataclass
- [ ] Test execution framework with sandboxing
- [ ] LLM clients (Gemini, OpenAI)
- [ ] Generation function factory
- [ ] Score calculation from test results
- [ ] Async-to-sync bridge for LLM calls
- [ ] Problem parsing and test extraction
- [ ] Example scripts and documentation

## Conclusion

This implementation plan provides a comprehensive roadmap for extending TreeQuest to solve coding problems using test-driven development. By leveraging TreeQuest's AB-MCTS algorithms with multiple LLMs, we create a powerful system that adaptively searches for optimal solutions while using test feedback to guide the search process.

The key insight is that TreeQuest handles all the complex search logic - we just need to provide:
1. A state representation (CodingProblemState)
2. Generation functions that produce (state, score) tuples
3. LLM integration for code generation
4. Test execution for scoring

The modular architecture ensures maintainability and extensibility, while the comprehensive testing strategy ensures reliability. The phased approach allows for iterative development and early validation of core concepts.