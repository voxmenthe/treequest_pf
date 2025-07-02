# TreeQuest: Code-Focused API Reference and Implementation Guide

## Overview

TreeQuest is a Python library for implementing tree search algorithms, with a focus on Adaptive Branching Monte Carlo Tree Search (AB-MCTS) and its variants. This reference provides exact API documentation matching the actual implementation.

## Installation

```bash
# Using uv (recommended)
uv add "treequest[abmcts-m]" "https://github.com/SakanaAI/treequest.git@main"

# Using pip
pip install "treequest[abmcts-m] @ git+https://github.com/SakanaAI/treequest.git"
```

## Core Types and Interfaces

### Type Definitions (src/treequest/types.py)

```python
from typing import TypeVar, Tuple, Callable, Optional, List, Mapping

# Generic type for node states - you can use any type
NodeStateT = TypeVar("NodeStateT")

# Core type: tuple of (state, score)
StateScoreType = Tuple[NodeStateT, float]

# Function signature for node generation
GenerateFnType = Callable[[Optional[NodeStateT]], Tuple[NodeStateT, float]]

# Function signature for custom ranking
RankingFnType = Callable[[List[StateScoreType[NodeStateT]]], List[StateScoreType[NodeStateT]]]
```

### Base Algorithm Interface

All algorithms inherit from `Algorithm[NodeStateT, AlgoStateT]`:

```python
class Algorithm:
    def init_tree(self) -> AlgoStateT:
        """Initialize empty tree state"""
        
    def step(self, 
             state: AlgoStateT, 
             generate_fn: Mapping[str, GenerateFnType[NodeStateT]], 
             inplace: bool = False) -> AlgoStateT:
        """Execute one tree expansion step"""
        
    def get_state_score_pairs(self, state: AlgoStateT) -> List[StateScoreType[NodeStateT]]:
        """Extract all (state, score) pairs from tree"""
```

## Available Algorithms

### 1. AB-MCTS-A: Adaptive Branching MCTS with Node Aggregation

```python
import treequest as tq

# Initialize with default parameters
algo = tq.ABMCTSA()

# Initialize with custom parameters
algo = tq.ABMCTSA(
    dist_type="gaussian",  # or "beta"
    reward_average_priors=None,  # Optional float or dict[str, float]
    prior_config=None,  # Full prior configuration
    model_selection_strategy="multiarm_bandit_thompson"  # or "stack", "multiarm_bandit_ucb"
)
```

**Parameters:**
- `dist_type`: Distribution type for probability modeling
  - `"gaussian"`: Normal-inverse-gamma distribution (default)
  - `"beta"`: Beta distribution with Jeffrey's prior
- `reward_average_priors`: Prior reward averages for actions
- `prior_config`: Full configuration for distribution priors
- `model_selection_strategy`: How to select between GEN/CONT nodes
  - `"multiarm_bandit_thompson"`: Thompson sampling (default)
  - `"stack"`: Traditional separate fits
  - `"multiarm_bandit_ucb"`: Upper Confidence Bound

**Default Prior Configurations:**
```python
# Gaussian prior config
{
    "m": 0,           # mean
    "kappa": 1,       # precision of mean
    "nu": 1,          # degrees of freedom
    "tau_square": 0.1 # scale parameter
}

# Beta prior config
{
    "a": 0.5,  # Jeffrey's prior alpha
    "b": 0.5   # Jeffrey's prior beta
}
```

### 2. AB-MCTS-M: Adaptive Branching MCTS with Bayesian Mixed Model

```python
# Requires PyMC installation: treequest[abmcts-m]
algo = tq.ABMCTSM(
    enable_pruning=True,
    reward_average_priors=None,
    model_selection_strategy="multiarm_bandit_thompson",
    min_subtree_size_for_pruning=4,
    same_score_proportion_threshold=0.75
)
```

**Parameters:**
- `enable_pruning`: Enable automatic pruning of underperforming subtrees
- `reward_average_priors`: Prior reward averages
- `model_selection_strategy`: Same options as AB-MCTS-A
- `min_subtree_size_for_pruning`: Minimum nodes before pruning eligible
- `same_score_proportion_threshold`: Fraction of same scores to trigger pruning

### 3. Standard MCTS

```python
algo = tq.StandardMCTS(
    samples_per_action=2,
    exploration_weight=1.4142135623730951  # sqrt(2)
)
```

**Parameters:**
- `samples_per_action`: Number of samples to generate per action
- `exploration_weight`: UCT exploration coefficient (default: √2)

### 4. Tree of Thoughts BFS

```python
algo = tq.TreeOfThoughtsBFSAlgo(
    breadth_limit=3,  # b parameter
    size_limit=5      # k parameter  
)
```

**Parameters:**
- `breadth_limit`: Number of best nodes to select at each step
- `size_limit`: Number of samples to generate per node

### 5. Best First Search

```python
algo = tq.BestFirstSearchAlgo(num_samples=2)
```

**Parameters:**
- `num_samples`: Number of samples per action

### 6. Multi-Armed Bandit UCB

```python
algo = tq.MultiArmedBanditUCB(exploration_weight=1.4142135623730951)
```

**Parameters:**
- `exploration_weight`: UCB exploration coefficient

## Node and Tree Structure

### Node Class

```python
@dataclasses.dataclass
class Node[NodeStateT]:
    state: Optional[NodeStateT]  # None for root
    score: float                 # [-1.0, 1.0], -1.0 for root
    expand_idx: int             # Expansion order, -1 for root
    parent: Optional['Node']    # Parent node reference
    children: List['Node']      # Child nodes
    
    @property
    def depth(self) -> int:
        """Compute depth by counting parent chain"""
```

### Tree Class

```python
@dataclasses.dataclass
class Tree[NodeStateT]:
    root: Node[NodeStateT]
    size: int  # Total number of nodes
    
    def get_nodes(self) -> List[Node[NodeStateT]]:
        """Get all nodes in the tree"""
        
    def get_state_score_pairs(self) -> List[StateScoreType[NodeStateT]]:
        """Get all (state, score) pairs excluding root"""
        
    def add_node(self, parent: Node[NodeStateT], 
                 state: NodeStateT, score: float) -> Node[NodeStateT]:
        """Add new node to tree"""
```

## Complete Usage Examples

### Example 1: Basic LLM Answer Generation and Refinement

```python
import dataclasses
import treequest as tq
from typing import Optional, Tuple

@dataclasses.dataclass
class AnswerState:
    """State representing an LLM-generated answer"""
    answer_text: str
    feedback: str
    iteration: int

def generate_llm_answer_with_feedback(
    parent_state: Optional[AnswerState]
) -> Tuple[AnswerState, float]:
    """Generate or refine answer based on parent state"""
    
    if parent_state is None:
        # Initial generation from root
        answer_text = call_llm_api("Generate solution for: " + problem_statement)
        feedback = evaluate_answer(answer_text)
        score = compute_score_from_feedback(feedback)
        state = AnswerState(answer_text, feedback, iteration=0)
    else:
        # Refine existing answer
        prompt = f"""Previous answer: {parent_state.answer_text}
        Feedback: {parent_state.feedback}
        Please improve the answer based on this feedback."""
        
        answer_text = call_llm_api(prompt)
        feedback = evaluate_answer(answer_text)
        score = compute_score_from_feedback(feedback)
        state = AnswerState(answer_text, feedback, 
                          iteration=parent_state.iteration + 1)
    
    return state, score

# Create algorithm instance
algo = tq.ABMCTSA(
    dist_type="gaussian",
    model_selection_strategy="multiarm_bandit_thompson"
)

# Initialize tree
tree_state = algo.init_tree()

# Run search for 50 iterations
for i in range(50):
    tree_state = algo.step(
        tree_state, 
        {"generate_or_refine": generate_llm_answer_with_feedback}
    )
    
    # Log progress every 10 steps
    if (i + 1) % 10 == 0:
        best_answers = tq.top_k(tree_state, algo, k=3)
        print(f"Step {i+1}: Top score = {best_answers[0][1]:.3f}")

# Get final results
best_state, best_score = tq.top_k(tree_state, algo, k=1)[0]
print(f"Best answer: {best_state.answer_text}")
print(f"Final score: {best_score}")
```

### Example 2: Multi-Model Search with Different LLMs

```python
from functools import partial
import treequest as tq

def generate_with_specific_llm(
    llm_name: str,
    temperature: float,
    parent_state: Optional[dict] = None
) -> Tuple[dict, float]:
    """Generate using a specific LLM configuration"""
    
    if parent_state is None:
        # Initial generation
        response = call_llm(
            model=llm_name,
            prompt=base_prompt,
            temperature=temperature
        )
    else:
        # Refinement with model-specific prompt
        refinement_prompt = create_refinement_prompt(
            parent_state['response'],
            parent_state['score'],
            llm_name
        )
        response = call_llm(
            model=llm_name,
            prompt=refinement_prompt,
            temperature=temperature
        )
    
    score = evaluate_response(response)
    state = {
        'response': response,
        'model': llm_name,
        'score': score,
        'parent': parent_state
    }
    
    return state, score

# Configure multiple generation functions
generate_functions = {
    "gpt4_creative": partial(generate_with_specific_llm, "gpt-4", 0.8),
    "gpt4_precise": partial(generate_with_specific_llm, "gpt-4", 0.2),
    "claude_balanced": partial(generate_with_specific_llm, "claude-3", 0.5),
    "gemini_analytical": partial(generate_with_specific_llm, "gemini-pro", 0.3),
}

# Use AB-MCTS-M for complex multi-model coordination
algo = tq.ABMCTSM(
    enable_pruning=True,
    min_subtree_size_for_pruning=5
)

tree_state = algo.init_tree()

# Run search
for _ in range(100):
    tree_state = algo.step(tree_state, generate_functions)

# Extract top results grouped by model
all_results = algo.get_state_score_pairs(tree_state)
by_model = {}
for state, score in all_results:
    model = state['model']
    if model not in by_model:
        by_model[model] = []
    by_model[model].append((state, score))

# Print best from each model
for model, results in by_model.items():
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        print(f"{model}: Best score = {results[0][1]:.3f}")
```

### Example 3: Code Generation with Test-Driven Refinement

```python
import treequest as tq
from typing import Optional, Tuple
import subprocess

@dataclasses.dataclass
class CodeState:
    code: str
    test_results: dict
    passes_all_tests: bool
    error_messages: list[str]

def generate_and_test_code(
    parent_state: Optional[CodeState]
) -> Tuple[CodeState, float]:
    """Generate code and run tests to compute score"""
    
    if parent_state is None:
        # Initial generation
        prompt = f"Write a Python function to solve: {problem_description}"
        code = generate_code_with_llm(prompt)
    else:
        # Refine based on test failures
        prompt = f"""Previous code:
{parent_state.code}

Test failures:
{parent_state.error_messages}

Please fix the code to pass all tests."""
        code = generate_code_with_llm(prompt)
    
    # Run tests
    test_results = run_test_suite(code)
    passes_all_tests = all(test_results.values())
    error_messages = extract_error_messages(test_results)
    
    # Score based on test pass rate
    score = sum(test_results.values()) / len(test_results)
    
    state = CodeState(
        code=code,
        test_results=test_results,
        passes_all_tests=passes_all_tests,
        error_messages=error_messages
    )
    
    return state, score

# Use standard MCTS for simpler exploration
algo = tq.StandardMCTS(
    samples_per_action=3,  # Generate 3 variations per action
    exploration_weight=2.0  # Higher exploration for diverse solutions
)

tree_state = algo.init_tree()

# Search until solution found or budget exhausted
for i in range(200):
    tree_state = algo.step(
        tree_state,
        {"generate": generate_and_test_code}
    )
    
    # Check if we found a solution
    best_state, best_score = tq.top_k(tree_state, algo, k=1)[0]
    if best_state.passes_all_tests:
        print(f"Solution found in {i+1} iterations!")
        print(best_state.code)
        break

# Visualize the search tree
tq.visualize_tree_graphviz(
    tree_state.tree,
    save_path="code_search_tree.png",
    title="Code Generation Search Tree"
)
```

### Example 4: Custom State and Scoring with Domain-Specific Logic

```python
import treequest as tq
import numpy as np
from typing import Optional, Tuple

class MoleculeState:
    """Custom state for molecular optimization"""
    def __init__(self, smiles: str, properties: dict):
        self.smiles = smiles
        self.properties = properties
        self.synthesis_score = properties.get('synthesis_score', 0)
        self.activity_score = properties.get('activity_score', 0)
        self.toxicity_score = properties.get('toxicity_score', 1)
    
    def __str__(self):
        return f"Molecule({self.smiles[:20]}...)"

def generate_or_modify_molecule(
    parent_state: Optional[MoleculeState]
) -> Tuple[MoleculeState, float]:
    """Generate new molecule or modify existing one"""
    
    if parent_state is None:
        # Generate initial molecule
        smiles = generate_random_molecule()
    else:
        # Apply chemical transformation
        transformation = select_transformation_based_on_scores(parent_state)
        smiles = apply_transformation(parent_state.smiles, transformation)
    
    # Compute properties
    properties = compute_molecular_properties(smiles)
    
    # Multi-objective scoring
    score = compute_composite_score(
        synthesis=properties['synthesis_score'],
        activity=properties['activity_score'],
        toxicity=properties['toxicity_score'],
        weights=[0.3, 0.5, 0.2]
    )
    
    state = MoleculeState(smiles, properties)
    return state, score

# Use Tree of Thoughts for systematic exploration
algo = tq.TreeOfThoughtsBFSAlgo(
    breadth_limit=5,  # Keep top 5 molecules at each level
    size_limit=10     # Generate 10 variations per molecule
)

tree_state = algo.init_tree()

# Run optimization
for depth in range(10):
    tree_state = algo.step(
        tree_state,
        {"modify": generate_or_modify_molecule}
    )
    
    # Get current best molecules
    top_molecules = tq.top_k(tree_state, algo, k=5)
    print(f"Depth {depth+1}: Best score = {top_molecules[0][1]:.3f}")

# Extract all molecules above threshold
all_molecules = algo.get_state_score_pairs(tree_state)
good_molecules = [(state, score) for state, score in all_molecules if score > 0.8]
print(f"Found {len(good_molecules)} molecules with score > 0.8")
```

### Example 5: Checkpointing and Resuming Searches

```python
import pickle
import treequest as tq

def save_tree_state(tree_state, filename):
    """Save tree state to disk"""
    with open(filename, 'wb') as f:
        pickle.dump(tree_state, f)

def load_tree_state(filename):
    """Load tree state from disk"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Start search
algo = tq.ABMCTSA()
tree_state = algo.init_tree()

# Run first batch
for i in range(50):
    tree_state = algo.step(tree_state, generate_functions)
    
    # Checkpoint every 10 steps
    if (i + 1) % 10 == 0:
        save_tree_state(tree_state, f"checkpoint_{i+1}.pkl")

# Save final state
save_tree_state(tree_state, "search_state_50.pkl")

# Later: Resume from checkpoint
tree_state = load_tree_state("search_state_50.pkl")

# Continue search
for i in range(50, 100):
    tree_state = algo.step(tree_state, generate_functions)

print(f"Total tree size: {tree_state.tree.size}")
```

## Advanced Features

### Custom Ranking Functions

```python
def custom_ranking_function(
    state_score_pairs: List[Tuple[StateT, float]]
) -> List[Tuple[StateT, float]]:
    """Custom ranking logic"""
    # Example: Penalize very long answers
    adjusted_pairs = []
    for state, score in state_score_pairs:
        if hasattr(state, 'answer_length'):
            # Penalize answers over 1000 chars
            penalty = max(0, (state.answer_length - 1000) / 5000)
            adjusted_score = score * (1 - penalty)
        else:
            adjusted_score = score
        adjusted_pairs.append((state, adjusted_score))
    
    # Sort by adjusted score
    return sorted(adjusted_pairs, key=lambda x: x[1], reverse=True)

# Use custom ranking
best_results = tq.top_k(
    tree_state, 
    algo, 
    k=10,
    ranking_fn=custom_ranking_function
)
```

### Visualization

```python
from treequest.visualization import visualize_tree_graphviz

# Basic visualization
visualize_tree_graphviz(
    tree_state.tree,
    save_path="search_tree.png"
)

# Detailed visualization
visualize_tree_graphviz(
    tree_state.tree,
    save_path="detailed_tree.pdf",
    format="pdf",
    show_scores=True,
    max_label_length=50,
    title="AB-MCTS Search Tree - Iteration 100"
)
```

### Algorithm-Specific Features

#### AB-MCTS-A Node Types
```python
# AB-MCTS-A creates special internal nodes
# GEN nodes: Represent "generate new" action
# CONT nodes: Represent "continue with existing" action
# These are handled internally by the algorithm
```

#### AB-MCTS-M Pruning
```python
# Automatic pruning removes underperforming subtrees
algo = tq.ABMCTSM(
    enable_pruning=True,
    min_subtree_size_for_pruning=4,  # Need 4+ nodes
    same_score_proportion_threshold=0.75  # 75%+ same score triggers pruning
)
```

## Best Practices

1. **Score Normalization**: Always return scores in [0, 1] range
2. **State Design**: Keep states serializable for checkpointing
3. **Generation Functions**: Should be deterministic given same input for reproducibility
4. **Memory Management**: For large searches, consider periodic pruning or depth limits
5. **Parallel Execution**: Generation functions can parallelize internally
6. **Error Handling**: Wrap generation functions with try-except for robustness

## Common Patterns

### Pattern 1: Feedback-Driven Refinement
```python
def refine_with_feedback(parent_state):
    if parent_state is None:
        return initial_generation()
    else:
        feedback = analyze_weaknesses(parent_state)
        return generate_improvement(parent_state, feedback)
```

### Pattern 2: Multi-Stage Generation
```python
generate_functions = {
    "brainstorm": brainstorm_ideas,
    "elaborate": elaborate_idea,
    "polish": polish_final_version,
}
```

### Pattern 3: Constraint Satisfaction
```python
def generate_with_constraints(parent_state):
    candidate = generate_candidate(parent_state)
    if satisfies_hard_constraints(candidate):
        score = evaluate_soft_constraints(candidate)
        return candidate, score
    else:
        return candidate, 0.0  # Invalid solutions get zero score
```

## Performance Considerations

- **Tree Size**: Grows linearly with iterations
- **Memory Usage**: Approximately 1KB per node (depends on state size)
- **Computation**: Dominated by generation function calls
- **Scaling**: 
  - Standard MCTS: O(samples_per_action × actions × iterations)
  - AB-MCTS variants: O(iterations) with adaptive branching
  - Tree of Thoughts: O(breadth_limit × size_limit × depth)

## Troubleshooting

### Issue: Scores not in [0, 1] range
```python
# Add score normalization
def safe_generate(parent_state):
    state, raw_score = original_generate(parent_state)
    normalized_score = max(0.0, min(1.0, raw_score))
    return state, normalized_score
```

### Issue: Memory growth with large trees
```python
# Limit tree depth
def depth_limited_generate(parent_state):
    if parent_state and get_depth(parent_state) >= MAX_DEPTH:
        return None  # Stop expanding
    return normal_generate(parent_state)
```

### Issue: Slow PyMC sampling in AB-MCTS-M
```python
# Use AB-MCTS-A for faster inference
algo = tq.ABMCTSA()  # Instead of ABMCTSM()
```

## Summary

TreeQuest provides a flexible framework for tree-based search with a focus on LLM applications. The key abstractions are:

1. **States**: Any Python object representing a node's content
2. **Scores**: Normalized feedback signals in [0, 1]
3. **Generation Functions**: Maps from optional parent state to (state, score)
4. **Algorithms**: Different strategies for tree expansion and exploration

The library's strength lies in its ability to adaptively balance exploration (trying new approaches) and exploitation (refining promising solutions) through sophisticated algorithms like AB-MCTS.