arXiv:2503.04412v3 [cs.AI] 27 Jun 2025


# Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search

| Author Name | Affiliation |
|-------------|-------------|
| Yuichi Inoue* | Sakana AI, Japan |
| Kou Misaki* | Sakana AI, Japan |
| Yuki Imajuku | Sakana AI, Japan |
| So Kuroki | Sakana AI, Japan |
| Taishi Nakamura | Sakana AI, Japan |
| Takuya Akiba | Sakana AI, Japan |

{y.inoue, takiba}@sakana.ai

## Abstract

Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in real tasks like coding. In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling.

## 1 Introduction

Recent work has shown that inference-time scaling, namely allocating more computation at inference time, can markedly boost the performance of large language models (LLMs) on complex tasks. As outlined in Section 2, existing approaches to inference-time scaling fall into three broad categories: (1) post-training fine-tuning, (2) reward-guided chain-of-thought (CoT) generation, and (3) multiple answer generation. In this paper, we focus on the third category. The multiple answer generation approach repeatedly queries an LLM at non-zero temperature to produce a set of candidate outputs and then selects the most promising one. This approach enhances the LLM's problem-solving abilities on-the-fly, without further training [1–11]. Because it is orthogonal to the other two families, it can be seamlessly combined with them.

The most widely successful approach in this category is repeated sampling, which includes techniques such as best-of-n sampling, majority voting, and self-consistency [2, 3, 12]. In repeated sampling, an LLM at non-zero temperature generates multiple candidate outputs independently from the same initial prompt, and a final solution is selected, typically by a simple heuristic. This paradigm has proved effective on challenging benchmarks, including coding competitions [1, 3] and ARC-AGI [13]. The strategy leverages the diverse and vast output space exposed by LLM generation, and sampling more responses increases the odds that one of them is high-quality. The empirical success of repeated sampling underscores that harnessing this diversity is central to effective inference-time scaling.

However, repeated sampling focuses exclusively on exploration and lacks an explicit mechanism for exploitation. In certain real-world scenarios, one can obtain external feedback on a candidate solution. For instance, in coding tasks, one can run tests to evaluate the correctness of generated


*Equal contribution. See author contributions for details.

Preprint. Under review.

---


## Figure 1: Visual comparison of AB-MCTS vs. baselines

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Characteristics</th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline 1: Repeated Sampling</td>
<td>Task Prompt → Answers (Sample n answers)</td>
<td>Repeated sampling only goes wide (i.e. pure exploration)</td>
</tr>
<tr>
<td>Baseline 2: Sequential Refinement</td>
<td>Task Prompt → Initial Answer → Refine n-1 times → Refined Answer</td>
<td>Sequential refinement only goes deep (i.e. pure exploitation)</td>
</tr>
<tr>
<td>Baseline 3: Standard MCTS</td>
<td>Tree structure with fixed branching factor (3 in this example)</td>
<td>MCTS expands a node at most once with a fixed width</td>
</tr>
<tr>
<td>Ours: Adaptive Branching MCTS</td>
<td>Tree structure with adaptive branching when appropriate</td>
<td>AB-MCTS systematically decides whether to go wider or deeper</td>
</tr>
</tbody>
</table>

**Figure 1:** Visual comparison of AB-MCTS vs. baselines. Unlike baselines that are purely wide (repeated sampling), purely deep (sequential refinement), or fixed-width (standard MCTS), AB-MCTS dynamically decides whether to branch outward or drill down, unifying both search directions.

programs and gather feedback on how to improve them [4, 5, 14]. In such settings, it is natural to select promising solutions and refine them based on available feedback, which repeated sampling alone cannot accomplish effectively.

Several approaches [6, 7, 9–11, 15] have been proposed for exploration and exploitation in such multi-turn settings, but the majority were designed before the power of inference-time scaling was fully recognized. Consequently, these methods use a fixed "width", i.e., they treat the number of answers generated from a single prompt as a fixed hyperparameter. For example, methods based on standard Monte Carlo Tree Search (MCTS) use a fixed branching factor (i.e., the number of child nodes per state) as a hyperparameter [9–11, 15]. As demonstrated by the success of repeated sampling, effective inference-time scaling requires leveraging a diverse and vast output space, so a fixed width hinders scaling.

In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework that generalizes repeated sampling with multi-turn exploration and exploitation (Figure 1). The main technical challenge is to introduce unbounded branching into MCTS. Unlike traditional MCTS, AB-MCTS does not fix the width as a static hyperparameter. Instead, at each node of the search tree, AB-MCTS adaptively decides whether to explore ("go wider") by generating new candidate responses or exploit ("go deeper") by refining existing ones, leveraging external feedback signals. Under the hood, we formalize our decision process via Bayesian posterior updates, ensuring each expansion balances exploration and exploitation in a principled manner. This design naturally extends repeated sampling, allowing us to harness the diverse and vast output space of LLMs when necessary. Consequently, our framework provides a powerful mechanism for balancing exploration and exploitation in the context of LLM inference-time scaling.

We evaluated AB-MCTS on complex coding and machine learning engineering benchmarks [1, 16], as well as ARC-AGI [17], using frontier models such as GPT-4o [18] and DeepSeek-V3 [19], in a scenario that scales up inference-time compute by allowing multiple generation calls for each task instance. Under the same computational budget, AB-MCTS achieved better results than previous approaches, such as repeated sampling and standard MCTS.

**Contributions.** ① We highlight the challenge of effectively incorporating unbounded branching into tree search. This is pivotal for combining the power of the diverse and vast output space of LLMs, a cornerstone of inference-time scaling, with solution refinement. ② To address this challenge, we introduce AB-MCTS, which systematically decides whether to "go wider" or "go deeper." We present two variants, AB-MCTS-M and AB-MCTS-A, based on different principles, each offering distinct trade-offs. ③ In a practical setting using frontier models and real-world complex tasks, we show that AB-MCTS outperforms existing methods.


2

---


## 2 Related Work

**Inference-Time Scaling by Post-Training Fine-Tuning.** Recent post-training work, exemplified by OpenAI o1/o3 [20, 21], uses reinforcement learning or supervised CoT fine-tuning to deepen LLM reasoning and boost single-answer quality [20–25]. Our approach instead generates many candidates and refines them with external feedback, pursuing a complementary objective.

**Inference-Time Scaling via Reward-Guided CoT.** Reward-guided CoT scales inference by searching one step (typically a sentence) at a time [26–34]. Primarily for math tasks, it aims to improve single-answer quality, making it orthogonal to our multiple-answer generation approach.

**Inference-Time Scaling by Multiple Answer Generation.** Since the community has come to appreciate the power of inference-time scaling, the strategy that has been studied widely is repeated sampling, in which the model generates many candidate answers and selects the best one [1–3, 35]. Although empirically strong and widely used, repeated sampling leaves obvious room for improvement because it does not refine its candidates using external feedback [4, 5]. Before the era of large-scale inference-time compute, a variety of task-specific strategies were proposed for relatively small scales; examples include tree expansions directed by LLMs [6] and Bayesian methods [7].

The most common alternative, however, relies on MCTS. LATS [9], RAP [15], SWE-Search [10], and RepoUnderstander [11] all employ MCTS in different settings while leaving the core algorithm unchanged; in every case, they use the conventional version with a fixed branching factor. Because we focus on the core search algorithms, we group these previous works under the term "standard MCTS" and compare them with our proposed framework in the experiments.

**Progressive Widening in MCTS.** Progressive widening [36, 37] is a classic technique that gradually increases actions considered per node. It was designed for games with unique actions and no side information for untried moves, relying on visit-count heuristics. Our approach differs as new branches are sampled from the same LLM. This homogeneity in generation allows a principled statistical rule for choosing between widening and deepening.

## 3 Method

### 3.1 Preliminaries

First, we introduce the setup and notation, with detailed elaboration provided in Appendix A.1. We consider a setting where an LLM, represented by a function f<sub>LLM</sub>, receives a textual prompt t<sub>in</sub> containing (1) task instructions with optional few-shot examples, and/or (2) previously generated outputs along with external feedback, and generates an answer t<sub>out</sub> = f<sub>LLM</sub>(t<sub>in</sub>). A scoring function R then evaluates an answer t<sub>out</sub> to produce a score r = R(t<sub>out</sub>), where higher scores indicate better performance. We typically assume that the score r is normalized to the range [0, 1], but our framework allows for arbitrary ranges as well. Our goal is to find an output t<sub>out</sub> that attains a high score r under limited calls to the LLM at inference time. Such tasks arise, for example, in code generation, where the correctness or quality of the output can be quantified; for instance, R may execute the generated code and return the fraction of test cases passed. In some cases, the true score evaluator may be inaccessible (e.g., hidden test cases), so we assume we have access to some surrogate or partial evaluator R, such as a public test evaluator, during the search. We aim to leverage this evaluator to guide an efficient search for better solutions.

### 3.2 Adaptive Branching MCTS

**MCTS for LLM-based Answer Generation.** We perform the answer search by constructing a search tree T, in which each non-root node N is associated with an LLM-generated answer to a given task. Our goal is to construct T so that it contains answers with scores as high as possible.

For this purpose, we employ MCTS, formulated iteratively as follows. Starting from a single root node, we perform n<sub>nodes</sub> iterations, each adding one new node, resulting in a total of 1 + n<sub>nodes</sub> nodes. Each iteration has three steps: (1) **Selection**, where we select a node N for expansion; (2) **Expansion**, where we expand N by generating a new answer from node N, creating a new child node N<sub>new</sub> and appending it to N. Specifically, if N is the root, the new answer is directly generated from the task prompt; if N is non-root, the new answer refines the answer associated with N, using external feedback; and (3) **Score backup**, where we propagate the score of N<sub>new</sub> up toward the root of the


3

---


**Figure 2:** Example tree structure and score posterior predictive distributions for AB-MCTS with mixed models (AB-MCTS-M). The left side shows an AB-MCTS-M Example Tree with root node N, branching into nodes N<sub>1</sub>, N<sub>2</sub>, and N<sub>3</sub> with various scores (r=0.8, r=0, r=0.2, etc.) and GEN nodes. Actions a<sub>0</sub>, a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub> are shown on the branches. The right side shows a Posterior Predictive Distribution graph with curves for a<sub>0</sub>, a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub> plotted against r from 0.0 to 1.0. Here, a<sub>1</sub> leads to a set of child nodes with higher scores, causing a peak at larger r. As more child samples are collected, the variance of the distribution decreases.

**Algorithm 1** Adaptive Branching MCTS

```
1:  function AB-MCTS(n_nodes)
2:    T ← INITIALIZETREE( )
3:    for n = 1, . . . , n_nodes do
4:        N ← SELECTEXPANSIONTARGET(T)                           ▷ Step 1. Select an expansion target
5:        N_new ← EXPAND(N, T)                  ▷ Step 2. Expand the selected node to generate a child
6:        SCOREBACKUP(N_new, T)                     ▷ Step 3. Backup the score from the generated node
7:    return SELECTBEST(T)
8:  function SELECTEXPANSIONTARGET(T)
9:    N ← GETROOT(T)
10:    while not ISLEAF(N) do
11:        N_next ← SELECTCHILD(N, T)                               ▷ Detailed in Sections 3.3 and 3.4
12:        if ISGENNODE(N_next) then                ▷ If a GEN node is selected, branch off from the node
13:           break
14:        N ← N_next
15:     return N
```

tree T. We adopt different backup rules in our proposed methods (see Section 3.3 and 3.4). In our setting, no separate rollout is needed, since each node's score r can be evaluated directly once an output is generated. After n<sub>nodes</sub> iterations, we select the best node based on a chosen criterion.

In standard MCTS, only leaf nodes are selected and expanded, (i.e. each node is expanded at most once), and the expansion adds a fixed number of child nodes. However, since each query to an LLM at non-zero temperature can yield different outputs from the same prompt, the branching factor is theoretically infinite. To accommodate such unbounded branching, we relax the standard MCTS constraints and allow selection and expansion of non-leaf nodes. Moreover, recent studies [3] suggest that drawing many outputs from the same prompt at non-zero temperature can improve performance. Allowing unbounded branching enables us to fully exploit these varied samples, whereas restricting the branching factor could miss correct answer generations and undermine overall performance.

**Adaptive Branching via the GEN Node.** To fully leverage the potential performance improvement from unbounded branching, we allow nodes that have already been expanded once to be expanded again and further branched, unlike in standard MCTS. To explicitly represent the action of generating new child nodes, we introduce a GEN node. Every node N (including newly expanded ones during iterations) has a GEN node as a child. When the GEN node with parent node N is selected during the selection step, we expand N to add a new child to it. Algorithm 1 outlines this approach, called Adaptive Branching Monte Carlo Tree Search (AB-MCTS).

The only remaining component we need is a selection policy, including when to select a GEN node. We propose two algorithms with different selection policies: AB-MCTS-M (Mixed model) and AB-MCTS-A (node Aggregation). Both follow the overall procedure in Algorithm 1 and use Thompson sampling to balance exploration and exploitation.

**Thompson Sampling for Node Selection.** In our proposed methods, we employ a Bayesian approach with Thompson sampling for node selection. In standard MCTS, the usual selection policy is based on UCT score [38], but since GEN nodes have no child nodes, UCT scores cannot be assigned. Thompson sampling not only resolves this issue but also enables parallel node expansion. This is


4

---


particularly beneficial when evaluating node scores is time-consuming, as in the case of MLE-Bench (See Appendix B.1 for MLE-Bench details). Next, we discuss how to utilize Thompson sampling to choose between a GEN node and existing child nodes during the selection step at node N. At node N, we have 1 + n<sub>child</sub> possible actions: choosing a GEN node or one of the n<sub>child</sub> existing nodes. Since our purpose is to expand the node with score as high as possible, we model the probability distributions of the score of an eventually expanded node at that iteration after choosing actions (i.e. the score of the node N<sub>new</sub> at Line 5 of Algorithm 1). Thompson sampling proceeds by first sampling scores from the modeled probability distributions for all the actions, and then choosing the action with the highest sampled score. A key question is how to model the expanded node's score probability distribution for each action. We address this with two strategies: a mixed Bayesian model (AB-MCTS-M) and a node aggregation method (AB-MCTS-A). In both cases, we model the score probability distributions by Bayesian posterior predictives, but with different statistical models.

## 3.3 AB-MCTS-M: Adaptive Branching MCTS with Mixed Model

During the selection at node N, estimating the score of a GEN node is challenging because it does not directly correspond to a specific LLM-generated observation. To address this, we employ a separate, node-specific mixed model fitted individually at each node during the selection step. In a mixed model, we assign a "group" label to each node, and nodes sharing the same label share a model parameter. This shared model parameter captures common characteristics of each group, such as the answer quality of the parent node of each group. In our setting, denoting each child node of N as N<sub>j</sub> (j = 1, . . . , n<sub>child</sub>), each group corresponds to the set of nodes T<sub>sub</sub>(N<sub>j</sub>) (i.e. all the nodes in the subtree under N<sub>j</sub>, see Figure 2), which consist of answers generated by refinement from the answer of the common ancestor node N<sub>j</sub>. Our assumption is that even after multiple stages of refinements, the quality associated with N<sub>j</sub>'s answer continues to be represented by this shared parameter (see Appendix A.2.2 for model parameter details). In contrast, the GEN node corresponds to a new group with no scores observed yet, but we note that, since it uses the shared parameter as the other groups, the other groups' node scores affect the GEN node's score distribution as well.

### Algorithm Outline

As described in Section 3.2, our goal is to model the probability distribution of the scores for the nodes that will eventually be expanded after choosing a GEN node or any child node during the selection step at node N. AB-MCTS-M assigns each subtree under N<sub>j</sub>, denoted as T<sub>sub</sub>(N<sub>j</sub>), as a distinct group (see Figure 2 for example subtree). The mixed model then leverages observed scores from these groups to compute the posterior predictive distributions of expected scores for new nodes generated from each group (See Figure 2 for a schematic illustration and Appendix A.2.2 for a detailed formulation of the mixed model.) We sample the scores from all the groups (the GEN node and T<sub>sub</sub>(N<sub>j</sub>)) using calculated posterior predictives. If the GEN node's sampled score is highest, we call f<sub>LLM</sub> to generate a new child node. Otherwise, we choose the child node N<sub>j</sub> with the highest score and continue the sampling step.

### Score Backup Mechanism

When a new node N is created, its observed score is added to the histories of N and its ancestors. This cumulative record is used to update the posterior distributions in the mixed model. The observed score is not backed up to a GEN node, but it indirectly influences the GEN node's score probability distribution through the shared parameters in the mixed model (see Appendix A.2.2 for details).

## 3.4 AB-MCTS-A: Adaptive Branching MCTS with Node Aggregation

In AB-MCTS-M, during the selection step at N, we use a mixed model that shares statistical strength across groups through the shared model parameters. In contrast, AB-MCTS-A is designed in the same spirit as the standard UCT-based MCTS, and there are no shared model parameters among the different actions. This design simplifies the statistical modeling and makes the computation more lightweight compared to AB-MCTS-M.

The major problem is how to back up scores to GEN nodes. Since the generated node is not attached as a child to a GEN node, it makes the backup of scores difficult to define. Here, we introduce a CONT node at the same tree level as all the GEN nodes (see Figure 3). Intuitively, the CONT node represents the action of continuing refinement from the current answer at node N, rather than generating a new node. By explicitly separating these two actions–generating new answers (GEN) and refining existing answers (CONT)–we create a clear path for score propagation. Specifically, the


5

---


score of the expanded node is first backed up to the GEN node, and since all ancestors of the GEN node are either nodes with LLM answers or CONT nodes, the score subsequently does not propagate through other GEN nodes (see Figure 3 for an example tree).

**Algorithm Outline.** AB-MCTS-A aggregates all child nodes under a single CONT node, which represents refinements from existing child nodes (see Figure 3). We model each node's score probability in a Bayesian framework and perform Thompson sampling on posterior predictives to decide between generating a new child (GEN) or refining an existing one (CONT). In contrast to AB-MCTS-M, we do not use shared parameters among different node probability distributions.

<table>
<thead>
<tr><th colspan="3">AB-MCTS-A Example Tree</th></tr>
</thead>
<tbody>
<tr><td colspan="3" style="text-align: center;">N<br/>a<sub>0</sub></td></tr>
<tr><td style="text-align: center;">GEN</td><td colspan="2" style="text-align: center;">CONT<br/>a<sub>1</sub> a<sub>2</sub> a<sub>3</sub></td></tr>
<tr><td></td><td style="text-align: center;">r=0.8</td><td style="text-align: center;">r=0</td><td style="text-align: center;">r=0.2</td></tr>
<tr><td style="text-align: center;">GEN<br/>r=0.8</td><td style="text-align: center;">CONT<br/>r=1.0</td><td style="text-align: center;">GEN</td><td style="text-align: center;">GEN<br/>r=0.3</td><td style="text-align: center;">CONT</td></tr>
</tbody>
</table>

For probability distributions, we use exponential families with conjugate priors to enable analytic and efficient posterior updates. We employ two variants: (1) AB-MCTS-A (Gaussian), using a normal-inverse-χ<sup>2</sup> prior for unbounded scores, and (2) AB-MCTS-A (Beta), using a Beta prior for scores in [0, 1]. See Figure 3 for example tree, and Appendix A.3 for detailed parameter updates.

Figure 3: **Example tree structure for AB-MCTS-A.** All child nodes are aggregated under a CONT node, and a GEN node doesn't have child nodes.

**Score Backup Mechanism.** During score-backup operations, the expanded node score is backed up to the GEN node, which led to the expansion of that node and the GEN node's ancestors. As we can see from Figure 3, a GEN node's ancestors include only generated nodes and CONT nodes, so the score is backed up to a GEN node only from the node that is created by choosing that GEN node. The backed-up score is used to update prior probability distribution parameters.

## 4 Experiments

### 4.1 Experimental Setup

**Benchmarks.** We evaluated AB-MCTS on four diverse benchmarks that require complex problem-solving: LiveCodeBench [14], CodeContest [1], ARC-AGI [17], and MLE-Bench [16]. LiveCodeBench and CodeContest consist of competitive programming problems that demand mathematical and algorithmic reasoning. ARC-AGI involves abstracting a common transformation rule from visual patterns and implementing it as code. MLE-Bench, derived from Kaggle competitions, involves constructing and optimizing machine learning models to achieve high scores based on the evaluation metrics of each competition. For all these benchmarks, LLMs generate Python code to solve each task, and external feedback (e.g., test case results, validation scores) is available to guide the search. More details on the benchmarks can be found in Appendix B.1.

**Models.** We perform our experiments using GPT-4o (gpt-4o-2024-08-06) [18], and DeepSeek-V3 (deepseek-chat) [19]. Each LLM generates a complete solution in a single API call. We define the generation budget as the maximum number of API calls and set the generation budget, which we set to 2<sup>7</sup> = 128. The temperature was set to 0.6 for GPT-4o following [3] and 1.0 for DeepSeek-V3 following the official document.

**Baselines.** We benchmark AB-MCTS against three representative approaches. (1) **Repeated Sampling (Best-of-n)** [1, 3, 39] independently generates up to n candidate solutions from a single LLM prompt, a simple yet competitive baseline for coding tasks. (2) **Sequential Refinement** [4] iteratively improves each solution by re-prompting the LLM with its own output and feedback. (3) **Standard MCTS** backs projects such as LATS [9], RAP [15], SWE-Search [10], and RepoUnderstander [11]. As we design a core search algorithm, we group and compare them as standard MCTS. Following the previous work [9, 15], each expansion adds five child nodes, and the search proceeds until it reaches the 2<sup>7</sup> nodes. Hyper-parameters for AB-MCTS are summarized in Appendix B.2.

### 4.2 Results


6

---


**Table 1: Performance of AB-MCTS against baselines across benchmarks and models.** This table compares AB-MCTS with the baseline methods. Evaluations were performed on LiveCodeBench, CodeContest, and ARC-AGI using GPT-4o and DeepSeek-V3 with a maximum generation budget (2<sup>7</sup>). Each entry provides a performance score (higher values are better) and its corresponding rank (in parentheses, 1st is best). The "Avg. Rank" column shows the average rank across all settings.

<table>
<thead>
<tr>
<th>Method</th>
<th colspan="2">LiveCodeBench</th>
<th colspan="2">CodeContest</th>
<th colspan="2">ARC-AGI</th>
<th>Avg. Rank</th>
</tr>
<tr>
<th></th>
<th>GPT-4o</th>
<th>DeepSeek-V3</th>
<th>GPT-4o</th>
<th>DeepSeek-V3</th>
<th>GPT-4o</th>
<th>DeepSeek-V3</th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>Repeated Sampling [1, 3, 39]</td>
<td>37.8 (4)</td>
<td>40.7 (6)</td>
<td>37.9 (4)</td>
<td>43.2 (5)</td>
<td>15.0 (1)</td>
<td>18.6 (1)</td>
<td>3.5</td>
</tr>
<tr>
<td>Sequential Refinement [4]</td>
<td>37.8 (4)</td>
<td>41.6 (5)</td>
<td>30.1 (6)</td>
<td>41.6 (6)</td>
<td>8.7 (6)</td>
<td>10.0 (6)</td>
<td>5.5</td>
</tr>
<tr>
<td>Standard MCTS [9, 15]</td>
<td>36.7 (6)</td>
<td>43.2 (1)</td>
<td>37.5 (5)</td>
<td>43.8 (3)</td>
<td>9.0 (5)</td>
<td>14.0 (5)</td>
<td>4.2</td>
</tr>
<tr>
<td>AB-MCTS-M</td>
<td>38.9 (2)</td>
<td>43.0 (2)</td>
<td>40.6 (1)</td>
<td>44.6 (2)</td>
<td>12.3 (4)</td>
<td>16.0 (3)</td>
<td>2.3</td>
</tr>
<tr>
<td>AB-MCTS-A (Gaussian)</td>
<td>39.1 (1)</td>
<td>42.5 (3)</td>
<td>40.2 (3)</td>
<td>43.4 (4)</td>
<td>13.0 (3)</td>
<td>18.3 (2)</td>
<td>2.7</td>
</tr>
<tr>
<td>AB-MCTS-A (Beta)</td>
<td>38.7 (3)</td>
<td>42.3 (4)</td>
<td>40.4 (2)</td>
<td>44.8 (1)</td>
<td>14.0 (2)</td>
<td>16.6 (4)</td>
<td>2.7</td>
</tr>
</tbody>
</table>

As detailed in Tables 1 and 2, our comprehensive evaluations reveal AB-MCTS as a consistently superior approach across diverse benchmarks and LLMs, achieving the top average rank and outperforming established baselines. This consistent success stems from the distinctive ability of AB-MCTS to dynamically adapt its search strategy by precisely balancing exploration and exploitation to the varying demands of each problem, an adaptability largely absent in baseline methods. We next detail these results by benchmark, followed by an analysis of the search behavior of AB-MCTS.

**Table 2: Performance on MLE-Bench tasks.** AB-MCTS-M demonstrates robust performance, achieving the best average rank across diverse ML tasks.

<table>
<thead>
<tr>
<th>Method</th>
<th>Nomad2018</th>
<th>Spooky.</th>
<th>Pizza.</th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Repeated Sampling</td>
<td>0.065 (3)</td>
<td>0.47 (4)</td>
<td>0.72 (2)</td>
<td>3.0</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>0.059 (1)</td>
<td>0.46 (3)</td>
<td>0.62 (3)</td>
<td>2.3</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>0.076 (4)</td>
<td>0.45 (2)</td>
<td>0.60 (4)</td>
<td>3.3</td>
</tr>
<tr>
<td>AB-MCTS-M</td>
<td>0.060 (2)</td>
<td>0.38 (1)</td>
<td>0.72 (1)</td>
<td>1.3</td>
</tr>
</tbody>
</table>

**LiveCodeBench and CodeContest.** Figure 4 (left and center) reports the success rate (Pass@1) versus the generation budget for GPT-4o on LiveCodeBench and CodeContest. As expected, all methods demonstrate improved performance with increasing computational budget. On both benchmarks, AB-MCTS algorithms generally outperform the baseline methods. Notably, on LiveCodeBench (Figure 4 left), AB-MCTS starts to pull ahead of the baselines even with a small budget of 2<sup>3</sup>. On CodeContest (Figure 4 center), AB-MCTS demonstrates superior performance compared to baselines at larger budgets of 2<sup>5</sup> and beyond. Appendix Figure 8 shows that while standard MCTS performs relatively well with DeepSeek-V3 compared to GPT-4o, our proposed methods achieve a comparable success rate on LiveCodeBench and surpass standard MCTS on CodeContest.

**ARC-AGI.** Figure 4 (right) shows the performance on ARC-AGI, a particularly challenging benchmark. Following ARC-AGI's official evaluation protocol, we report Pass@2. Consistent with previous work [13], repeated sampling proves to be a strong baseline in our setup, indicating the importance of broad exploration for this task. While standard MCTS yields only marginal improvements with larger budgets, our AB-MCTS framework achieves performance comparable to repeated sampling. This suggests AB-MCTS's capability to effectively explore potentially by dynamically widening its search when beneficial. Similar results were observed with DeepSeek-V3, as detailed in Appendix Figure 8.

**MLE-Bench.** Table 2 and Appendix Figure 10 present the performance on three competitions from MLE-Bench using GPT-4o. Since MLE-Bench requires substantial GPU resources for training and evaluating machine learning models, we exclusively used GPT-4o and focused on the baseline methods and AB-MCTS-M (see also Appendix B.1). The best-performing baseline method varies across competitions. This again highlights that different tasks benefit from different exploration-exploitation trade-offs. In contrast, AB-MCTS-M consistently delivers strong performance in these tasks. This consistent success across diverse competitions underscores AB-MCTS-M's inherent strength in effectively adapting its search strategy to varying problem structures.

## 4.3 Analysis

**Analysis of Search Behavior: Width vs. Depth.** To quantitatively analyze how AB-MCTS balances exploration and exploitation, we examined the average depth and the average width at each depth of the generated search trees. Figure 5 shows that AB-MCTS methods tend to generate wider trees


7

---


## Figure 4: Performance comparison on LiveCodeBench, CodeContest, and ARC-AGI

<table>
<thead>
<tr>
<th>Benchmark</th>
<th>Performance Metric</th>
<th>Methods Compared</th>
</tr>
</thead>
<tbody>
<tr>
<td>LiveCodeBench (higher is better)</td>
<td>Success Rate (Pass@1)</td>
<td rowspan="3">
• Repeated Sampling<br>
• AB-MCTS-M<br>
• Sequential Refinement<br>
• AB-MCTS-A (Gaussian)<br>
• Standard MCTS<br>
• AB-MCTS-A (Beta)
</td>
</tr>
<tr>
<td>CodeContest (higher is better)</td>
<td>Success Rate (Pass@1)</td>
</tr>
<tr>
<td>ARC-AGI (higher is better)</td>
<td>Success Rate (Pass@1)</td>
</tr>
</tbody>
</table>

We compare the six methods using GPT-4o by plotting the success rate against the generation budget. The inset plots provide a detailed view of performance at a maximum generation budget (2<sup>7</sup>); the mean success rate, its 95% confidence interval, and the results from the individual runs are shown. Variance at a generation budget of 2<sup>0</sup> arises from conducting each experiment independently with nonzero temperature. See Figure 7 for experiments on ARC-AGI with a larger budget.

## Figure 5: Comparing algorithms by search tree shape and performance

<table>
<thead>
<tr>
<th>Benchmark</th>
<th>X-axis</th>
<th>Y-axis</th>
<th>Methods Compared</th>
</tr>
</thead>
<tbody>
<tr>
<td>LiveCodeBench (higher is better)</td>
<td rowspan="3">Mean Depth / Mean Width (log2)</td>
<td rowspan="3">Success Rate (Pass@1)</td>
<td rowspan="3">
• Repeated Sampling<br>
• AB-MCTS-M<br>
• Sequential Refinement<br>
• AB-MCTS-A (Gaussian)<br>
• Standard MCTS<br>
• AB-MCTS-A (Beta)
</td>
</tr>
<tr>
<td>CodeContest (higher is better)</td>
</tr>
<tr>
<td>ARC-AGI (higher is better)</td>
</tr>
</tbody>
</table>

Each point shows the performance against the average tree shape for a given algorithm at a specific generation budget. The x-axis represents the log-ratio of mean depth to mean width. Mean width is the average number of nodes per depth. Larger and smaller x-axis values indicate deeper and wider searches, respectively.

compared to standard MCTS. This occurs because AB-MCTS can adaptively decide to explore wider (select the GEN node) from any existing node, unlike standard MCTS. This mechanism allows for more flexible exploration across various tree depths. In addition to this flexibility in exploring wider, as seen in Table 2, AB-MCTS also achieves strong performance on benchmarks where sequential refinement excels, suggesting that AB-MCTS effectively identifies and exploits promising branches by selecting existing child nodes for refinement. This adaptive nature allows it to combine the strengths of exploration and exploitation, resulting in robust performance across diverse benchmarks.

**Scaling with Increased Budget.** Highly complex problems often require a substantial generation budget to find a correct solution. ARC-AGI is a prime example, where extensive exploration via repeated sampling is known to improve performance even at large budgets as reported in [13]. To investigate the scaling properties of our approach, we extended the experiments on ARC-AGI using DeepSeek-V3 with a larger generation budget up to 2<sup>9</sup> = 512. As shown in Figure 7, the performance of AB-MCTS continues to improve substantially as the budget increases from 200 to 500, while the improvement rate of repeated sampling begins to plateau. Standard MCTS also continues to improve with a larger budget, yet shows a significantly lower success rate compared to the AB-MCTS methods. This performance gap highlights that AB-MCTS is more effective at directing its search towards promising branches within the search tree at large computational scales.


8

<table>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td></td>
        <td></td>
        <td rowspan="2"></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td></td>
        <td></td>
    </tr></table>

<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
    </tr>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th colspan="2" rowspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2" rowspan="2"></th>
    </tr>
    <tr>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th colspan="2"></th>
    </tr></table>

<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th colspan="2" rowspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th rowspan="2"></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
    </tr></table>

<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    </thead>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    </thead>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

<table>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

---


## Random Acts of Pizza: AB-MCTS-M

**Figure 6:** Example search tree generated by AB-MCTS-M on Random Acts of Pizza (MLE-Bench). The figure shows how AB-MCTS-M dynamically balances exploration and exploitation. Each node represents a solution. Nodes are colored according to their evaluation score used as the search signal for AB-MCTS-M. The number inside each node indicates the order of generation. Grey nodes mark candidates whose code failed to execute and therefore received no score.

<table>
<thead>
<tr><th>Node Level</th><th>Node Numbers</th><th>Score Range</th></tr>
</thead>
<tbody>
<tr><td>Root</td><td>Root</td><td>0.78</td></tr>
<tr><td>Level 1</td><td>1, 2, 3, 4, 5, 11, 16, 24, 68, 72, 90, 107, 120, 123, 126</td><td>0.76</td></tr>
<tr><td>Level 2</td><td>9, 10, 6, 7, 18, 19, 22, 30, 38, 45, 56, 62, 69, 71, 91, 116, 8, 12, 114</td><td>0.74</td></tr>
<tr><td>Level 3</td><td>13, 31, 58, 92, 105, 14, 15, 17, 20, 23, 85, 21, 36, 26, 35, 65, 122, 44, 46, 59, 43, 52, 86, 48, 49, 118, 70, 73, 83, 104, 121, 76, 80, 84, 95, 97, 78, 79, 99, 109, 112, 127</td><td>0.72</td></tr>
<tr><td>Level 4</td><td>29, 34, 40, 47, 50, 93, 81, 128, 25, 27, 64, 28, 54, 124, 66, 117, 32, 57, 87, 88, 33, 37, 39, 103, 102, 77, 94, 63, 75, 51, 61, 110, 60, 106, 53, 100, 96, 74, 89, 119</td><td>0.70</td></tr>
<tr><td>Level 5</td><td>55, 41, 42, 82, 125, 67, 98, 101, 113, 111, 108</td><td>0.68</td></tr>
<tr><td>Level 6</td><td>115</td><td>0.68</td></tr>
</tbody>
</table>

**Qualitative Analysis of the Search Trees.** Figure 6 and Appendix Figure 11 present example search trees generated by AB-MCTS-M and standard MCTS. These visualizations illustrate more adaptive branching by AB-MCTS-M compared to standard MCTS. This adaptive nature reveals that AB-MCTS-M flexibly balances exploration and exploitation throughout the search process, dynamically allocating budget to explore diverse new candidates ("going wider") and refine promising ones ("going deeper"). Further discussion can be found in Appendix C.3.

**Efficiency and Performance against Repeated Sampling.** While repeated sampling benefits from potential efficiencies such as parallel sampling and no feedback computation costs, our results demonstrate the significant advantages of AB-MCTS. On ARC-AGI, where repeated sampling is notably strong, AB-MCTS not only continues to improve with increased budget but also ultimately achieves performance levels that repeated sampling cannot reach (Figure 7). Furthermore, on LiveCodeBench and CodeContest, AB-MCTS variants can reach the peak performance of repeated sampling substantially earlier in many cases (Figure 4, Appendix Figure 8). This indicates that even when accounting for the inherent advantages of repeated sampling, AB-MCTS emerges as a promising approach to efficiently use the generation budget to achieve superior results in diverse scenarios.

**ARC-AGI Accuracy (higher is better)**

**Figure 7:** Performance comparison on ARC-AGI with increased budget. Scalability of AB-MCTS was assessed with a generation budget extended up to 512. Plotted points represent moving averages to clarify performance trends.

<table>
<thead>
<tr><th>Generation Budget</th><th>Success Rate (Pass@2)</th></tr>
</thead>
<tbody>
<tr><td>0</td><td>0.00</td></tr>
<tr><td>100</td><td>0.05</td></tr>
<tr><td>200</td><td>0.10</td></tr>
<tr><td>300</td><td>0.15</td></tr>
<tr><td>400</td><td>0.18</td></tr>
<tr><td>500</td><td>0.20</td></tr>
</tbody>
</table>

## 5 Conclusions

This paper introduced Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework to enhance LLM performance on complex tasks by effectively integrating multi-turn exploration and exploitation. Unlike previous methods, AB-MCTS dynamically decides to "go wider" or "go deeper" based on external feedback, leveraging Bayesian decision-making. Our experimental results show AB-MCTS outperforms repeated sampling and standard MCTS, demonstrating the value of adaptively handling the challenge of unbounded branching for effective inference-time scaling.

**Limitations.** Our approach assumes the existence of a reliable score evaluator, but developing such an evaluator itself can be challenging depending on the task. Future work could also explore search strategies that incorporate more fine-grained real-world cost factors beyond API call counts, potentially enhancing the practical utility of AB-MCTS. We believe that addressing these challenges will further enhance the applicability of AB-MCTS across a wider range of problems.


9

<table>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td>Repeated 
Sequential 
Standard </td>
        <td>Sampling
Refinement
MCTS</td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td>AB−MCTS
AB−MCTS
AB−MCTS</td>
        <td>−M
−A 
−A (Beta)</td>
        <td>(Gaussian)</td>
    </tr></table>

---


## Author Contributions

Kou Misaki co-designed AB-MCTS and implemented its algorithm and core experimental code. Yuichi Inoue designed and led the experiments, proposed and conducted the experimental analysis, and co-led experimental code development. Yuki Imajuku conducted the CodeContest and LiveCodeBench experiments and co-led experimental code development. So Kuroki implemented and conducted the MLE-Bench experiments. Taishi Nakamura conducted the ARC-AGI experiments. Takuya Akiba initiated the project, co-designed AB-MCTS, advised on experimental code design, and provided overall supervision. All authors contributed to experimental code development, results interpretation, and manuscript refinement.

## Acknowledgements

The authors would like to thank Edoardo Cetin, Luke Darlow, Taro Makino, Kosuke Nakago, Makoto Shing, and Yutaro Yamada for helpful feedback on an earlier version of the draft.

## References

[1] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. *Science*, 378(6624):1092–1097, 2022.

[2] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In *International Conference on Learning Representations*, 2023.

[3] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

[4] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. *Advances in Neural Information Processing Systems*, 36, 2024.

[5] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems*, 2024.

[6] Jierui Li, Hung Le, Yingbo Zhou, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. CodeTree: Agent-guided tree search for code generation with large language models. In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 3711–3726, 2025.

[7] Hao Tang, Keya Hu, Jin Peng Zhou, Si Cheng Zhong, Wei-Long Zheng, Xujie Si, and Kevin Ellis. Code repair with LLMs gives an exploration-exploitation tradeoff. In *Advances in Neural Information Processing Systems*, 2024.

[8] Kuang-Huei Lee, Ian Fischer, Yueh-Hua Wu, Dave Marwood, Shumeet Baluja, Dale Schuurmans, and Xinyun Chen. Evolving deeper llm thinking. arXiv preprint arXiv:2501.09891, 2025.

[9] Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In *International Conference on Machine Learning*, 2024.

[10] Antonis Antoniades, Albert Örwall, Kexun Zhang, Yuxi Xie, Anirudh Goyal, and William Yang Wang. SWE-search: Enhancing software agents with monte carlo tree search and iterative refinement. In *International Conference on Learning Representations*, 2025.

[11] Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin Li. How to understand whole software repository? arXiv preprint arXiv:2406.01422, 2024.


10

---

| Reference Number | Reference Text |
|------------------|----------------|
| [12] | Zhenwen Liang, Ye Liu, Tong Niu, Xiangliang Zhang, Yingbo Zhou, and Semih Yavuz. Improving llm reasoning through scaling inference computation with collaborative verification. arXiv preprint arXiv:2410.05318, 2024. |
| [13] | Ryan Greenblatt. Getting 50% (sota) on arc-agi with gpt-4o. https://www.lesswrong.com/posts/Rdwui3wHxCeKb7feK/getting-50-sota-on-arc-agi-with-gpt-4o, 2024. Accessed: January 21, 2025. |
| [14] | Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In *International Conference on Learning Representations*, 2025. |
| [15] | Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. Reasoning with language model is planning with world model. In *Empirical Methods in Natural Language Processing*, pages 8154–8173, 2023. |
| [16] | Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Aleksander Madry, and Lilian Weng. MLE-bench: Evaluating machine learning agents on machine learning engineering. In *International Conference on Learning Representations*, 2025. |
| [17] | François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019. |
| [18] | OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. |
| [19] | DeepSeek-AI. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. |
| [20] | OpenAI. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. |
| [21] | OpenAI. Competitive programming with large reasoning models. arXiv preprint arXiv:2502.06807, 2025. |
| [22] | DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. |
| [23] | Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025. |
| [24] | Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025. |
| [25] | Kimi Team. Kimi k1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025. |
| [26] | Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems*, 2024. |
| [27] | Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling test-time compute optimally can be more effective than scaling LLM parameters. In *International Conference on Learning Representations*, 2025. |
| [28] | Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. Alphamath almost zero: Process supervision without process. In *Advances in Neural Information Processing Systems*, 2024. |
| [29] | Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, et al. Omni-MATH: A universal olympiad level mathematic benchmark for large language models. In *International Conference on Learning Representations*, 2025. |
| [30] | Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. Marco-o1: Towards open reasoning models for open-ended solutions. arXiv preprint arXiv:2411.14405, 2024. |


11

---


| Reference Number | Reference Text |
|------------------|----------------|
| [31] | Zhenting Qi, Mingyuan MA, Jiahang Xu, Li Lyna Zhang, Fan Yang, and Mao Yang. Mutual reasoning makes smaller LLMs stronger problem-solver. In *International Conference on Learning Representations*, 2025. |
| [32] | Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rstar-math: Small llms can master math reasoning with self-evolved deep thinking. arXiv preprint arXiv:2501.04519, 2025. |
| [33] | Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for LLM problem-solving. In *International Conference on Learning Representations*, 2025. |
| [34] | Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. ReST-MCTS*: LLM self-training via process reward guided tree search. In *Advances in Neural Information Processing Systems*, 2024. |
| [35] | Rylan Schaeffer, Joshua Kazdan, John Hughes, Jordan Juravsky, Sara Price, Aengus Lynch, Erik Jones, Robert Kirk, Azalia Mirhoseini, and Sanmi Koyejo. How do large language monkeys get their power (laws)?, 2025. |
| [36] | Rémi Coulom. Computing "elo ratings" of move patterns in the game of go. *ICGA journal*, 30(4):198–208, 2007. |
| [37] | Adrien Couëtoux, Jean-Baptiste Hoock, Nataliya Sokolovska, Olivier Teytaud, and Nicolas Bonnard. Continuous upper confidence trees. In *Learning and Intelligent Optimization: 5th International Conference, LION 5, Rome, Italy, January 17-21, 2011. Selected Papers 5*, pages 433–445. Springer, 2011. |
| [38] | Levente Kocsis and Csaba Szepesvári. Bandit based monte-carlo planning. In *European conference on machine learning*, pages 282–293. Springer, 2006. |
| [39] | Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In *International Conference on Learning Representations*, 2024. |
| [40] | An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024. |
| [41] | Oriol Abril-Pla, Virgile Andreani, Colin Carroll, Larry Dong, Christopher J Fonnesbeck, Maxim Kochurov, Ravin Kumar, Junpeng Lao, Christian C Luhmann, Osvaldo A Martin, et al. PyMC: A modern, and comprehensive probabilistic programming framework in python. *PeerJ Computer Science*, 9:e1516, 2023. |
| [42] | Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, and Noah Goodman. Hypothesis search: Inductive reasoning with language models. In *International Conference on Learning Representations*, 2024. |
| [43] | Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, and Henry Pinkard. Arc-agi-2: A new challenge for frontier ai reasoning systems. arXiv preprint arXiv:2505.11831, 2025. |
| [44] | Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Technical report, Google DeepMind, jun 2025. URL https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf. Technical report, accessed 26 Jun 2025. |
| [45] | OpenAI. OpenAI o4-mini System Card. https://openai.com/index/o3-o4-mini-system-card/, apr 2025. Accessed 26 Jun 2025. |
| [46] | Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. |


12

---


# A Method Details

## A.1 Extended Preliminaries

### A.1.1 Problem Setup

First, we define the problem setting and introduce the mathematical notation.

We consider problems where the input is a natural language prompt t<sub>in</sub> (which may include few-shot examples, task instructions, etc.). Given t<sub>in</sub>, an LLM produces a natural language output t<sub>out</sub>, which is then scored by a score evaluator to yield a final score r. We assume 0 ≤ r ≤ 1, where higher values of r correspond to better answers.

This two-stage pipeline can be expressed as:

$$r = R(t_{out}) = R(f_{LLM}(t_{in})),\qquad(1)$$

where the function f<sub>LLM</sub> represents an LLM that generates the answer, and R is the score evaluator. f<sub>LLM</sub> is stochastic and may produce different outputs t<sub>out</sub> for the same input t<sub>in</sub>. Here we allow t<sub>out</sub> to include information beyond the direct answer (e.g., the reasoning steps), and we assume that R properly performs the parsing of this answer to perform the score evaluation.

Our framework applies to any task whose final answer can be quantitatively scored, such as programming tasks [1, 14], math problems [29], and machine learning competitions [16]. We assume the score evaluator R is already defined in a task-specific manner, and our goal is to find t<sub>out</sub> that attains as high an r as possible during the inference-time answer search.

In some tasks that mimic real competitions, the ground-truth score evaluator R<sub>gt</sub> (reflecting the correctness of the answer) is not accessible during the answer search stage. For example, in MLE-Bench [16], the test dataset used for final evaluation is withheld, and in coding competition tasks [1, 14], participants can often only submit their code a limited number of times to obtain the score on hidden test cases. In such situations, to search for the best answer, one may resort to a different score evaluator. For instance, in MLE-Bench, this could be the performance on a public dataset, while in coding competitions, it might be the fraction of solved public test cases. For mathematical tasks, a separately trained reward model [40] may be used. Throughout this work, we assume there is some accessible score evaluator at the answer search stage that can assess the quality of t<sub>out</sub>.

### A.1.2 Existing Methods for Inference-Time Answer Search

We now review two standard approaches that focus on exploration or exploitation alone.

**Only Go Wide: Repeated Sampling.** A straightforward approach to inference-time answer search is to repeatedly sample an answer from the LLM with a nonzero temperature. We refer to each sampling step as the direct answer generation process. By performing it n times,

$$t_{out}^m = f_{LLM}(t_{in}), \qquad m \in \{1, \ldots, n\},\qquad(2)$$

we obtain multiple candidate answers. Then, one can select the best answer based on a predefined criterion, such as the highest score r (best-of-n), majority voting, or self-consistency. Brown et al. [3] recently showed that as n increases, the coverage of generated answers improves. A similar approach was employed by AlphaCode [1] to achieve human-level performance on competitive programming tasks.

**Only Go Deep: Sequential Refinement.** Alternatively, we can leverage the answer refinement process and apply it sequentially to perform the answer search. We consider the situation where we already let some answer generator solve the problem at hand k times, and collected the input-output pairs t<sub>in</sub><sup>j</sup> and t<sub>out</sub><sup>j</sup> where j ∈ {1, . . . , k} for those answer generations.

We define the answer refinement process as a two-step procedure: (1) creation of a new refinement input from the existing input-output pairs, and (2) generation of a new answer t<sub>out</sub><sup>k+1</sup> from t<sub>in</sub><sup>k+1</sup>. Symbolically,

$$t_{out}^{k+1} = f_{LLM}(t_{in}^{k+1}) = f_{LLM}\left(h_{refine}\left(\{t_{in}^j, t_{out}^j\}_{j \in \{1,\ldots,k\}}\right)\right),\qquad(3)$$

where h<sub>refine</sub> is a refinement input generator that provides all the information necessary for refinement, such as feedback on each answer (e.g., code execution results or errors for coding tasks).


13

---


Applying this refinement step iteratively yields n answers:

$$t^1_{in} = t_{in},$$                                                        (4)

$$t^1_{out} = f_{LLM}(t^1_{in}),$$                                                 (5)

$$t^m_{in} = h_{refine}\left(\{t^j_{in}, t^j_{out}\}_{j∈\{1,...,m-1\}}\right)$$                           (6)

$$t^m_{out} = f_{LLM}(t^m_{in})$$                                                     (7)

for m ∈ {2, . . . , n}. Finally, one selects the best candidate among {t<sup>a</sup><sub>out</sub>}<sub>a∈{1,...,n}</sub> based on a chosen criterion, similar to the repeated sampling approach.

We can regard the two methods above (pure exploration and pure exploitation) as special cases of a tree search: the former expands only from the root node, while the latter continues from the most recently reached leaf node on a single linear path, exploring it in depth without branching outward. The standard MCTS naturally incorporates the two, while it uses a fixed branching factor, thereby limiting the performance gain from the repeated sampling when using LLM. Our AB-MCTS employs a more flexible branching algorithm and effectively leverages performance improvements obtained by repeated sampling.

## A.2 AB-MCTS-M Details

### A.2.1 Background on Mixed Models

Mixed linear models are extensions of traditional linear models that explicitly model non-independence among observations by incorporating both fixed and random effects. Fixed effects capture consistent, predictable patterns across the entire population or dataset (e.g., treatment effects common to all groups). In contrast, the random effects model variability arises within or between specific nested or hierarchical groups (e.g., individual differences among participants, or variations between schools).

### A.2.2 Detailed Mixed Model Formulation and Example Code

In AB-MCTS-M, we fit a separate mixed model at each node N of the MCTS tree, that is, at each sub-step within the MCTS selection step. Specifically, let N<sub>j</sub> (j = 1, . . . , n<sub>child</sub>) denote the direct child nodes of N, and define T<sub>sub</sub>(N<sub>j</sub>) as the subtree under N<sub>j</sub>, including N<sub>j</sub> itself. For a newly generated node Ñ where (i) j = 0 (i.e. for GEN node) and Ñ is a direct child of N, or (ii) j = 1, . . . , n<sub>child</sub> and Ñ is a node expanded from some node in T<sub>sub</sub>(N<sub>j</sub>) at that iteration step, we assume:

$$r_{\tilde{N}} = \alpha_j + \sigma_y\epsilon_{\tilde{N}}, \quad \alpha_j = \mu_\alpha + \sigma_\alpha\epsilon_j,$$                             (8)

$$\epsilon_{\tilde{N}} \sim \mathcal{N}(0, 1), \quad \epsilon_j \sim \mathcal{N}(0, 1),$$                               (9)

Here, α<sub>j</sub> is a "group-level" intercept capturing the quality of the base solution at N<sub>j</sub>, while σ<sub>y</sub>ε<sub>Ñ</sub> represents per-instance noise. The GEN node (action a<sub>0</sub>) is treated as a newly introduced group without its own direct observations. However, its group-level intercept α<sub>0</sub> is inferred not from the prior alone but rather from the posterior distribution over μ<sub>α</sub> and σ<sub>α</sub>, which is informed by the other observed data.

We place priors on (μ<sub>α</sub>, σ<sub>α</sub>, σ<sub>y</sub>) and estimate them via MCMC (Markov Chain Monte Carlo), then perform Thompson Sampling. Because the GEN group (j = 0) has no direct observations, its posterior remains more uncertain and thus encourages exploration. To illustrate how to estimate posterior predictives, in Listing 1, we provide PyMC [41] code corresponding to the example tree shown in Figure 2. Here, we adopted the same priors as in our experimental setting, as noted in Appendix B.2:

$$\mu_\alpha \sim \mathcal{N}(0.5, 0.2^2), \quad \sigma_\alpha \sim \mathcal{N}_{half}(0.2^2), \quad \sigma_y \sim \mathcal{N}_{half}(0.3^2),$$                (10)

In this implementation, the variable `alpha` is node-specific (indexed by `child_idx`), yet shares parameters `mu_alpha`, representing the overall average answer quality determined by the inherent difficulty of the task, and `sigma_alpha`, representing the variability in answer quality arising from the LLM's response diversity. Differences in answer quality among nodes are captured by the variable `eps_j`.


14

---


```python
import pymc as pm

# Child indices use 0-based indexing; note the difference from index j in
# Equations (8-9)
child_indices = [0, 0, 0, 1, 2, 2]
rewards = [0.8, 0.8, 1.0, 0, 0.2, 0.3]
coords = {"child_idx": [0, 1, 2]}

with pm.Model(coords=coords) as model:
    ## Priors
    mu_alpha = pm.Normal("mu_alpha", mu=0.5, sigma=0.2)
    
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.2)
    sigma_y = pm.HalfNormal("sigma_y", sigma=0.3)
    ## Priors END
    
    eps_j = pm.Normal("eps_j", mu=0, sigma=1, dims="child_idx")
    alpha = mu_alpha + eps_j * sigma_alpha
    
    r = pm.Normal("r", mu=alpha[child_indices], sigma=sigma_y, observed=
    rewards)
```
**Listing 1: AB-MCTS-M fitting model example code**

```python
eps_j_gen = pm.Normal("eps_j_gen", mu=0, sigma=1)
alpha_gen = mu_alpha + eps_j_gen * sigma_alpha
r_gen = pm.Normal("r_gen", mu=alpha_gen, sigma=sigma_y)
```
**Listing 2: AB-MCTS-M GEN node reward modeling**

To compute the probability distribution for the GEN node, we introduce a slightly modified predictive model by adding an additional variable representing the GEN node reward, as shown in Listing 2. Since eps_j_gen has no associated observed data, r_gen typically exhibits higher variance compared to r. Intuitively, r_gen incorporates both the variance arising from the refinement process and the inherent variability in answer generation at node N. This increased variance encourages greater exploration during Thompson Sampling.

After model fitting, the posterior predictive distributions of r (existing child nodes) and r_gen (GEN node) are utilized for Thompson Sampling.

## A.3 AB-MCTS-A Details: Parameter Update Rules

### A.3.1 AB-MCTS-A (Gaussian) Parameter Update Rules

As for the parameter update rules, for the Gaussian case, we use a normal-inverse-χ<sup>2</sup> prior:

$$p(r | \{r_n\}_{n=1}^N) = \mathcal{N}(\mu | \tilde{m}, \frac{\sigma^2}{\tilde{\kappa}})\chi^{-2}(\sigma^2 | \tilde{\nu}, \tilde{\tau}^2), \quad (11)$$

$$\tilde{m} = \frac{\hat{\kappa}\hat{m} + N\bar{r}}{\tilde{\kappa}}, \quad (12)$$

$$\tilde{\kappa} = \hat{\kappa} + N, \quad (13)$$

$$\tilde{\nu} = \hat{\nu} + N, \quad (14)$$

$$\tilde{\nu}\tilde{\tau}^2 = \hat{\nu}\hat{\tau}^2 + \sum_{n=1}^N (r_n - \bar{r})^2 + \frac{N\hat{\kappa}}{\hat{\kappa} + N}(\tilde{m} - \bar{r})^2, \quad (15)$$

$$\bar{r} = \frac{1}{N}\sum_{n=1}^N r_n, \quad (16)$$

where r<sub>n</sub> is the observed score.


15

---


## A.3.2 AB-MCTS-A (Beta) Parameter Update Rules

Alternatively, if r ∈ [0, 1], we can use a Beta distribution with the following parameter update rules after observing {r<sub>n</sub>}<sup>N</sup><sub>n=1</sub>:

$$p(r | \{r_n\}_{n=1}^N) = B(r | \hat{\alpha}, \hat{\beta}),$$
$$(17)$$

$$\hat{\alpha} = \tilde{\alpha} + \sum_{n=1}^N r_n,$$
$$(18)$$

$$\hat{\beta} = \tilde{\beta} + \sum_{n=1}^N (1 - r_n),$$
$$(19)$$

where B(· | α, β) denotes the Beta distribution. We note that, usually, this update rule is used in conjunction with the Bernoulli trial, but here we directly use Beta distribution to model the score distribution. In practice, this parameter update rule worked well according to our experimental results.

## B Additional Experimental Details

### B.1 Tasks and Datasets

We evaluated our approach on four benchmarks: CodeContest [1], LiveCodeBench [14], the Abstraction and Reasoning Corpus (ARC) [17], and MLE-Bench [16]. All of these benchmarks feature tasks that are often solved via code generation. Each experiment was run multiple times using non-zero temperature to account for stochasticity (n = 5 for LiveCodeBench, n = 3 for CodeContest and ARC-AGI). For MLE-Bench, experiments were run with n = 1 due to the significant computational cost.

**CodeContest and LiveCodeBench** are well-established competitive programming benchmarks, both providing public tests and hidden tests. We use the public tests to calculate each node's score and the hidden tests for final evaluation. A solution is counted as correct only if it passes all hidden test cases for a given problem, and the success rate is defined as the fraction of problems for which the chosen solution is fully correct. The prompt templates we used are based on those from previous work [14]. In LiveCodeBench, we only use problems released between August and November 2024, aligning with the previous work [19] to prevent data contamination.

**ARC-AGI** requires discovering a shared transformation rule from multiple input-output examples, then using it to predict the output for a test input. Generating code from sample grids is a frequently used approach for ARC [7, 13, 42]. We instruct the LLM to infer a transformation rule from the provided input/output examples and generate corresponding Python code. Each node's score is determined by the fraction of examples it correctly transforms. The node that achieves the highest score is then used to transform the test examples; if the output exactly matches the ground truth, that node's score is set to 1. We evaluate our method on the same set of 100 public evaluation problems and prompts used in prior work [13].

**MLE-Bench** comprises practical machine learning tasks derived from Kaggle competitions. In order to enable fair comparisons [16], we adopt three low-complexity challenges (*Nomad2018 Predicting Transparent Conductors*, *Spooky Author Identification*, and *Random Acts of Pizza*). Each competition's training data is randomly split into 80% for training and 20% for validation. The validation set is used to obtain the scores for each node. We select a node with the highest validation score at a given inference budget and then evaluate it on the hidden test set to get the final result. Following previous research [16], we use the AIDE scaffold for our experiments. Evaluating the generated machine learning models within these competitions is notably resource-intensive, requiring substantial GPU power even to process a single solution candidate. In our experiments, each solution candidate was executed on a single H100 GPU with a time limit of one hour. This computational demand is still considerably higher than for other benchmarks discussed in this work. Consequently, due to these significant costs, comprehensive experimentation across all methods and models on MLE-Bench was prohibitive. We therefore focused our evaluations using GPT-4o for these tasks specifically on AB-MCTS-M.


16

---

## B.2 AB-MCTS Parameters

Due to its Bayesian nature, the hyperparameters of AB-MCTS consist only of prior parameters. We used the same prior parameters for all the tasks without task-specific domain knowledge (except for the range of the score being approximately [0, 1]), thereby minimizing any potential bias. Consequently, our priors have the following shared properties:

• The vast majority of the probability mass (or all of it, in the case AB-MCTS-A (Beta)) is within [0, 1].

• The average value of score is 0.5, reflecting a neutral initial assumption regarding answer quality (where 0 indicates the worst and 1 the best)

• The probability mass does not concentrate excessively in any particular region, reflecting our unbiased prior.

We assign the following priors for AB-MCTS-M in Equations 8 and 9:

$$μ_α ∼ N(0.5, 0.2^2), \quad σ_α ∼ N_{half}(0.2^2), \quad σ_y ∼ N_{half}(0.3^2), \quad (20)$$

where N<sub>half</sub> is the half-normal distribution (please refer to Section 3.3 and Appendix A.2). For AB-MCTS-A (Gaussian), we set m̃ = 0, κ̃ = 1, ν̃ = 1, and τ̃<sup>2</sup> = 0.1 in Equations 11 - 16, and for AB-MCTS-A (Beta), we set α̃ = 0.5 and β̃ = 0.5 in Equations 17 - 19. As we noted earlier, we chose these parameters in a way that imposes as few assumptions as possible to minimize bias.

We expect the dependency on specific initial prior parameter values to be minimal, largely due to the substantial computational budgets in our evaluations and because posterior distributions become increasingly data-dominated as the search tree expands with more score observations (up to 2<sup>7</sup> nodes in most of the experiments and 2<sup>9</sup> in the extended ARC-AGI experiments). Because our node selection methods (AB-MCTS-M, with its mixed model, and AB-MCTS-A using conjugate priors) are fundamentally Bayesian, the priors primarily influence the early stages of the search. Moreover, the "borrowing strength" mechanism inherent in the mixed model stabilizes posterior estimates and facilitates convergence. As the search progresses and score observations accumulate, the initial prior influence naturally diminishes.

## C Additional Experiments and Analysis

<table>
<thead>
<tr>
<th>LiveCodeBench<br>(higher is better)</th>
<th>CodeContest<br>(higher is better)</th>
<th>ARC-AGI<br>(higher is better)</th>
</tr>
</thead>
<tbody>
<tr>
<td>
<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
    </tr>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th colspan="2" rowspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2" rowspan="2"></th>
    </tr>
    <tr>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
    </tr></table>
</td>
<td>
<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th colspan="2" rowspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th rowspan="2"></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr></table>
</td>
<td>
<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th colspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th colspan="2" rowspan="2"></th>
        <th></th>
        <th></th>
        <th></th>
        <th rowspan="2"></th>
    </tr>
    <tr>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
        <th rowspan="2"></th>
    </tr>
    <tr>
        <th></th>
    </tr>
    </thead>
    <tr>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>
</td>
</tr>
</tbody>
</table>

— Repeated Sampling — AB-MCTS-M — Sequential Refinement — AB-MCTS-A (Gaussian) — Standard MCTS — AB-MCTS-A (Beta)

**Figure 8:** Performance comparison with DeepSeek-V3 on LiveCodeBench, CodeContest, and ARC-AGI. We compare AB-MCTS methods with the baselines by plotting the success rate against the generation budget.

### C.1 Results with DeepSeek-V3 on Competitive Programming and ARC-AGI

This appendix provides supplementary results using the DeepSeek-V3 model, focusing on both overall performance and search behavior characteristics.


17

---


**Figure 9: Comparing algorithms by search tree shape and performance** Each point shows the performance against the average tree shape for a given algorithm at a specific generation budget. The x-axis represents the log-ratio of mean tree depth to mean tree width. Mean width is calculated as the average number of nodes per depth. Larger x-axis values indicate deeper searches, while smaller values indicate wider searches.

<table>
<thead>
<tr>
<th>Benchmark</th>
<th>Algorithm</th>
<th>Mean Depth / Mean Width (log2)</th>
<th>Success Rate (Pass@1)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="15">LiveCodeBench (higher is better)</td>
<td>Repeated Sampling</td>
<td>-4</td>
<td>0.38</td>
</tr>
<tr>
<td>Repeated Sampling</td>
<td>-3</td>
<td>0.39</td>
</tr>
<tr>
<td>Repeated Sampling</td>
<td>-2</td>
<td>0.40</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>0</td>
<td>0.41</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>1</td>
<td>0.42</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>2</td>
<td>0.43</td>
</tr>
<tr>
<td>AB-MCTS-M</td>
<td>-3</td>
<td>0.41</td>
</tr>
<tr>
<td>AB-MCTS-M</td>
<td>-2</td>
<td>0.42</td>
</tr>
<tr>
<td>AB-MCTS-M</td>
<td>-1</td>
<td>0.43</td>
</tr>
<tr>
<td>AB-MCTS-A (Gaussian)</td>
<td>-2</td>
<td>0.42</td>
</tr>
<tr>
<td>AB-MCTS-A (Gaussian)</td>
<td>-1</td>
<td>0.43</td>
</tr>
<tr>
<td>AB-MCTS-A (Gaussian)</td>
<td>0</td>
<td>0.44</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>-1</td>
<td>0.42</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>0</td>
<td>0.43</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>1</td>
<td>0.44</td>
</tr>
<tr>
<td rowspan="15">CodeContest (higher is better)</td>
<td>Repeated Sampling</td>
<td>-4</td>
<td>0.28</td>
</tr>
<tr>
<td>Repeated Sampling</td>
<td>-3</td>
<td>0.30</td>
</tr>
<tr>
<td>Repeated Sampling</td>
<td>-2</td>
<td>0.32</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>0</td>
<td>0.35</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>1</td>
<td>0.38</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>2</td>
<td>0.40</td>
</tr>
<tr>
<td>AB-MCTS-M</td>
<td>-3</td>
<td>0.35</td>
</tr>
<tr>
<td>AB-MCTS-M</td>
<td>-2</td>
<td>0.38</td>
</tr>
<tr>
<td>AB-MCTS-M</td>
<td>-1</td>
<td>0.40</td>
</tr>
<tr>
<td>AB-MCTS-A (Gaussian)</td>
<td>-2</td>
<td>0.36</td>
</tr>
<tr>
<td>AB-MCTS-A (Gaussian)</td>
<td>-1</td>
<td>0.39</td>
</tr>
<tr>
<td>AB-MCTS-A (Gaussian)</td>
<td>0</td>
<td>0.41</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>-1</td>
<td>0.37</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>0</td>
<td>0.39</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>1</td>
<td>0.40</td>
</tr>
<tr>
<td rowspan="12">ARC-AGI (higher is better)</td>
<td>Repeated Sampling</td>
<td>-4</td>
<td>0.03</td>
</tr>
<tr>
<td>Repeated Sampling</td>
<td>-3</td>
<td>0.06</td>
</tr>
<tr>
<td>Repeated Sampling</td>
<td>-2</td>
<td>0.09</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>0</td>
<td>0.12</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>1</td>
<td>0.15</td>
</tr>
<tr>
<td>Sequential Refinement</td>
<td>2</td>
<td>0.18</td>
</tr>
<tr>
<td>AB-MCTS-A (Beta)</td>
<td>-2</td>
<td>0.12</td>
</tr>
<tr>
<td>AB-MCTS-A (Beta)</td>
<td>-1</td>
<td>0.15</td>
</tr>
<tr>
<td>AB-MCTS-A (Beta)</td>
<td>0</td>
<td>0.18</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>-1</td>
<td>0.10</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>0</td>
<td>0.13</td>
</tr>
<tr>
<td>Standard MCTS</td>
<td>1</td>
<td>0.16</td>
</tr>
</tbody>
</table>

Figure 8 shows the performance (Pass@1) with DeepSeek-V3 on the Competitive Programming (LiveCodeBench, CodeContest) and ARC-AGI benchmarks. While the overall performance trends were similar to those with GPT-4o (Figure 4), the relative strengths of the baseline methods varied with DeepSeek-V3. For instance, standard MCTS achieved the highest success rate on LiveCodeBench. On CodeContest, the performance differences between AB-MCTS and the top-performing baselines were less pronounced than with GPT-4o. Despite these variations, AB-MCTS variants consistently placed among the leading methods across all tasks. This indicates that AB-MCTS reliably delivers strong performance even when changes in the underlying model alter the effectiveness of different baseline strategies.

The analysis of search tree shape versus performance with DeepSeek-V3 (Figure 9) closely mirrors the findings from GPT-4o (Figure 5). As expected, repeated sampling forms wide search trees, while sequential refinement develops deep trees. AB-MCTS methods consistently generated wider trees compared to standard MCTS. This tendency towards wider exploration is notable even on tasks like LiveCodeBench with DeepSeek-V3, where sequential refinement outperformed repeated sampling. The strong performance of AB-MCTS algorithms in such scenarios suggests their ability to not only explore broadly but also to effectively identify and deepen promising branches through their adaptive node selection. This highlights the robust nature of AB-MCTS in balancing these competing demands across different models.

**Figure 10: Performance comparison on three MLE-Bench tasks using GPT-4o.** Each plot shows performance versus the total generation budget. For Nomad2018 Predicting Transparent Conductors and Spooky Author Identification, lower scores are better (RMSLE and Log Loss, respectively); for Random Acts of Pizza, higher is better (ROC AUC). At each budget, we choose the single solution based on validation-set performance and report its test-set score.

<table>
<thead>
<tr>
<th>Task</th>
<th>Generation Budget</th>
<th>Sequential Refinement Score</th>
<th>Metric</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">Nomad2018 (lower is better)</td>
<td>2<sup>1</sup></td>
<td>0.075</td>
<td rowspan="7">RMSLE</td>
</tr>
<tr>
<td>2<sup>2</sup></td>
<td>0.070</td>
</tr>
<tr>
<td>2<sup>3</sup></td>
<td>0.065</td>
</tr>
<tr>
<td>2<sup>4</sup></td>
<td>0.060</td>
</tr>
<tr>
<td>2<sup>5</sup></td>
<td>0.058</td>
</tr>
<tr>
<td>2<sup>6</sup></td>
<td>0.057</td>
</tr>
<tr>
<td>2<sup>7</sup></td>
<td>0.055</td>
</tr>
<tr>
<td rowspan="7">Spooky Author Identification (lower is better)</td>
<td>2<sup>1</sup></td>
<td>0.55</td>
<td rowspan="7">Log Loss</td>
</tr>
<tr>
<td>2<sup>2</sup></td>
<td>0.50</td>
</tr>
<tr>
<td>2<sup>3</sup></td>
<td>0.48</td>
</tr>
<tr>
<td>2<sup>4</sup></td>
<td>0.45</td>
</tr>
<tr>
<td>2<sup>5</sup></td>
<td>0.42</td>
</tr>
<tr>
<td>2<sup>6</sup></td>
<td>0.40</td>
</tr>
<tr>
<td>2<sup>7</sup></td>
<td>0.35</td>
</tr>
<tr>
<td rowspan="7">Random Acts of Pizza (higher is better)</td>
<td>2<sup>1</sup></td>
<td>0.50</td>
<td rowspan="7">ROC AUC</td>
</tr>
<tr>
<td>2<sup>2</sup></td>
<td>0.55</td>
</tr>
<tr>
<td>2<sup>3</sup></td>
<td>0.60</td>
</tr>
<tr>
<td>2<sup>4</sup></td>
<td>0.65</td>
</tr>
<tr>
<td>2<sup>5</sup></td>
<td>0.68</td>
</tr>
<tr>
<td>2<sup>6</sup></td>
<td>0.70</td>
</tr>
<tr>
<td>2<sup>7</sup></td>
<td>0.75</td>
</tr>
</tbody>
</table>


18

<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    </thead>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    </thead>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

<table>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

<table>
    <thead>
    <tr>
        <th></th>
        <th>Repeated </th>
        <th>Sampling</th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    </thead>
    <tr>
        <td></td>
        <td>Sequential 
Standard 
AB-MCTS-M</td>
        <td>MCTS</td>
        <td>Refinement</td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    </thead>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>Repeated </td>
        <td>Sampling</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td>Standard 
AB-MCTS-M</td>
        <td>MCTS</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

<table>
    <thead>
    <tr>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
        <th></th>
    </tr>
    </thead>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td rowspan="2"></td>
    </tr>
    <tr>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
        <td rowspan="2">Repeated </td>
        <td rowspan="2">Sampling</td>
        <td rowspan="2"></td>
        <td rowspan="2"></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td>Standard 
AB-MCTS-M</td>
        <td>MCTS</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr></table>

---


## Figure 11: Example search trees generated by AB-MCTS-M and standard MCTS on MLE-Bench

### Nomad2018: AB-MCTS-M

**Root Node Tree Structure:**
- Root node connects to multiple levels of child nodes
- Nodes are numbered from 1 to 128
- Tree shows hierarchical branching structure with varying depths
- **Value Scale:** 0.032 - 0.039

### Nomad2018: Standard MCTS

**Root Node Tree Structure:**
- Root node with systematic branching pattern
- Nodes numbered sequentially with clear hierarchical levels
- More structured appearance compared to AB-MCTS-M version
- **Value Scale:** 0.032 - 0.039

### Spooky Author Identification: AB-MCTS-M

**Root Node Tree Structure:**
- Complex branching pattern from root node
- Nodes distributed across multiple levels
- Dense connectivity in middle sections
- **Value Scale:** 0.40 - 0.65

### Spooky Author Identification: Standard MCTS

**Root Node Tree Structure:**
- More linear, sequential node arrangement
- Clear hierarchical levels from root
- Systematic branching pattern
- **Value Scale:** 0.40 - 0.65

### Random Acts of Pizza: AB-MCTS-M

**Root Node Tree Structure:**
- Irregular branching pattern
- Nodes spread across varying depths
- Complex interconnections between levels
- **Value Scale:** 0.68 - 0.78

### Random Acts of Pizza: Standard MCTS

**Root Node Tree Structure:**
- Highly structured, grid-like arrangement
- Sequential node numbering in organized rows
- Clear hierarchical progression from root
- **Value Scale:** 0.68 - 0.78

**Caption:** The example tree for Random Acts of Pizza generated by AB-MCTS-M is the same tree shown in Figure 6


19

---


## C.2 Results with GPT-4o on three competitions from MLE-Bench

Figure 10 compares AB-MCTS-M against the baseline methods on the three MLE-Bench competitions using GPT-4o, plotting performance scores against the generation budget. Notably, the most effective baseline varies across these competitions. For instance, on Nomad2018, sequential refinement ultimately achieves the best score while repeated sampling shows no improvement. On Spooky Author Identification, standard MCTS exhibits continued improvement throughout the budget range. Conversely, on Random Acts of Pizza, repeated sampling significantly outperforms other baselines. Despite this variability in baseline effectiveness, AB-MCTS-M consistently delivers strong performance across all three competitions. This highlights the robustness and adaptability of AB-MCTS-M, suggesting its capability to effectively adjust its search approach to the differing characteristics of each task, especially when the optimal strategy is not apparent beforehand.

## C.3 Example search trees generated by each methods on MLE-Bench

Figure 11 presents example search trees generated by AB-MCTS-M and standard MCTS for the three MLE-Bench competitions. Across all tasks, the trees generated by AB-MCTS-M visually suggest a more flexible approach compared to standard MCTS, effectively combining broader exploration with focused exploitation of promising nodes.

# D Multi-LLM AB-MCTS

Depending on the task, using more than one LLM for answer search can be advantageous. For example, if LLM A can generate more diverse initial answers and LLM B excels at refinement, building the answer tree with both models is expected to improve performance. This section demonstrates how AB-MCTS can be extended to scenarios in which multiple LLMs are available for answer generation, and also reports the experimental results for ARC-AGI-2, demonstrating the effectiveness of the proposed method.

## D.1 Method

### D.1.1 Multiple LLMs as Answer Generators

Suppose L LLMs are available for answer generation. We denote by f<sub>l</sub><sup>LLM</sup> the answer generator implemented by the l-th LLM, with l = 1, . . . , L, following the notation introduced in Equation 1. The overall procedure is identical to the single-LLM AB-MCTS (Algorithm 1), with the addition of a step to select one of the L available generators for node expansion. This selection occurs at each expansion phase, utilizing the current state of the answer tree T. The chosen generator, f<sub>l</sub><sup>LLM</sup>, is then used to expand the selected node. We introduce two distinct algorithms for this generator selection process.

### D.1.2 Generator Selection Algorithm I: Single GEN Node

In this algorithm, node selection proceeds identically to AB-MCTS, followed by an additional generator selection step. During this step, each node N in the entire answer tree T is annotated with the index l of the generator that produced it. For every generator, we obtain the set of nodes N<sub>l</sub> ⊆ T, containing all nodes it has generated; N<sub>l</sub> is empty if the generator has not been selected yet. The scores of the nodes in N<sub>l</sub> are used to calculate the posterior distribution of the expected score of a new node that will be generated by the generator l. After all L posteriors have been computed, Thompson sampling is applied: a single sample is drawn from each distribution, and the generator with the highest sample is selected for the node expansion. These posteriors are modeled in the same way as the distributions used for node expansion target selection, as detailed below.

**Multi-LLM AB-MCTS-M with Generator Selection Algorithm I.** In this algorithm, Multi-LLM AB-MCTS-M treats the generators as groups within the mixed-effects model defined as

$$r_{\tilde{N}} = \alpha_l + \sigma_y \epsilon_{\tilde{N}}, \quad \alpha_l = \mu_\alpha + \sigma_\alpha \epsilon_l,$$ (21)

$$\epsilon_{\tilde{N}} \sim \mathcal{N}(0, 1), \quad \epsilon_l \sim \mathcal{N}(0, 1),$$ (22)


20

---


where the group index j has been replaced by the generator index l from Equations 8–9. As for the prior distributions, we can use the ones in Equation 10. After calculating the score posterior distributions using N<sub>l</sub> for all l, we perform Thompson sampling to select one generator.

**Multi-LLM AB-MCTS-A with Generator Selection Algorithm I.** In this algorithm, Multi-LLM AB-MCTS-A assigns each l-th generator an independent prior–Gaussian (Equations 11-16) or Beta (Equations 17-19)–depending on the score metric, and updates these priors to posteriors using the scores of the nodes in N<sub>l</sub>. After calculating the posteriors for all l, we perform Thompson sampling to choose a generator.

## D.1.3 Generator Selection Algorithm II: Multiple GEN Nodes

An alternative approach attaches multiple GEN nodes as children to every node in the tree–one for each of the L available generators. At each node N in the expansion target selection process, the following process occurs:

1. The AB-MCTS selection logic is applied independently within the sub-trees associated with each generator. This means that for each generator l, we run a selection process considering its associated GEN node and any child nodes previously generated by it from node N.

2. Thompson sampling is used to identify the best node (either the GEN node or an existing child node) for each generator l.

3. Finally, the scores of the L best nodes selected in the previous step are compared, and the one with the highest overall score is selected.

This algorithm is designed to better capture the local context of the search tree, allowing for the adaptive selection of the most suitable generator at each specific stage of the solution process.

## D.2 Experiments

In this section, we evaluate the effectiveness of Multi-LLM AB-MCTS by reporting the results and analysis of ARC-AGI-2 [43]. ARC-AGI-2 is an enhanced benchmark building upon the original ARC-AGI, specifically designed to assess higher-level cognitive abilities of artificial intelligence systems rigorously. It maintains the same fundamental principles as ARC-AGI, emphasizing tasks that require general fluid intelligence rather than extensive prior knowledge or memorization. We selected this benchmark, which is hard to solve even for frontier LLMs (less than 5% success rate [43]), to assess whether we can combine multiple frontier reasoning LLMs to obtain better performance for challenging tasks.

### D.2.1 Experimental Setup

In this experiment, we evaluated our approach on the 120 problems comprising the public evaluation set of ARC-AGI-2. For each problem, the generation budget was set to 250. The solution generation and refinement procedures are the same as those of ARC-AGI-1 experiment: The models were instructed to generate the transformation rule as Python code, and the search was guided by a reward signal corresponding to the number of demonstration cases correctly solved by the generated code. For solution generation, we used three frontier reasoning models: Gemini-2.5 Pro (`gemini-2.5-pro-preview-05-06`) [44], o4-mini (`o4-mini-2025-04-16`) [45], and DeepSeek-R1-0528 (`deepseek-r1-0528`) [46]. We set the temperature to 0.6 for all models.

To primarily evaluate the potential of Multi-LLM AB-MCTS, we used the Pass@k metric. This metric measures whether at least one correct solution is found within k attempts. This differs from the official ARC-AGI-2 contest standard, which typically uses a Pass@2 criterion (i.e., one of two submitted final answers must be correct). Evaluating Pass@2 requires an additional selection mechanism to identify promising candidates from the search history. Therefore, this experiment focuses on the search capability itself via Pass@k. Due to the significant API costs of the used frontier reasoning models, we focused on the evaluation of Single-LLM AB-MCTS-A and Multi-LLM AB-MCTS-A (and repeated sampling for o4-mini), since it worked most efficiently in our experiment on ARC-AGI-1. Also, we employed the generator selection algorithm II (see Section D.1.3 for details) for our experiments.


21

---


## Performance of AB-MCTS vs Multi-LLM AB-MCTS

<table>
<thead>
<tr>
<th>Number of LLM Calls (k)</th>
<th>o4-mini + Gemini-2.5-Pro + R1-0528</th>
<th>o4-mini + Gemini-2.5-Pro</th>
<th>o4-mini</th>
<th>o4-mini (Repeated Sampling)</th>
<th>Gemini-2.5-Pro</th>
<th>DeepSeek-R1-0528</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>0.00</td>
<td>0.00</td>
<td>0.00</td>
<td>0.00</td>
<td>0.00</td>
<td>0.00</td>
</tr>
<tr>
<td>50</td>
<td>0.28</td>
<td>0.26</td>
<td>0.24</td>
<td>0.20</td>
<td>0.15</td>
<td>0.08</td>
</tr>
<tr>
<td>100</td>
<td>0.30</td>
<td>0.28</td>
<td>0.26</td>
<td>0.22</td>
<td>0.16</td>
<td>0.09</td>
</tr>
<tr>
<td>150</td>
<td>0.31</td>
<td>0.29</td>
<td>0.27</td>
<td>0.23</td>
<td>0.16</td>
<td>0.09</td>
</tr>
<tr>
<td>200</td>
<td>0.32</td>
<td>0.30</td>
<td>0.275</td>
<td>0.23</td>
<td>0.16</td>
<td>0.09</td>
</tr>
<tr>
<td>250</td>
<td>0.32</td>
<td>0.30</td>
<td>0.275</td>
<td>0.23</td>
<td>0.16</td>
<td>0.09</td>
</tr>
</tbody>
</table>

**Figure 12:** Pass@k (coverage) of ARC-AGI-2 for each generation budget. The methods tested are repeated sampling, AB-MCTS, and Multi-LLM AB-MCTS methods.

<table>
<thead>
<tr>
<th>Problem Ranking (1st is highest reward)</th>
<th>o4-mini Usage</th>
<th>Gemini-2.5-Pro Usage</th>
<th>R1-0528 Usage</th>
<th>Solved Status</th>
</tr>
</thead>
<tbody>
<tr>
<td>1-20</td>
<td>Low</td>
<td>High</td>
<td>Medium</td>
<td>Most Solved (★)</td>
</tr>
<tr>
<td>21-40</td>
<td>Medium</td>
<td>High</td>
<td>Low</td>
<td>Many Solved (★)</td>
</tr>
<tr>
<td>41-60</td>
<td>High</td>
<td>Medium</td>
<td>Low</td>
<td>Some Solved (★)</td>
</tr>
<tr>
<td>61-80</td>
<td>High</td>
<td>Low</td>
<td>Medium</td>
<td>Few Solved (★)</td>
</tr>
<tr>
<td>81-100</td>
<td>High</td>
<td>Low</td>
<td>Medium</td>
<td>Few Solved (★)</td>
</tr>
<tr>
<td>101-120</td>
<td>High</td>
<td>Low</td>
<td>Low</td>
<td>Rarely Solved</td>
</tr>
</tbody>
</table>

**Figure 13:** Distribution of LLM usage in the 120 ARC-AGI-2 problems with Multi-LLM AB-MCTS-A. Trials are sorted by the maximum reward obtained from demonstration cases (higher reward to the left). Starred trials indicate a correct final solution was found.

## D.2.2 Results

The performance of our proposed methods was compared against a repeated sampling, which was the most efficient method for the ARC-AGI-1 experiment, and the result is shown in Fig. 12. As shown, Repeated Sampling using the o4-mini model achieved a 23% Pass@k success rate on the public evaluation set, and the single-model AB-MCTS using o4-mini improved the success rate to 27.5%. The performance advantage of AB-MCTS over Repeated Sampling becomes more evident as the generation budget increases, particularly after approximately 50 budget.

By employing Multi-LLM AB-MCTS, which integrates Gemini-2.5-Pro and DeepSeek-R1-0528, we further improved the performance, ultimately finding correct solutions for over 30% of the problems. Notably, although DeepSeek-R1-0528 exhibited lower individual performance, its integration into the Multi-LLM framework led to an increase in the number of solved problems.

## D.2.3 Analysis

The Multi-LLM AB-MCTS framework demonstrated an ability to effectively allocate different LLMs to problems based on their characteristics. As illustrated in Figure 13, the distribution of LLM usage varied across problems. For trials that quickly achieved a high reward from the demonstration cases (left side of the figure), the more proficient model tended to be assigned. Conversely, for trials


22

---


## Example Search Tree by Multi-LLM AB-MCTS

The search tree diagram shows a hierarchical structure with numbered nodes representing generation steps. The nodes are color-coded according to different LLMs:
- Green: o4-mini
- Red: Gemini-2.5-Pro  
- Orange: R1-0528
- Yellow: Solved

The tree starts with a root node and branches out through multiple levels, with various numbered nodes (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49) distributed across different paths. One particular node is highlighted in yellow, indicating the solution node.

**Figure 14:** An example search tree from a successful trial on ARC-AGI-2 using Multi-LLM AB-MCTS-A. The number in each node indicates the generation step, and the color represents the selected LLM. The yellow node generated the code that correctly solved the test case. This problem was not solved by any single model in isolation.

where obtaining a high reward was more challenging (right side), the models were utilized in a more balanced manner.

Furthermore, we observed instances where problems unsolvable by any single LLM were solved through the collaboration of multiple models. This suggests a synergistic interaction that transcends simply matching the best model to a problem. Figure 14 and Figure 15 depict a search process where an incorrect solution generated by o4-mini served as a useful hint for DeepSeek-R1-0528 and Gemini-2.5-Pro, which then collaboratively produced the correct solution. This result indicates that Multi-LLM AB-MCTS can facilitate flexible and effective collaboration among heterogeneous frontier LLMs.

### D.3 Challenges and Future Work

While our primary evaluation focused on search capability using the Pass@k metric, we conducted a preliminary evaluation based on the Pass@2 criterion for reference. Using a simple rule-based method to select two final answers (prioritizing code with a high reward generated later in the search), the Multi-LLM AB-MCTS achieved a Pass@2 of 19.2%. Although this is a promising result, a significant gap of over 10 percentage points remains compared to the 30% Pass@250 rate. Future work should focus on closing this gap by developing more sophisticated final-answer selection algorithms. Potential directions include building more accurate reward models or integrating an LLM-as-a-Judge for a more nuanced evaluation of candidate solutions.


23

---

Problem: 36a08778

Training set                                Test set

[Grid patterns with colored cells]          [Grid pattern with colored cells]
                                           
                                           ?

Analysis of the previous solution by DeepSeek-R1-0528

After analyzing the discrepancies between the current implementation and the expected outputs, I identified two main issues:

1. **Missing Cell Coloring During Horizontal Spread**: When the flood function encountered a gap (a cell below with value 7), it broke the spread without coloring the current cell. This caused cells that should become water (6) to remain empty (7).

2. **Global Visited Sets Causing Skipped Processing**: The visited sets were shared across all seed processing, preventing multiple seeds from properly interacting with cells processed by earlier seeds. This disrupted the natural water flow simulation.

The solution involves:
1. Adding cell coloring at gap locations during horizontal spread before initiating a drop
2. Removing the global visited sets entirely since grid state changes naturally prevent reprocessing
3. Maintaining recursion limits for deep but finite grids

Solution generated by o4-mini                    New solution refined by DeepSeek-R1-0528

```python
def transform(grid):
    H = len(grid)
    W = len(grid[0])
    # We will pour water (6) from every existing 6-seed in the input,
    # treating 2's as solid obstacles, 7's as empty space.
    # Any 7 we visit becomes 6
    # We'll do two mutually-recursive routines: drop() and flood()
    import sys
    sys.setrecursionlimit(10000)
    
    visited_drop = set()  # to avoid infinite recursion in drop
    visited_flood = set()  # to avoid infinite recursion in flood
    
    def drop(r, c):
        # (r, c) is a water source; let it fall straight down
        # until the cell below is red (2) or we fall off the bottom.
        # Color any 7's we pass into 6.
        if (r, c) in visited_drop:
            return
        visited_drop.add((r, c))
        rr = r
        # Fall until we hit either off-grid below or red pixel below
        while True:
            if rr + 1 >= H:
                # bottom of grid: cannot flood here
                break
            if grid[rr + 1][c] == 2:
                # we're sitting on top of red pool/flood
                break
            # otherwise we can move down one
            rr += 1
            if grid[rr][c] != 7:
                grid[rr][c] = 6
            # continue falling
        # if we stopped because of red right beneath, do horizontal flood
        if rr + 1 < H and grid[rr + 1][c] == 2:
            flood(rr, c)
    
    def flood(r, c):
        # at (r, c) we are on top of red block;
        # spread left and right, coloring any 7's we see,
        # but if we find a spot whose below-cell is 7,
        # we spill there via drop()
        if (r, c) in visited_flood:
            return
        visited_flood.add((r, c))
        # Spread left
        x = c
        while True:
            if x - 1 < 0:
                break
            if grid[r][x - 1] != 7:
                break
            # if the cell below that spot is empty (7), we spill there
            if r + 1 < H and grid[r + 1][x - 1] == 7:
                drop(r, x - 1)
                break
            # otherwise we can safely color it
            grid[r][x - 1] = 6
            x -= 1
        # Spread right
        x = c
        while True:
            if x + 1 >= W:
                break
            if grid[r][x + 1] != 7:
                break
            if r + 1 < H and grid[r + 1][x + 1] == 7:
                drop(r, x + 1)
                break
            grid[r][x + 1] = 6
            x += 1
    
    # Find all the 6-seeds in the original input
    seeds = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 6:
                seeds.append((i, j))
    # Pour from each seed
    for (sr, sc) in seeds:
        drop(sr, sc)
    return grid
```

