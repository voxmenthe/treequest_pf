# the system prompt for step-by-step reasoning taken from https://github.com/huggingface/search-and-learn
SAL_STEP_BY_STEP_SYSTEM_PROMPT = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."

QWEN_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
