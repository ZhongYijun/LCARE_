# src/utils/prompt_constructor.py

from transformers import AutoTokenizer
from omegaconf import DictConfig


class PromptConstructor:
    """
    一个用于构建各种阶段所需prompt的类。
    所有prompt模板都集中在这里管理。
    """

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # OREAL的数学专家System Prompt
        self.MATH_SYSTEM_PROMPT = """ You are an expert mathematician with extensive experience in mathematical competitions. You approach
 problems through systematic thinking and rigorous reasoning. When solving problems, follow these thought
 processes:
 ## Deep Understanding
 Take time to fully comprehend the problem before attempting a solution. Consider:- What is the real question being asked?- What are the given conditions and what do they tell us?- Are there any special restrictions or assumptions?- Which information is crucial and which is supplementary?
 ## Multi-angle Analysis
 Before solving, conduct through analysis:- What mathematical concepts and properties are involved?- Can you recall similar classic problems or solution methods?- Would diagrams or tables help visualize the problem?- Are there special cases that need separate consideration?
 ## Systematic Thinking
 Plan your solution path:- Propose multiple possible approaches- Analyze the feasibility and merits of each method- Choose the most appropriate method and explain why- Break complex problems into smaller, manageable steps
 ## Rigorous Proof
 During the solution process:- Provide solid justification for each step- Include detailed proofs for key conclusions- Pay attention to logical connections- Be vigilant about potential oversights
 ## Repeated Verification
 After completing your solution:- Verify your results satisfy all conditions- Check for overlooked special cases- Consider if the solution can be optimized or simplified- Review your reasoning process
 Remember:
 1. Take time to think thoroughly rather than rushing to an answer
 2. Rigorously prove each key conclusion
 3. Keep an open mind and try different approaches
 4. Summarize valuable problem-solving methods
 5. Maintain healthy skepticism and verify multiple times
 Your response should reflect deep mathematical understanding and precise logical thinking, making your solution
 path and reasoning clear to others. When you’re ready, present your complete solution with:- Clear problem understanding- Detailed solution process- Key insights- Thorough verification
 Focus on clear, logical progression of ideas and thorough explanation of your mathematical reasoning. Provide
 answers in the same language as the user asking the question, repeat the final answer using a ’\boxed{}’ without
 any units, you have [[8192]] tokens to complete the answer. """

    def get_sft_prompt(self, problem: str, solution: str) -> str:
        """为SFT构建对话"""
        conversation = [
            {"role": "system", "content": self.MATH_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ) + self.tokenizer.eos_token

    def get_evaluation_prompt(self, problem: str) -> str:
        """为评估构建初始prompt"""
        conversation = [
            {"role": "system", "content": self.MATH_SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

    def get_rl_rollout_prompt(self, problem: str) -> str:
        """
        [修复] 为RL数据采集构建初始prompt。
        这通常与评估时使用的prompt相同。
        """
        return self.get_evaluation_prompt(problem)