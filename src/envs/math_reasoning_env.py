# src/envs/math_reasoning_env.py

import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List

from src.utils.verifier import Verifier
from src.utils.prompt_constructor import PromptConstructor

class MathReasoningEnv(gym.Env):
    """
    [V-FINAL] 将数学问题解决过程封装成一个RL环境。
    reset方法现在会返回更丰富的元数据，包括pass_rate。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, problem_set: List[Dict], verifier: Verifier, prompt_constructor: PromptConstructor,
                 max_steps: int):
        super().__init__()
        if not problem_set:
            raise ValueError("Problem set cannot be empty.")

        self.problem_set = problem_set
        self.verifier = verifier
        self.prompt_constructor = prompt_constructor
        self.max_steps = max_steps

        self.action_space = spaces.Text(max_length=2048, charset='utf-8')
        self.observation_space = spaces.Text(max_length=8192, charset='utf-8')

        self._current_problem: Optional[Dict] = None
        self._current_state: str = ""
        self._current_step: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        重置环境，开始一个新的问题。
        返回 (初始观察, 信息字典)。
        """
        super().reset(seed=seed)

        problem_idx = self.np_random.integers(0, len(self.problem_set))
        self._current_problem = self.problem_set[problem_idx]

        self._current_state = self.prompt_constructor.get_rl_rollout_prompt(self._current_problem['problem'])
        self._current_step = 0

        # --- info字典现在包含所有RL Agent需要的元数据
        info = {
            "problem_id": problem_idx,
            "problem_text": self._current_problem['problem'],
            "ground_truth_answer": self._current_problem['final_answer'],
            "pass_rate": self._current_problem.get("pass_rate", 0.5) # OREAL奖励重塑需要，提供默认值
        }
        return self._current_state, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        执行一个动作（生成一段文本）。
        返回 (观察, 奖励, 是否正常终止, 是否截断, 信息字典)。
        """
        # 将生成的文本追加到当前状态
        self._current_state += action
        self._current_step += 1

        # 检查终止条件
        terminated = self.prompt_constructor.tokenizer.eos_token in action or "\\boxed{" in action
        truncated = self._current_step >= self.max_steps

        # 奖励由外部的Agent根据最终结果计算，环境本身不提供奖励
        reward = 0.0

        # 如果回合结束，进行验证并返回结果
        info = {}
        if terminated or truncated:
            # 剥离prompt，只验证答案部分
            prompt_for_eval = self.prompt_constructor.get_rl_rollout_prompt(self._current_problem['problem'])
            solution_only = self._current_state[len(prompt_for_eval):].strip()

            is_correct = self.verifier.verify(
                solution_text=solution_only,
                ground_truth=self._current_problem['final_answer'],
                question=self._current_problem['problem'],
                use_llm_judger=False # 评估时禁用LLM Judger以保证一致性
            )
            info['is_correct'] = is_correct
            info['final_state'] = self._current_state

        return self._current_state, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print("-" * 50)
            print(f"Current Step: {self._current_step}")
            print("Current State:")
            print(self._current_state)
            print("-" * 50)