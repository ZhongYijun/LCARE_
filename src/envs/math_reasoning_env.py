# # src/envs/math_reasoning_env.py

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# from typing import Dict, Any, Tuple, Optional, List
# from omegaconf import DictConfig

# from src.utils.verifier import Verifier
# from src.utils.prompt_constructor import PromptConstructor

# class MathReasoningEnv(gym.Env):
#     """
#     [V-FINAL] 将数学问题解决过程封装成一个RL环境。
#     - __init__ 现在接收主配置以访问verifier开关。
#     - step方法现在能正确地传递use_llm_judger参数。
#     """
#     metadata = {'render_modes': ['human']}

#     def __init__(self, problem_set: List[Dict], verifier: Verifier, prompt_constructor: PromptConstructor,
#                  max_steps: int, config: DictConfig): # [MODIFIED] 添加config参数
#         super().__init__()
#         if not problem_set:
#             raise ValueError("Problem set cannot be empty.")

#         self.problem_set = problem_set
#         self.verifier = verifier
#         self.prompt_constructor = prompt_constructor
#         self.max_steps = max_steps
#         self.config = config # [NEW] 存储主配置

#         self.action_space = spaces.Text(max_length=2048, charset='utf-8')
#         self.observation_space = spaces.Text(max_length=8192, charset='utf-8')

#         self._current_problem: Optional[Dict] = None
#         self._initial_prompt: str = ""
#         self._current_state: str = ""
#         self._current_step: int = 0

#     def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[str, Dict]:
#         super().reset(seed=seed)
#         problem_idx = self.np_random.integers(0, len(self.problem_set))
#         self._current_problem = self.problem_set[problem_idx]
        
#         self._initial_prompt = self.prompt_constructor.get_rl_rollout_prompt(self._current_problem['problem'])
#         self._current_state = self._initial_prompt
#         self._current_step = 0

#         info = {
#             "problem_id": problem_idx,
#             "problem_text": self._current_problem['problem'],
#             "ground_truth_answer": self._current_problem['final_answer'],
#             "pass_rate": self._current_problem.get("pass_rate", 0.5)
#         }
#         return self._current_state, info

#     def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
#         self._current_state += action
#         self._current_step += 1

#         terminated = self.prompt_constructor.tokenizer.eos_token in action or "\\boxed{" in action
#         truncated = self._current_step >= self.max_steps

#         reward = 0.0
#         info = {}
#         if terminated or truncated:
#             solution_only = self._current_state[len(self._initial_prompt):].strip()
            
#             # [CRITICAL FIX] 从配置中读取开关，并传递给verify函数
#             use_judger = self.config.trainer.verifier.get("use_llm_judger_in_rl", False)
            
#             is_correct = self.verifier.verify(
#                 solution_text=solution_only,
#                 ground_truth=self._current_problem['final_answer'],
#                 question=self._current_problem['problem'],
#                 use_llm_judger=use_judger # 传递开关
#             )
#             info['is_correct'] = is_correct
#             info['final_state'] = self._current_state

#         return self._current_state, reward, terminated, truncated, info

# src/envs/math_reasoning_env.py (修复后)

import gymnasium as gym
from gymnasium import spaces

from typing import Dict, Any, Tuple, Optional, List
from omegaconf import DictConfig

from src.utils.verifier import Verifier
from src.utils.prompt_constructor import PromptConstructor


class MathReasoningEnv(gym.Env):
    """
    [V-FINAL - KEY-FIXED] 将数学问题解决过程封装成一个RL环境。
    - 修复了由于字典键名不匹配导致的KeyError。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, problem_set: List[Dict], verifier: Verifier, prompt_constructor: PromptConstructor,
                 max_steps: int, config: DictConfig):
        super().__init__()
        if not problem_set:
            raise ValueError("Problem set cannot be empty.")

        self.problem_set = problem_set
        self.verifier = verifier
        self.prompt_constructor = prompt_constructor
        self.max_steps = max_steps
        self.config = config

        self.action_space = spaces.Text(max_length=2048, charset='utf-8')
        self.observation_space = spaces.Text(max_length=8192, charset='utf-8')

        self._current_problem: Optional[Dict] = None
        self._initial_prompt: str = ""
        self._current_state: str = ""
        self._current_step: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[str, Dict]:
        super().reset(seed=seed)
        problem_idx = self.np_random.integers(0, len(self.problem_set))
        self._current_problem = self.problem_set[problem_idx]

        # [FIX] 使用正确的键 'problem_text'
        self._initial_prompt = self.prompt_constructor.get_rl_rollout_prompt(self._current_problem['problem_text'])
        self._current_state = self._initial_prompt
        self._current_step = 0

        # [FIX] 使用正确的键来构建info字典
        info = {
            "problem_id": problem_idx,
            "problem_text": self._current_problem['problem_text'],
            "ground_truth_answer": self._current_problem['ground_truth_answer'],
            "pass_rate": self._current_problem.get("pass_rate", 0.5)
        }
        return self._current_state, info

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        self._current_state += action
        self._current_step += 1

        terminated = self.prompt_constructor.tokenizer.eos_token in action or "\\boxed{" in action
        truncated = self._current_step >= self.max_steps

        reward = 0.0
        info = {}
        if terminated or truncated:
            solution_only = self._current_state[len(self._initial_prompt):].strip()

            use_judger = self.config.trainer.verifier.get("use_llm_judger_in_rl", False)

            # [FIX] 使用正确的键 'ground_truth_answer' 和 'problem_text'
            is_correct = self.verifier.verify(
                solution_text=solution_only,
                ground_truth=self._current_problem['ground_truth_answer'],
                question=self._current_problem['problem_text'],
                use_llm_judger=use_judger
            )
            info['is_correct'] = is_correct
            info['final_state'] = self._current_state

        return self._current_state, reward, terminated, truncated, info