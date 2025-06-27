# src/utils/verifier.py

import os
import time
from typing import Tuple, Optional
from omegaconf import DictConfig
from openai import OpenAI, APIConnectionError, RateLimitError

# 保持相对导入，这在我们的项目结构中是正确的
from .math_equivalence import is_equiv, extract_answer


class Verifier:
    """
    [V-FINAL-3] 集成了DeepSeek API并修复了所有已知警告的混合型答案验证器。
    """

    def __init__(self, config: DictConfig):
        self.verifier_config = config.evaluation.get("verifier_config", {})
        self.use_llm_judger_in_rl = self.verifier_config.get("use_llm_judger_in_rl", False)

        self.client = None
        if self.use_llm_judger_in_rl:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                print("WARNING: DEEPSEEK_API_KEY environment variable not set. LLM judger will be disabled.")
                self.use_llm_judger_in_rl = False
            else:
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
                self.model_name = "deepseek-chat"
                print(
                    f"✅ Verifier initialized with DeepSeek LLM-based judger (model: {self.model_name}) ENABLED for RL training.")

        if not self.use_llm_judger_in_rl:
            print("✅ Verifier initialized in rule-based-only mode.")

        self.verify_prompt_template = """You are a helpful assistant who evaluates the correctness and quality of models' outputs. Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. Here are some evaluation criteria: 1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct. 2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question. 3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct. 4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct. 5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer. Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of: A: CORRECT B: INCORRECT Just return the letters \"A\" or \"B\", with no text around it. Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer. <Original Question Begin>: {question} <Original Question End> <Gold Target Begin>: {gold_answer} <Gold Target End> <Predicted Answer Begin>: {answer} <Predicted End> Judging the correctness of candidates' answers:"""

    def verify(self, solution_text: str, ground_truth: str, **kwargs) -> bool:
        """
        核心验证方法。

        Args:
            solution_text (str): 模型生成的完整文本。
            ground_truth (str): 标准答案。
            **kwargs: 可选参数, 如 'question', 'is_training_rl' 和 'use_llm_judger' (bool)。
        """
        use_llm = kwargs.get('use_llm_judger', False)

        if kwargs.get("is_training_rl", False):
            use_llm = self.use_llm_judger_in_rl

        if (extracted_answer := extract_answer(solution_text)) and is_equiv(extracted_answer, ground_truth):
            return True

        if use_llm and self.client:
            _, model_verified = self._evaluate_with_llm(
                question=kwargs.get('question', ''), answer=solution_text, gold_answer=ground_truth
            )
            return model_verified if model_verified is not None else False

        return False

    def _evaluate_with_llm(self, question: str, answer: str, gold_answer: str) -> Tuple[str, Optional[bool]]:
        """使用openai包调用DeepSeek API进行验证"""
        prompt = self.verify_prompt_template.format(question=question, answer=answer, gold_answer=gold_answer)

        for _ in range(self.verifier_config.get("max_retries", 2)):
            res = None
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )
                res_string = response.choices[0].message.content.strip()

                if res_string == "A": return res_string, True
                if res_string == "B": return res_string, False

                if "A" in res_string and "B" not in res_string: return res_string, True
                if "B" in res_string and "A" not in res_string: return res_string, False

                print(f"Warning: DeepSeek Verifier returned ambiguous result: '{res_string}'")
                return res_string, None
            except (APIConnectionError, RateLimitError) as e:
                print(f"Error calling DeepSeek API: {e}")
                time.sleep(4)
            except (KeyError, IndexError, AttributeError) as e:
                response_text = res.text if res and hasattr(res, 'text') else "N/A"
                print(f"Error parsing response from DeepSeek API: {e}. Response: {response_text}")
                return "Parsing Error", None

        return "Failed after retries", None