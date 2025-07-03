# # src/utils/verifier.py
#
# import os
# import time
# import requests
# import json
# from typing import Tuple, Optional
# from omegaconf import DictConfig
#
# from .math_equivalence import is_equiv, extract_answer
#
# class Verifier:
#     """
#     [FINAL & ROBUST] 混合型答案验证器。
#     """
#     def __init__(self, config: DictConfig):
#
#         # 【核心修复】直接从传入的 verifier_config 中读取开关
#         verifier_config = config.get("verifier", {})
#
#         self.model_verifier_hosts = verifier_config.get("hosts", [])
#         self.api_key = os.environ.get("DEEPSEEK_API_KEY") or verifier_config.get("api_key", "")
#         self.max_retries = verifier_config.get("max_retries", 3)
#         self.retry_delay = verifier_config.get("retry_delay", 2.0)
#         self.use_llm_judger = verifier_config.get("use_llm_judger", False)
#
#         # [CRITICAL FIX] 恢复被意外删除的prompt模板定义
#         self.verify_prompt_template = """You are a helpful assistant who evaluates the correctness and quality of models' outputs. Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. Here are some evaluation criteria: 1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct. 2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question. 3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct. 4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct. 5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer. Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of: A: CORRECT B: INCORRECT Just return the letters \"A\" or \"B\", with no text around it. Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer. <Original Question Begin>: {question} <Original Question End> <Gold Target Begin>: {gold_answer} <Gold Target End> <Predicted Answer Begin>: {answer} <Predicted End> Judging the correctness of candidates' answers:"""
#
#         if self.use_llm_judger:
#             print("✅ LLM Judger is ENABLED by config.")
#             if not self.model_verifier_hosts:
#                 print("⚠️ WARNING: LLM Judger is enabled, but `verifier.hosts` is empty. API calls will fail.")
#             if not self.api_key:
#                 print("⚠️ WARNING: LLM Judger is enabled, but no API Key found in `DEEPSEEK_API_KEY` env var or config. API calls will fail.")
#         else:
#             print("INFO: LLM Judger is DISABLED by config. Using rule-based verifier only.")
#
#
#     def verify(self, solution_text: str, ground_truth: str, **kwargs) -> bool:
#         if is_equiv(extract_answer(solution_text), ground_truth):
#             return True
#
#         if self.use_llm_judger and self.model_verifier_hosts and self.api_key:
#             _, model_verified = self._evaluate_with_llm(
#                 question=kwargs.get('question', ''),
#                 answer=solution_text,
#                 gold_answer=ground_truth
#             )
#             return model_verified is True
#
#         return False
#
#     def _evaluate_with_llm(self, question: str, answer: str, gold_answer: str) -> Tuple[Optional[str], Optional[bool]]:
#         prompt = self.verify_prompt_template.format(question=question, answer=answer, gold_answer=gold_answer)
#
#         request_body = {
#             "model": "deepseek-chat",
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.0,
#             "max_tokens": 10
#         }
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.api_key}"
#         }
#
#         for attempt in range(self.max_retries):
#             host = self.model_verifier_hosts[attempt % len(self.model_verifier_hosts)]
#             url = f"https://{host}/v1/chat/completions"
#
#             try:
#                 res = requests.post(url, headers=headers, json=request_body, timeout=20)
#                 res.raise_for_status()
#
#                 res_json = res.json()
#                 res_string = res_json["choices"][0]["message"]["content"].strip().upper()
#
#                 if "CORRECT" in res_string or "A" == res_string:
#                     return res_string, True
#                 elif "INCORRECT" in res_string or "B" == res_string:
#                     return res_string, False
#                 else:
#                     print(f"WARNING: LLM Judger returned ambiguous result: '{res_string}'")
#                     return res_string, None
#
#             except requests.exceptions.HTTPError as e:
#                 print(f"❌ HTTP Error on attempt {attempt+1}/{self.max_retries}: {e}")
#             except requests.exceptions.RequestException as e:
#                 print(f"❌ Network/Request Error on attempt {attempt+1}/{self.max_retries}: {e}")
#             except Exception as e:
#                 print(f"❌ An unexpected error occurred on attempt {attempt+1}/{self.max_retries}: {e}")
#
#             if attempt < self.max_retries - 1:
#                 time.sleep(self.retry_delay)
#
#         return "Failed after retries", None

# src/utils/verifier.py
# [FINAL & ROBUST - ALIYUN API COMPATIBLE]

import os
import time
import requests
import random
from typing import Tuple, Optional
from omegaconf import DictConfig
import logging
from src.utils.distributed_utils import is_main_process
from .math_equivalence import is_equiv, extract_answer

# 设置一个专用的logger
logger = logging.getLogger(__name__)


class Verifier:
    """
    [FINAL & ROBUST] 混合型答案验证器。
    - 能够通过配置文件灵活切换LLM Judger (如DeepSeek, Aliyun Dashscope)。
    - 包含更详细的日志记录，便于调试。
    """

    def __init__(self, verifier_config: DictConfig):
        self.config = verifier_config

        # 从独立的verifier_config加载所有设置
        self.use_llm_judger = self.config.get("use_llm_judger", False)
        self.model_verifier_hosts = self.config.get("hosts", [])

        # [MODIFIED] 优先从环境变量获取API Key
        # For Aliyun: DASHSCOPE_API_KEY, For DeepSeek: DEEPSEEK_API_KEY
        self.api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or self.config.get(
            "api_key", "")

        self.model_name = self.config.get("model_name", "deepseek-chat")  # 默认为deepseek
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 2.0)

        self.verify_prompt_template = """You are a helpful assistant... (内容与之前相同，此处省略) ...Judging the correctness of candidates' answers:"""

        if is_main_process():
            if self.use_llm_judger:
                logger.info("✅ LLM Judger is ENABLED.")
                logger.info(f"   - API Host(s): {self.model_verifier_hosts}")
                logger.info(f"   - Model Name: {self.model_name}")
                if not self.model_verifier_hosts:
                    logger.warning(
                        "⚠️ LLM Judger is enabled, but `verifier.hosts` is empty. API calls will likely fail.")
                if not self.api_key:
                    logger.warning(
                        "⚠️ LLM Judger is enabled, but no API Key found in environment variables or config. API calls will fail.")
            else:
                logger.info("ℹ️ LLM Judger is DISABLED. Using rule-based verifier only.")

    def verify(self, solution_text: str, ground_truth: str, **kwargs) -> bool:
        """
        验证答案。首先尝试基于规则的快速验证，如果失败且LLM Judger开启，则调用LLM进行二次判断。
        """
        # 步骤 1: 基于规则的快速验证
        extracted_answer = extract_answer(solution_text)
        if is_equiv(extracted_answer, ground_truth):
            logger.debug(f"Rule-based verifier passed for answer: '{extracted_answer}'")
            return True

        # 步骤 2: 如果规则验证失败，且LLM Judger开启，则调用LLM
        if self.use_llm_judger and self.model_verifier_hosts and self.api_key:
            logger.debug(f"Rule-based verifier failed. Escalating to LLM Judger for answer: '{extracted_answer}'")
            _, model_verified = self._evaluate_with_llm(
                question=kwargs.get('question', ''),
                answer=solution_text,  # [FIX] 传递完整的solution_text给LLM，让它看到上下文
                gold_answer=ground_truth
            )
            return model_verified is True

        # 如果LLM Judger关闭或配置不全，则返回规则验证的结果
        return False

    def _evaluate_with_llm(self, question: str, answer: str, gold_answer: str) -> Tuple[Optional[str], Optional[bool]]:
        prompt = self.verify_prompt_template.format(question=question, answer=answer, gold_answer=gold_answer)

        # [MODIFIED] 构建适用于阿里云和OpenAI兼容API的请求体
        request_body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(self.max_retries):
            # 随机选择一个host，以实现简单的负载均衡
            host = random.choice(self.model_verifier_hosts)
            # [MODIFIED] 构建完整的API URL
            url = f"https://{host}/api/v1/chat/completions"  # 阿里云百炼的URL格式

            logger.debug(f"Calling LLM Judger API (Attempt {attempt + 1}/{self.max_retries}): {url}")

            try:
                res = requests.post(url, headers=headers, json=request_body, timeout=20)
                res.raise_for_status()

                res_json = res.json()

                # [MODIFIED] 兼容不同API的返回格式
                if "choices" in res_json and res_json["choices"]:
                    res_string = res_json["choices"][0]["message"]["content"].strip().upper()
                elif "output" in res_json and "text" in res_json["output"]:  # 兼容阿里云的某些返回格式
                    res_string = res_json["output"]["text"].strip().upper()
                else:
                    logger.error(f"LLM Judger returned an unexpected JSON structure: {res_json}")
                    continue

                logger.debug(f"LLM Judger raw response: '{res_string}'")

                if "CORRECT" in res_string or "A" == res_string:
                    return res_string, True
                elif "INCORRECT" in res_string or "B" == res_string:
                    return res_string, False
                else:
                    logger.warning(f"LLM Judger returned ambiguous result: '{res_string}'")
                    return res_string, None

            except requests.exceptions.RequestException as e:
                logger.error(f"API Request Error on attempt {attempt + 1}/{self.max_retries}: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred on attempt {attempt + 1}/{self.max_retries}: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        logger.error(f"LLM Judger failed after {self.max_retries} retries.")
        return "Failed after retries", None