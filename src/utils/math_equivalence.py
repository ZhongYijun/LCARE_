# src/utils/math_equivalence.py

import re
from math import isclose
from typing import Optional


def _fix_fracs(string: str) -> str:
    """处理\\frac{a}{b}和\\frac ab格式"""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            if substr and substr[0] == "{":
                new_str += "\\frac" + substr
            elif len(substr) >= 2:
                a = substr[0]
                b = substr[1]
                if b == '{':  # 处理 \frac1{...}
                    new_str += "\\frac{" + a + "}" + substr[1:]
                else:
                    new_str += f"\\frac{{{a}}}{{{b}}}" + substr[2:]
            else:
                new_str += "\\frac" + substr
    return new_str


def _fix_a_slash_b(string: str) -> str:
    """处理 a/b 格式"""
    parts = string.split('/')
    if len(parts) != 2:
        return string
    try:
        # 确保分子分母都是数字
        float(parts[0])
        float(parts[1])
        return f"\\frac{{{parts[0]}}}{{{parts[1]}}}"
    except (ValueError, TypeError):
        return string


def _fix_sqrt(string: str) -> str:
    """处理 \\sqrtN 和 \\sqrt{N}"""
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _remove_right_units(string: str) -> str:
    """移除末尾的文本单位, e.g. \\text{ dollars}"""
    return re.sub(r"\\text{.*?}", "", string).strip()


def _strip_string(string: str) -> str:
    """
    对数学答案字符串进行全面的归一化，以便进行比较。
    """
    if not isinstance(string, str):
        string = str(string)

    string = string.strip()

    # 移除LaTeX盒子, e.g. \boxed{answer} -> answer
    string = re.sub(r"\\boxed\{(.*?)\}", r"\1", string)
    string = re.sub(r"\\fbox\{(.*?)\}", r"\1", string)

    # 标准化LaTeX命令
    string = string.replace("\\left", "").replace("\\right", "")
    # [修复] 移除冗余的转义符
    string = string.replace("\\!", "").replace(" ", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace("\\cdot", "*")
    string = string.replace("..", ".")  # 修复小数点错误

    # 移除文本
    string = _remove_right_units(string)
    string = re.sub(r"\\text{.*?}", "", string)

    # 修复常见格式
    string = _fix_sqrt(string)
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    # 标准化数字和小数点
    string = string.replace(" .", " 0.")
    if string.startswith("."): string = "0" + string
    # 移除不重要的尾随零， e.g. 1.200 -> 1.2, but 1200 stays 1200
    if '.' in string:
        string = string.rstrip('0').rstrip('.')

    # 移除等式左边, e.g. x=5 -> 5
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # 再次移除可能产生的多余空格
    string = string.replace(" ", "")

    if string.endswith('.'): string = string[:-1]

    return string


def is_equiv(str1: str, str2: str) -> bool:
    """
    判断两个数学答案字符串是否等价。
    """
    if str1 is None and str2 is None: return True
    if str1 is None or str2 is None: return False
    if not isinstance(str1, str) or not isinstance(str2, str): return False

    try:
        # 归一化后进行字符串比较
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if ss1 == ss2:
            return True
    except Exception:
        # 如果归一化失败，则退回到原始比较
        return str1.strip() == str2.strip()

    try:
        # 尝试作为浮点数比较
        f1 = float(ss1)
        f2 = float(ss2)
        return isclose(f1, f2, rel_tol=1e-4)
    except (ValueError, TypeError):
        pass

    return False


def extract_answer(solution_text: str) -> Optional[str]:
    """
    从模型生成的完整文本中提取最终答案。
    """
    if not isinstance(solution_text, str):
        return None

    # 优先匹配\boxed{}
    boxed_match = re.search(r"\\boxed\{(.*?)\}", solution_text)
    if boxed_match:
        return boxed_match.group(1)

    # 后备：匹配 "The final answer is"
    answer_is_match = re.search(r"[Tt]he(?: final)? answer is:?\s*(.*)", solution_text)
    if answer_is_match:
        # 取到行尾或句号
        return answer_is_match.group(1).split('\n')[0].strip().rstrip('.')

    # 后备：提取最后一个数字
    numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", solution_text.replace(",", ""))
    return numbers[-1] if numbers else None