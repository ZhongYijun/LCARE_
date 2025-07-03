# # src/rl/sum_tree.py

# import numpy as np
# from typing import Any


# class SumTree:
#     """
#     一个用于优先经验回放(PER)的SumTree数据结构。
#     它被实现为一个二叉树，存储在单个Numpy数组中，以实现高效的更新和采样。
#     树的叶子节点存储每个样本的优先级，内部节点存储其子节点的优先级之和。
#     """

#     def __init__(self, capacity: int):
#         """
#         初始化SumTree。

#         Args:
#             capacity (int): 缓冲区可以存储的最大样本数。
#         """
#         if not isinstance(capacity, int) or capacity <= 0:
#             raise ValueError(f"Capacity must be a positive integer, got {capacity}")

#         # 树的节点总数是 2 * capacity - 1
#         self.capacity = capacity
#         # 树结构，用numpy数组存储
#         self.tree = np.zeros(2 * capacity - 1)
#         # 实际的数据存储
#         self.data = np.zeros(capacity, dtype=object)

#         # 指向下一个要插入数据的位置
#         self.data_pointer = 0
#         # 当前存储的样本数量
#         self.size = 0

#     @property
#     def total_priority(self) -> float:
#         """返回所有优先级的总和，即树的根节点值。"""
#         return self.tree[0]

#     def add(self, priority: float, data: Any):
#         """
#         向树中添加一个新样本及其优先级。
#         如果缓冲区已满，则覆盖最旧的数据。
#         """
#         self.data[self.data_pointer] = data
#         self.update(self.data_pointer, priority)

#         self.data_pointer += 1
#         if self.data_pointer >= self.capacity:
#             self.data_pointer = 0  # 循环写入

#         if self.size < self.capacity:
#             self.size += 1

#     def update(self, data_idx: int, priority: float):
#         """
#         更新指定索引的数据的优先级，并沿树向上传播变化。

#         Args:
#             data_idx (int): data数组中的索引。
#             priority (float): 新的优先级值。
#         """
#         if not 0 <= data_idx < self.capacity:
#             raise IndexError("Data index out of range.")

#         tree_idx = data_idx + self.capacity - 1
#         change = priority - self.tree[tree_idx]
#         self.tree[tree_idx] = priority

#         # 沿树向上回溯，更新所有父节点的和
#         while tree_idx != 0:
#             tree_idx = (tree_idx - 1) // 2
#             self.tree[tree_idx] += change

#     def get(self, s: float) -> tuple[int, float, Any]:
#         """
#         根据给定的累积优先级值s，从树中采样一个样本。

#         Args:
#             s (float): 一个在[0, total_priority)范围内的随机值。

#         Returns:
#             tuple[int, float, Any]: (树叶子节点的索引, 该样本的优先级, 该样本的数据)
#         """
#         parent_idx = 0
#         while True:
#             left_child_idx = 2 * parent_idx + 1
#             right_child_idx = left_child_idx + 1

#             # 如果到达叶子节点
#             if left_child_idx >= len(self.tree):
#                 leaf_idx = parent_idx
#                 break

#             # 否则，决定往左子树还是右子树走
#             if s <= self.tree[left_child_idx]:
#                 parent_idx = left_child_idx
#             else:
#                 s -= self.tree[left_child_idx]
#                 parent_idx = right_child_idx

#         data_idx = leaf_idx - self.capacity + 1
#         return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

# src/rl/sum_tree.py

import numpy as np
from typing import Any


class SumTree:
    """
    [CARE-V1.1] 一个用于优先经验回放(PER)的SumTree数据结构。
    - 修复了所有类型提示问题，以匹配严格的静态检查。
    """

    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"Capacity must be a positive integer, got {capacity}")

        self.capacity = capacity
        # 树的节点总数是 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # 实际的数据指针存储
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    @property
    def total_priority(self) -> float:
        """
        [FIX] 返回所有优先级的总和。
        显式转换为float以满足类型提示。
        """
        return float(self.tree[0])

    def add(self, priority: float, data: Any):
        """向树中添加一个新样本及其优先级。"""
        tree_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # 循环写入

        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx: int, priority: float):
        """更新指定索引的数据的优先级，并沿树向上传播变化。"""
        if tree_idx < 0 or tree_idx >= len(self.tree):
            return # 安全检查

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # 只要不是根节点，就继续向上传播
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s: float) -> tuple[int, float, Any]:
        """
        [FIX] 根据给定的累积优先级值s，从树中采样一个样本。
        显式转换为float以满足类型提示。
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            
            # 如果左子节点超出范围，说明当前parent_idx就是叶子节点
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            right_child_idx = left_child_idx + 1
            if s <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                s -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), self.data[data_idx]