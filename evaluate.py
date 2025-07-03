# # evaluate.py
# # [L-CARE V6 - FINAL, FIXED, DDP-ACCELERATED]
#
# import os
# import json
# import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler
# from tqdm import tqdm
# import hydra
# from omegaconf import DictConfig
# from functools import partial
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel
# import matplotlib.pyplot as plt
#
# import logging
# from collections import defaultdict
# from datetime import datetime
# from typing import Optional
#
# # [FIX] 导入缺失的类型和DDP
# from torch.nn.parallel import DistributedDataParallel as DDP
#
# # 本地项目导入
# from src.utils.verifier import Verifier
# from src.utils.prompt_constructor import PromptConstructor
# from src.datasets.evaluation_dataset import EvaluationDataset, collate_fn_eval
# from src.utils.logger import SwanLabLogger
# from src.utils.distributed_utils import is_main_process, setup_distributed, cleanup_distributed, get_rank, \
#     get_world_size
#
# # 配置Python标准日志记录器
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# def log_and_plot_history(experiment_dir: str, swanlab_logger: SwanLabLogger, current_step: int):
#     """
#     扫描实验目录下的所有评估结果，绘制历史准确率曲线并上传到SwanLab。
#     """
#     if not is_main_process():
#         return
#
#     history = defaultdict(list)
#     steps = []
#
#     eval_results_dir = os.path.join(experiment_dir, "eval_results")
#     if not os.path.isdir(eval_results_dir):
#         logger.info("未找到评估历史目录，跳过绘图。")
#         return
#
#     # 1. 收集所有评估结果
#     # 按照迭代步数排序
#     try:
#         # 提取 "iter_XXX" 中的数字并排序
#         eval_folders = [d for d in os.listdir(eval_results_dir) if d.startswith("iter_")]
#         sorted_eval_folders = sorted(eval_folders, key=lambda x: int(x.split('_')[1]))
#     except (FileNotFoundError, IndexError, ValueError):
#         logger.warning(f"评估结果目录 {eval_results_dir} 结构不规范，无法排序。")
#         return
#
#     for eval_folder in sorted_eval_folders:
#         summary_path = os.path.join(eval_results_dir, eval_folder, "evaluation_summary.json")
#         if os.path.exists(summary_path):
#             try:
#                 steps.append(int(eval_folder.split('_')[1]))
#                 with open(summary_path, 'r', encoding='utf-8') as f:
#                     summary = json.load(f)
#                     for dataset_name, results in summary.items():
#                         history[dataset_name].append(results.get("pass@1", 0))
#             except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
#                 logger.warning(f"无法解析或处理评估文件 {summary_path}: {e}")
#
#     if not history or not steps:
#         logger.info("未找到足够的历史评估数据来绘图。")
#         return
#
#     # 2. 绘制图表
#     plt.style.use('dark_background')
#     fig, ax = plt.subplots(figsize=(12, 7))
#
#     for dataset_name, scores in history.items():
#         if len(scores) == len(steps):
#             ax.plot(steps, scores, marker='o', linestyle='-', label=dataset_name)
#         else:
#             logger.warning(
#                 f"数据集 '{dataset_name}' 的历史数据点 ({len(scores)}) 与步数 ({len(steps)}) 数量不匹配，跳过绘图。")
#
#     ax.set_title(f'Pass@1 Accuracy History for {os.path.basename(experiment_dir)}', fontsize=16, color='white')
#     ax.set_xlabel('Training Iteration', fontsize=12, color='white')
#     ax.set_ylabel('Pass@1 Accuracy', fontsize=12, color='white')
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
#     ax.legend()
#     ax.tick_params(axis='x', colors='white')
#     ax.tick_params(axis='y', colors='white')
#     fig.tight_layout()
#
#     # 3. 保存并上传到SwanLab
#     plot_path = os.path.join(experiment_dir, "accuracy_history.png")
#     fig.savefig(plot_path, facecolor='black', bbox_inches='tight')
#     plt.close(fig)
#
#     if swanlab_logger and swanlab_logger.use_swanlab:
#         try:
#             import swanlab
#             # [FIX] 使用 current_step 来记录图表
#             swanlab_logger.log({"Evaluation History": swanlab.Image(plot_path)}, step=current_step)
#             logger.info(f"✅ 准确率历史曲线图已上传至SwanLab。")
#         except ImportError:
#             logger.warning("SwanLab未安装，无法上传历史曲线图。")
#         except Exception as e:
#             logger.error(f"上传图表到SwanLab时出错: {e}")
#
#
# def run_evaluation(config: DictConfig, swanlab_logger: Optional[SwanLabLogger] = None,
#                    current_iteration: Optional[int] = None):
#     """
#     [V6] 主评估函数。
#     - 修复了所有已知的import和函数调用错误。
#     - 接收 current_iteration 用于日志记录。
#     """
#     eval_config = config.evaluation
#     exp_name = config.experiment_name
#
#     # [FIX] 使用传入的或新生成的run_name作为目录名
#     eval_run_name = eval_config.get("run_name", f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
#     output_dir = os.path.join(eval_config.output_dir, exp_name, eval_run_name)
#
#     if is_main_process():
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f"--- 开始评估流程 (V6 - 最终修复版) ---")
#         logger.info(f"评估实验: {exp_name}/{eval_run_name}")
#         logger.info(f"结果将保存至: {output_dir}")
#
#     setup_distributed(get_rank(), get_world_size())
#     device = torch.device(f"cuda:{get_rank()}")
#
#     tokenizer = AutoTokenizer.from_pretrained(eval_config.base_model_path, trust_remote_code=True, padding_side='left')
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     model = AutoModelForCausalLM.from_pretrained(
#         eval_config.base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
#     ).to(device)
#
#     if eval_config.get("load_lora_adapter", False):
#         if is_main_process(): logger.info(f"加载LoRA适配器: '{eval_config.model_path}'")
#         model = PeftModel.from_pretrained(model, eval_config.model_path, is_trainable=False)
#     else:
#         if is_main_process(): logger.info("评估基座模型，不加载LoRA。")
#
#     model = DDP(model, device_ids=[get_rank()], find_unused_parameters=True)
#     model.eval()
#
#     verifier = Verifier(config.verifier)
#     prompt_constructor = PromptConstructor(config, tokenizer)
#
#     full_summary = {}
#     for ds_conf in eval_config.datasets:
#         name = ds_conf['name']
#         if is_main_process(): logger.info(f"\n--- 评估数据集: {name} ---")
#
#         # [FIX] 修正了EvaluationDataset的初始化，传入了tokenizer
#         dataset = EvaluationDataset(ds_conf, prompt_constructor, tokenizer)
#         sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
#         dataloader = DataLoader(dataset, batch_size=eval_config.batch_size_per_gpu, sampler=sampler,
#                                 collate_fn=partial(collate_fn_eval, tokenizer=tokenizer))
#
#         local_results = []
#         pbar = tqdm(dataloader, disable=(not is_main_process()), desc=f"评估 {name}")
#
#         for batch in pbar:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#
#             with torch.no_grad():
#                 generated_sequences = model.module.generate(
#                     input_ids=input_ids, attention_mask=attention_mask,
#                     max_new_tokens=2048, do_sample=False, pad_token_id=tokenizer.pad_token_id,
#                     eos_token_id=tokenizer.eos_token_id
#                 )
#
#             full_solution_texts = tokenizer.batch_decode(generated_sequences.cpu(), skip_special_tokens=True)
#             prompts_text = batch['problems_text']
#
#             for i in range(len(prompts_text)):
#                 prompt_text = prompts_text[i]
#                 prompt_for_model = prompt_constructor.get_evaluation_prompt(prompt_text)
#                 full_text = full_solution_texts[i]
#                 solution_only = full_text[len(prompt_for_model):].strip() if full_text.startswith(
#                     prompt_for_model) else full_text.strip()
#
#                 is_correct = verifier.verify(solution_text=solution_only, ground_truth=batch['ground_truths'][i],
#                                              question=prompt_text)
#
#                 local_results.append({
#                     "problem_idx_in_dataset": batch['problem_indices'][i].item(),
#                     "problem": prompt_text, "generated_solution": solution_only,
#                     "ground_truth_answer": batch['ground_truths'][i], "is_correct": is_correct
#                 })
#
#         gathered_results = [None] * get_world_size()
#         dist.all_gather_object(gathered_results, local_results)
#
#         if is_main_process():
#             flat_results = [item for sublist in gathered_results for item in sublist]
#             correct_count = sum(1 for item in flat_results if item['is_correct'])
#             total_count = len(dataset)  # 使用len(dataset)确保总数正确
#
#             pass_at_1 = (correct_count / total_count) if total_count > 0 else 0
#             full_summary[name] = {"pass@1": pass_at_1, "correct": correct_count, "total": total_count}
#             logger.info(f"数据集 '{name}' 结果: pass@1 = {pass_at_1:.4f} ({correct_count}/{total_count})")
#
#             detailed_results_path = os.path.join(output_dir, f"detailed_results_{name}.jsonl")
#             with open(detailed_results_path, 'w', encoding='utf-8') as f:
#                 for entry in sorted(flat_results, key=lambda x: x['problem_idx_in_dataset']):
#                     f.write(json.dumps(entry) + '\n')
#             logger.info(f"详细结果已保存至 {detailed_results_path}")
#
#     if is_main_process():
#         summary_path = os.path.join(output_dir, "evaluation_summary.json")
#         with open(summary_path, 'w', encoding='utf-8') as f:
#             json.dump(full_summary, f, indent=4)
#         logger.info(f"\n✅ 评估完成。总结报告: {summary_path}")
#
#         if swanlab_logger and swanlab_logger.use_swanlab and current_iteration is not None:
#             log_data = {}
#             total_correct, total_problems = 0, 0
#             for ds_name, res in full_summary.items():
#                 log_data[f"eval_pass@1/{ds_name}"] = res["pass@1"]
#                 total_correct += res["correct"]
#                 total_problems += res["total"]
#             if total_problems > 0:
#                 log_data["eval_pass@1/Overall_Average"] = total_correct / total_problems
#
#             # [FIX] 使用传入的 current_iteration 作为 step
#             swanlab_logger.log(log_data, step=current_iteration)
#             logger.info("评估结果已记录到SwanLab。")
#
#             try:
#                 # 实验目录是 output_dir 的上一级
#                 log_and_plot_history(os.path.dirname(output_dir), swanlab_logger, current_iteration)
#             except Exception as e:
#                 logger.error(f"绘制历史图表时出错: {e}")
#
#     cleanup_distributed()
#
#
# @hydra.main(version_base=None, config_path="../configs", config_name="lcare_config")
# def hydra_main(config: DictConfig) -> None:
#     run_evaluation(config)
#
#
# if __name__ == '__main__':
#     hydra_main()

# # evaluate.py
# # [L-CARE V15.4-FIXED - FINAL ROBUST & SYNERGISTIC FORM]
# import os
# import json
# import torch
# import hydra
# from omegaconf import DictConfig
# from tqdm import tqdm
# from typing import Optional
# import logging
# from peft import PeftModel
# import torch.distributed as dist
# # Local project imports
# from src.utils.config_loader import load_and_resolve_config
# from src.utils.distributed_utils import setup_distributed, cleanup_distributed, get_rank, is_main_process
# from src.utils.logger import SwanLabLogger
# from src.utils.verifier import Verifier
# from src.utils.prompt_constructor import PromptConstructor
# from src.models.actor_critic import LCARE_Actor
# from src.datasets.evaluation_dataset import EvaluationDataset, collate_fn_eval
# from torch.utils.data import DataLoader, DistributedSampler
# from functools import partial
# from transformers import AutoTokenizer, GenerationConfig
#
# # Setup logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# def run_evaluation(
#         config: DictConfig,
#         swanlab_logger: Optional[SwanLabLogger] = None,
#         current_iteration: int = -1,
#         is_async_eval: bool = False # [V15.4-FIX] Flag to control distributed setup
# ):
#     """
#     Main function to run evaluation on specified datasets.
#     Can be run standalone (distributed) or as a single-process async task.
#     """
#     if not is_async_eval:
#         logger.info("--- Starting Standalone Evaluation Flow (Distributed) ---")
#     else:
#         logger.info(f"--- Starting Async Evaluation for Iteration {current_iteration} (Single-Process) ---")
#
#     # --- 1. Setup Environment and Device ---
#     try:
#         if not is_async_eval:
#             setup_distributed(get_rank(), int(os.environ.get("WORLD_SIZE", 1)))
#             rank = get_rank()
#             world_size = int(os.environ.get("WORLD_SIZE", 1))
#             device = torch.device(f"cuda:{rank}")
#         else:
#             rank = 0
#             world_size = 1
#             device = torch.device("cuda:0")
#             logger.info("Running in single-process async mode on cuda:0.")
#
#         # --- 2. Load Model and Tokenizer ---
#         eval_cfg = config.evaluation
#         model_cfg = config.model
#
#         tokenizer = AutoTokenizer.from_pretrained(eval_cfg.model_path, trust_remote_code=True)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#
#         # Use LCARE_Actor as the container for the model
#         actor = LCARE_Actor(model_cfg, tokenizer).to(device)
#
#         if eval_cfg.load_lora_adapter:
#             logger.info(f"[Rank {rank}] Loading LoRA adapter from: {eval_cfg.model_path}")
#             # Ensure the base model is on the correct device before loading adapter
#             actor.model = PeftModel.from_pretrained(actor.model.to(device), eval_cfg.model_path, is_trainable=False)
#
#         actor.to(device).eval()
#
#         if not is_async_eval and world_size > 1:
#             actor = torch.nn.parallel.DistributedDataParallel(actor, device_ids=[rank])
#
#         # --- 3. Prepare Tools and Data ---
#         verifier = Verifier(config.verifier)
#         prompt_constructor = PromptConstructor(config, tokenizer)
#
#         all_results = {}
#         output_base_dir = os.path.join(eval_cfg.output_dir, eval_cfg.run_name)
#         if is_main_process():
#             os.makedirs(output_base_dir, exist_ok=True)
#
#         # --- 4. Evaluation Loop ---
#         for dataset_cfg in eval_cfg.datasets:
#             dataset_name = dataset_cfg.name
#             logger.info(f"[Rank {rank}] Starting evaluation on dataset: {dataset_name}")
#
#             dataset = EvaluationDataset(dataset_cfg, prompt_constructor, tokenizer)
#             if not dataset:
#                 logger.warning(f"Dataset '{dataset_name}' is empty or failed to load. Skipping.")
#                 continue
#
#             sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
#             collate_fn_with_tokenizer = partial(collate_fn_eval, tokenizer=tokenizer)
#             dataloader = DataLoader(dataset, batch_size=eval_cfg.batch_size, sampler=sampler, collate_fn=collate_fn_with_tokenizer)
#
#             correct_count, total_count = 0, 0
#             failed_samples = []
#
#             pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}", disable=not is_main_process())
#
#             for batch in pbar:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#
#                 with torch.no_grad():
#                     # Use the generate method from the underlying model
#                     model_to_generate = actor.module if isinstance(actor, torch.nn.parallel.DistributedDataParallel) else actor
#                     gen_conf = GenerationConfig(max_new_tokens=2048, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
#                     action_ids, _ = model_to_generate.generate(input_ids, attention_mask, gen_conf.to_dict())
#
#                 # Decode and verify results
#                 full_solution_texts = tokenizer.batch_decode(action_ids, skip_special_tokens=True)
#                 prompts = [prompt_constructor.get_evaluation_prompt(p) for p in batch['problems_text']]
#
#                 for i in range(len(full_solution_texts)):
#                     solution_only = full_solution_texts[i][len(prompts[i]):].strip()
#                     is_correct = verifier.verify(solution_text=solution_only, ground_truth=batch['ground_truths'][i], question=batch['problems_text'][i])
#
#                     if is_correct:
#                         correct_count += 1
#                     elif is_main_process():
#                         failed_samples.append({"problem": batch['problems_text'][i], "generated_solution": solution_only, "ground_truth": batch['ground_truths'][i]})
#
#                     total_count += 1
#
#             # --- 5. Aggregate and Log Results ---
#             if world_size > 1:
#                 counts = torch.tensor([correct_count, total_count], dtype=torch.long, device=device)
#                 dist.all_reduce(counts, op=dist.ReduceOp.SUM)
#                 correct_count, total_count = counts.tolist()
#
#             if is_main_process() and total_count > 0:
#                 pass_rate = correct_count / total_count
#                 logger.info(f"Results for {dataset_name}: Pass@1 = {pass_rate:.4f} ({correct_count}/{total_count})")
#
#                 dataset_results = {"pass@1": pass_rate, "correct": correct_count, "total": total_count, "iteration": current_iteration}
#                 all_results[dataset_name] = dataset_results
#
#                 if swanlab_logger:
#                     swanlab_logger.log({f"eval/{dataset_name}_pass@1": pass_rate}, step=current_iteration)
#
#                 # Save detailed results
#                 summary_path = os.path.join(output_base_dir, "evaluation_summary.json")
#                 with open(summary_path, 'w') as f:
#                     json.dump(all_results, f, indent=4)
#
#                 failed_path = os.path.join(output_base_dir, f"detailed_results_{dataset_name}.jsonl")
#                 with open(failed_path, 'w') as f:
#                     for sample in failed_samples:
#                         f.write(json.dumps(sample) + '\n')
#
#     finally:
#         # --- 6. Cleanup ---
#         if not is_async_eval and dist.is_initialized():
#             cleanup_distributed()
#
# @hydra.main(version_base=None, config_path="../configs", config_name="lcare_config")
# def main(config: DictConfig):
#     config = load_and_resolve_config(config)
#     # When run as main, it's never an async eval
#     run_evaluation(config, is_async_eval=False)
#
# if __name__ == "__main__":
#     main()

# evaluate.py
# [L-CARE V15.5-FIXED - FINAL ROBUST & SYNERGISTIC FORM]

import os
import json
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Optional
import logging
from peft import PeftModel
from functools import partial
import torch.distributed as dist
# Local project imports
from src.utils.config_loader import load_and_resolve_config
from src.utils.distributed_utils import setup_distributed, cleanup_distributed, get_rank, get_world_size, \
    is_main_process
from src.utils.logger import SwanLabLogger
from src.utils.verifier import Verifier
from src.utils.prompt_constructor import PromptConstructor
from src.models.actor_critic import LCARE_Actor
from src.datasets.evaluation_dataset import EvaluationDataset, collate_fn_eval
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, GenerationConfig

# Setup a dedicated logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_evaluation(
        config: DictConfig,
        swanlab_logger: Optional[SwanLabLogger] = None,
        current_iteration: int = -1,
        is_async_eval: bool = False  # Flag to control distributed setup
):
    """
    Main function to run evaluation on specified datasets.
    This function is designed to be robust for two scenarios:
    1. Standalone Distributed Run: Triggered via `04_run_evaluation.sh`. It sets up its own process group.
    2. Asynchronous Single-Process Run: Triggered by `rl_agent.py`. It runs on a single GPU
       and MUST NOT initialize a new process group.
    """
    if not is_async_eval:
        logger.info("--- Starting Standalone Evaluation Flow (Distributed Mode) ---")
    else:
        logger.info(f"--- Starting Async Evaluation for Iteration {current_iteration} (Single-Process Mode) ---")

    # --- 1. Setup Environment and Device ---
    try:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")  # Default to cuda:0 for single process

        if not is_async_eval:
            # Only setup distributed environment if it's a standalone run
            setup_distributed(get_rank(), int(os.environ.get("WORLD_SIZE", 1)))
            rank = get_rank()
            world_size = get_world_size()
            device = torch.device(f"cuda:{rank}")
        else:
            logger.info("Running in single-process async mode on cuda:0.")

        # --- 2. Load Model and Tokenizer (Robust Loading Logic) ---
        eval_cfg = config.evaluation
        model_cfg = config.model

        # [CRITICAL FIX] Always load tokenizer and config from the BASE model path.
        base_model_path = eval_cfg.base_model_path
        adapter_path = eval_cfg.get("adapter_path")  # Use .get() for safety, returns None if not exists

        logger.info(f"[Rank {rank}] Loading tokenizer from base model path: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load the base model architecture first
        logger.info(f"[Rank {rank}] Loading base model architecture...")
        # We use LCARE_Actor as a standard container for our Causal LM
        actor = LCARE_Actor(model_cfg, tokenizer)

        # If an adapter path is provided, load and apply it to the base model
        if adapter_path and os.path.isdir(adapter_path):
            logger.info(f"[Rank {rank}] Loading and applying LoRA adapter from: {adapter_path}")
            # PeftModel.from_pretrained handles loading the adapter and merging it correctly
            actor.model = PeftModel.from_pretrained(actor.model.to(device), adapter_path, is_trainable=False)
        else:
            logger.info(f"[Rank {rank}] No valid adapter path provided. Evaluating the base model.")

        actor.to(device).eval()

        # If in distributed mode, wrap the model with DDP
        if not is_async_eval and world_size > 1:
            actor = torch.nn.parallel.DistributedDataParallel(actor, device_ids=[rank])

        # --- 3. Prepare Tools and Data ---
        verifier = Verifier(config.verifier)
        prompt_constructor = PromptConstructor(config, tokenizer)

        all_results = {}
        # Ensure the output directory structure is created by the main process
        output_base_dir = os.path.join(eval_cfg.output_dir, eval_cfg.run_name)
        if is_main_process():
            os.makedirs(output_base_dir, exist_ok=True)

        # --- 4. Evaluation Loop ---
        for dataset_cfg in eval_cfg.datasets:
            dataset_name = dataset_cfg.name
            if is_main_process():
                logger.info(f"--- Starting evaluation on dataset: {dataset_name} ---")

            dataset = EvaluationDataset(dataset_cfg, prompt_constructor, tokenizer)
            if not dataset:
                logger.warning(f"Dataset '{dataset_name}' is empty or failed to load. Skipping.")
                continue

            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=False) if not is_async_eval and world_size > 1 else None
            collate_fn_with_tokenizer = partial(collate_fn_eval, tokenizer=tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=eval_cfg.batch_size,
                sampler=sampler,
                collate_fn=collate_fn_with_tokenizer,
                num_workers=2  # Can use a few workers for data loading
            )

            correct_count, total_count = 0, 0
            failed_samples = []

            pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}", disable=not is_main_process())

            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                with torch.no_grad():
                    # Use the generate method from the underlying model, handling DDP wrapper
                    model_to_generate = actor.module if isinstance(actor,
                                                                   torch.nn.parallel.DistributedDataParallel) else actor
                    gen_conf = GenerationConfig(max_new_tokens=2048, do_sample=False,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id)

                    # The generate method is defined in LCARE_Actor
                    action_ids, _ = model_to_generate.generate(input_ids, attention_mask, gen_conf.to_dict())

                # Decode and verify results
                full_solution_texts = tokenizer.batch_decode(action_ids, skip_special_tokens=True)
                prompts = [prompt_constructor.get_evaluation_prompt(p) for p in batch['problems_text']]

                for i in range(len(full_solution_texts)):
                    # Robustly strip the prompt from the beginning of the generated text
                    prompt_text = prompts[i]
                    generated_text = full_solution_texts[i]
                    if generated_text.startswith(prompt_text):
                        solution_only = generated_text[len(prompt_text):].strip()
                    else:
                        solution_only = generated_text.strip()

                    is_correct = verifier.verify(solution_text=solution_only, ground_truth=batch['ground_truths'][i],
                                                 question=batch['problems_text'][i])

                    if is_correct:
                        correct_count += 1
                    elif is_main_process():  # Only main process collects failed samples
                        failed_samples.append(
                            {"problem": batch['problems_text'][i], "generated_solution": solution_only,
                             "ground_truth": batch['ground_truths'][i]})

                    total_count += 1

            # --- 5. Aggregate and Log Results ---
            if not is_async_eval and world_size > 1:
                counts = torch.tensor([correct_count, total_count], dtype=torch.long, device=device)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                correct_count, total_count = counts.tolist()

            if is_main_process() and total_count > 0:
                pass_rate = correct_count / total_count
                logger.info(f"Results for {dataset_name}: Pass@1 = {pass_rate:.4f} ({correct_count}/{total_count})")

                dataset_results = {"pass@1": pass_rate, "correct": correct_count, "total": total_count,
                                   "iteration": current_iteration}
                all_results[dataset_name] = dataset_results

                if swanlab_logger:
                    # Log to SwanLab with a clear naming convention
                    log_metric = {f"eval/{dataset_name}_pass@1": pass_rate}
                    # Use the correct step (iteration number) for logging
                    step_to_log = current_iteration if current_iteration != -1 else 0
                    swanlab_logger.log(log_metric, step=step_to_log)

                # Save detailed results to disk
                summary_path = os.path.join(output_base_dir, "evaluation_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(all_results, f, indent=4)

                failed_path = os.path.join(output_base_dir, f"detailed_results_{dataset_name}.jsonl")
                with open(failed_path, 'w') as f:
                    for sample in failed_samples:
                        f.write(json.dumps(sample) + '\n')

    finally:
        # --- 6. Cleanup ---
        # Only cleanup if we initialized the process group in this function
        if not is_async_eval and dist.is_initialized():
            cleanup_distributed()


@hydra.main(version_base=None, config_path="../configs", config_name="lcare_config")
def main(config: DictConfig):
    """Entry point for standalone evaluation."""
    config = load_and_resolve_config(config)
    # When run as main, it's never an async eval.
    # We don't pass a logger or iteration, as it's a one-off run.
    run_evaluation(config, is_async_eval=False)


if __name__ == "__main__":
    main()