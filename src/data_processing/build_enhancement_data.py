# src/data_processing/build_enhancement_data.py

import os
import json
from collections import Counter
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from omegaconf import DictConfig
import hydra


def tag_skills(problem_text: str) -> list:
    problem_text = problem_text.lower()
    skills = []
    if any(s in problem_text for s in ["triangle", "sin", "cos", "tan", "degree"]):
        skills.append("trigonometry")
    if any(s in problem_text for s in ["probability", "dice", "coin", "sample", "random"]):
        skills.append("probability")
    if any(s in problem_text for s in ["series", "sum", "sequence", "arithmetic", "geometric"]):
        skills.append("sequences_and_series")
    if any(s in problem_text for s in ["derivative", "integral", "limit"]):
        skills.append("calculus")
    if any(s in problem_text for s in ["prime", "divisibility", "integer", "modular"]):
        skills.append("number_theory")
    if any(s in problem_text for s in ["geometry", "circle", "area", "volume"]):
        skills.append("geometry")
    if not skills:
        skills.append("general_algebra")
    return list(set(skills))


def create_enhancement_data(config: DictConfig) -> None:
    print("--- Creating Skill Enhancement Data for Curriculum Learning ---")

    eval_output_dir = os.path.join(config.evaluation.output_dir, config.experiment_name)
    enhancement_cfg = config.skill_enhancement
    data_cfg = config.data

    failed_problems_path = os.path.join(eval_output_dir, enhancement_cfg.failed_samples_file)
    skill_counter = Counter()
    try:
        with open(failed_problems_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if not data.get('is_correct', True):
                    skills = tag_skills(data['problem'])
                    for skill in skills:
                        skill_counter[skill] += 1
    except FileNotFoundError:
        print(f"ERROR: Failed samples file not found at '{failed_problems_path}'.")
        print("Please run evaluation on an experiment first.")
        return

    if not skill_counter:
        print("No failed skills identified. No enhancement data needed.")
        return

    num_weakest_skills = enhancement_cfg.get("num_weakest_skills_to_focus", 3)
    weakest_skills = {skill for skill, count in skill_counter.most_common(num_weakest_skills)}
    print(f"Identified weakest skills to enhance: {weakest_skills}")

    enhancement_source_path = enhancement_cfg.source_dataset_path
    print(f"Loading enhancement data source: {enhancement_source_path}")
    enhancement_source_ds = load_dataset(enhancement_source_path, split="train")

    enhancement_samples = []
    samples_per_skill = enhancement_cfg.max_samples // len(weakest_skills) if weakest_skills else 0

    for skill in weakest_skills:
        print(f"Filtering for skill: '{skill}'...")
        filtered_ds = enhancement_source_ds.filter(
            lambda example: example.get('problem') and skill in tag_skills(example['problem'])
        )
        num_to_take = min(len(filtered_ds), samples_per_skill)

        if num_to_take == 0: continue

        selected_samples = filtered_ds.shuffle(seed=config.seed).select(range(num_to_take))

        for item in tqdm(selected_samples, desc=f"  - Collecting {skill} samples"):
            enhancement_samples.append({
                "problem": item['problem'],
                "solution_cot": item['solution'],
            })

    if not enhancement_samples:
        print("Could not find any matching samples for enhancement. Exiting.")
        return

    df = pd.DataFrame(enhancement_samples)
    output_filename = data_cfg.enhancement_sft_file
    output_path = os.path.join(data_cfg.processed_dir, output_filename)
    df.to_parquet(output_path, index=False)

    print(f"✅ Created skill enhancement data with {len(df)} samples, saved to {output_path}")


@hydra.main(version_base=None, config_path="../../configs", config_name="lcare_config")
def main(config: DictConfig) -> None:
    create_enhancement_data(config)


if __name__ == "__main__":
    main()