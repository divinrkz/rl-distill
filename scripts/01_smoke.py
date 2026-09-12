import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset

from rl_distill.formats import load_formats
from rl_distill.generate import free_llm, generate, load_llm
from rl_distill.grading import is_correct
from rl_distill.runio import load_config, make_run_dir, save_config, save_json


def load_problems(dataset_cfg):
    ds = load_dataset(dataset_cfg["name"], split=dataset_cfg["split"])
    return [ds[i] for i in dataset_cfg["indices"]]


def run_model(model_cfg, fmt, problems, config, run_dir):
    llm = load_llm(model_cfg["model_name"], config["vllm"])
    tokenizer = llm.get_tokenizer()

    prompts = [fmt.build_prompt(p["problem"], tokenizer) for p in problems]
    completions = generate(llm, prompts, config["generation"])

    records, n_correct = [], 0
    for problem, prompt, comps in zip(problems, prompts, completions):
        text = comps[0]
        answer = fmt.extract_answer(text)
        correct = is_correct(text, problem["answer"])
        n_correct += int(correct)
        records.append(
            {
                "problem": problem["problem"],
                "gold": problem["answer"],
                "prompt": prompt,
                "completion": text,
                "thinking": fmt.extract_thinking(text),
                "answer": answer,
                "correct": correct,
            }
        )

    free_llm(llm)

    model_dir = run_dir / model_cfg["id"]
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / "results.json", records)
    return n_correct, len(records)


def main(config_path):
    config = load_config(config_path)
    run_dir = make_run_dir(config["out_base"], config["run_name"])
    save_config(run_dir, config)

    formats = load_formats(config["formats"])
    problems = load_problems(config["dataset"])

    print(f"Run dir: {run_dir}")
    for model_cfg in config["models"]:
        fmt = formats[model_cfg["format"]]
        n_correct, n = run_model(model_cfg, fmt, problems, config, run_dir)
        print(f"  {model_cfg['id']}: {n_correct}/{n} correct")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.yaml")
