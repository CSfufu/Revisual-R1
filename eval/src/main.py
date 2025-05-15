import os
import json
import argparse
from utils.model_wrapper import OpenAIWrapper
from utils.data_loaders import (
    load_fufu_hard1_dataset
)
from utils.processing import (
    slice_dataset,
    process_generation,
    process_evaluation,
    calculate_metrics
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified evaluation for multimodal math datasets")

    dirname = os.path.abspath(os.path.dirname(__file__))

    # Model and runtime parameters
    parser.add_argument("--model-name", type=str, required=True, help="The name of the model to use")
    parser.add_argument("--openai-api-key", type=str, required=True, help="The API key for the OpenAI API")
    parser.add_argument("--openai-base-url", type=str, default="https://api.openai.com/v1", help="The base URL for the OpenAI API")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache predictions")
    parser.add_argument("--output-dir", type=str, default=os.path.join(dirname, "../output"), help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum number of tokens to generate")
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    # parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--system-prompt", type=str, default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.", help="System prompt for the model")

    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision or 'all'")
    parser.add_argument("--dataset-dir", type=str, default=os.path.join(dirname, "../dataset"), help="")

    parser.add_argument("--eval-threads", type=int, default=8, help="Number of threads for evaluation")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for evaluation")

    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets_to_eval = args.datasets.split(",") if args.datasets != "all" else [
        "fufu_hard1"
    ]

    all_samples = {}

    for dataset_name in datasets_to_eval:
        if dataset_name == "fufu_hard1":
            all_samples["fufu_hard1"] = load_fufu_hard1_dataset(args.dataset_dir)

    if not all_samples:
        print("No datasets loaded. Please check the paths and dataset names.")
        return

    model = OpenAIWrapper(
        args.model_name,
        args.openai_api_key,
        args.openai_base_url,
        args.max_tokens,
        args.temperature,
        args.top_p)

    # Process in batches
    all_results = {}
    for dataset_name in all_samples.keys():
        all_results[dataset_name] = []

    for dataset_name, samples in all_samples.items():
        predictions = process_generation(model, samples, args)

        results = process_evaluation(predictions, samples, args)

        metrics = calculate_metrics(results)

        output_dict = {
            "results": results,
            "metrics": metrics,
            "config": vars(args)
        }

        output_path = os.path.join(args.output_dir, f"{args.model_name}___{dataset_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

        print(f"{dataset_name.upper()} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        if 'sub_metrics' in metrics:
            print("  Task/Category Accuracies:")
            for task, metric in metrics['sub_metrics'].items():
                print(f"    {task}: {metric['accuracy']:.4f} ({metric['correct']}/{metric['total']})")
        print()

    print(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
