import os
import math
from PIL import Image
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utils.model_parser import (
    generate_prediction,
    evaluate_prediction
)


def slice_dataset(dataset, start, end):
    result = []
    for i, example in enumerate(dataset):
        if i < start:
            continue
        if i >= end:
            break
        result.append(example)
    return result


def process_generation(model, tasks, args) -> List[str]:
    predictions = []
    with ThreadPoolExecutor(max_workers=args.eval_threads) as executor:
        futures = []

        for i, task in enumerate(tasks):
            future = executor.submit(
                generate_prediction,
                model,
                task,
                args
            )
            futures.append((future, i, task))

        for future, i, task in tqdm(futures, desc="Generating predictions"):
            while True:
                try:
                    prediction = future.result()
                    predictions.append(prediction)
                    break
                except Exception as e:
                    raise RuntimeError(f"Error generating prediction {i}: {str(e)}")

    return predictions


def process_evaluation(predictions, tasks, args) -> List[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=args.eval_threads) as executor:
        futures = []

        for i, task in enumerate(tasks):
            prediction = predictions[i]

            future = executor.submit(
                evaluate_prediction,
                prediction,
                task,
                args
            )
            futures.append((future, i, prediction, task))

        for future, i, prediction, task in tqdm(futures, desc="Evaluating predictions"):
            try:
                accuracy = future.result()
                result = {
                    "id": task["id"],
                    "question": task["question"],
                    "answer": task["answer"],
                    "dataset": task["dataset"],
                    "source": task["source"],
                    "prediction": prediction,
                    "accuracy": accuracy,
                    "correct": accuracy > 0,
                }
                results.append(result)
            except Exception as e:
                raise RuntimeError(f"Error evaluating prediction {i}: {str(e)}")

    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    if not results:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / float(total)
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "sub_metrics": {},
    }

    for result in results:
        if result["source"] not in metrics["sub_metrics"]:
            metrics["sub_metrics"][result["source"]] = {"correct": 0, "total": 0}
        if result["correct"]:
            metrics["sub_metrics"][result["source"]]["correct"] += 1
        metrics["sub_metrics"][result["source"]]["total"] += 1

    for source in metrics["sub_metrics"]:
        metrics["sub_metrics"][source]["accuracy"] = metrics["sub_metrics"][source]["correct"] / float(metrics["sub_metrics"][source]["total"])

    return metrics
