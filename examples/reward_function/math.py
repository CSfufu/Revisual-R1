# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def self_critic_reward(predict_str: str) -> float:
    """Reward occurrences of self‑reflective words such as 'but', 'wait', 'maybe', etc.

    Each occurrence contributes 0.05 points up to a maximum of 1.0.
    The keyword list can be extended as needed.
    """
    keywords = [
        "but", "wait", "maybe", "mistake", "However", "perhaps", "Let me check", "revise", "second thought", "Wait", "But", "perhaps", "That seems contradictory", "Alternatively", "On the other hand", "I think", "I guess", "I suppose", "I wonder", "I believe", "I feel", "I mean", "I assume", "I expect", "I hope", "I doubt", "I question", "I reflect", "I reconsider", "I re-evaluate", "mistake", "rethink", "reassess", "revisit", "recheck", "reconsider", "re-evaluate", "rethink", "reassess", "revisit", "recheck", "self-reflective", "self-reflection", "self-critique", "self-criticism", "self-analysis"
    ]
    count = sum(len(re.findall(rf"\b{kw}\b", predict_str, re.IGNORECASE)) for kw in keywords)
    return min(1.0, count * 0.05)

def image_view_reward(predict_str: str) -> float:
    """Reward evidence of referring to / analysing a diagram or image.

    The presence of each keyword (e.g. "image", "figure", "diagram") adds 0.1
    up to a maximum of 1.0. Designed for problems that include a picture and
    expect visual reasoning commentary.
    """
    keywords = [
        "image", "picture", "figure", "diagram", "shown", "as shown",
        "see the", "the graph", "the drawing", "look at",
    ]
    count = sum(len(re.findall(rf"{kw}", predict_str, re.IGNORECASE)) for kw in keywords)
    return min(1.0, count * 0.1)

def image_revisit_reward(predict_str: str) -> float:
    """
    Calculates a reward score for revisiting/referencing the image multiple times
    during mathematical reasoning, with special emphasis on going back to check
    the image after initial reasoning.
    
    Args:
        predict_str: The mathematical reasoning text to analyze
        
    Returns:
        float: A score between 0.0 and 1.0 representing the reward
    """
    # Patterns for initial image references
    initial_reference_patterns = [
        r"the image shows",
        r"according to the image", 
        r"the image illustrates",
        r"the image depicts",
        r"in the image",
        r"from the image",
        r"the diagram",
        r"the figure",
        r"as shown",
        r"is displayed",
    ]
    
    # Patterns for revisiting the image
    revisit_patterns = [
        r"looking (at|back at) the image again",
        r"upon (further|closer) inspection",
        r"revisiting the (image|figure|diagram)",
        r"taking another look",
        r"on second thought.{1,30}(image|figure|diagram)",
        r"wait.{1,30}(image|figure|diagram)",
        r"but.{1,30}(image|figure|diagram)",
        r"checking the (image|figure|diagram) again",
        r"let me look at the (image|figure|diagram) again",
        r"the (image|figure|diagram).{1,30}shows",
        r"no, the (image|figure|diagram)",
        r"if I look at the (image|figure|diagram) more carefully",
    ]
    
    # Patterns for insight after revisiting
    correction_patterns = [
        r"I (see|notice|realize|observe) that",
        r"actually",
        r"in fact",
        r"I was (wrong|mistaken|incorrect)",
        r"contrary to",
        r"unlike what I thought",
        r"(now|then) I (can|see|understand)",
        r"this (contradicts|confirms)",
        r"(now|then) it (is clear|becomes clear)",
        r"I missed that",
        r"I didn't notice",
        r"I overlooked",
    ]
    

    initial_ref_count = sum(len(re.findall(pattern, predict_str, re.IGNORECASE)) 
                          for pattern in initial_reference_patterns)
    
    revisit_count = sum(len(re.findall(pattern, predict_str, re.IGNORECASE)) 
                       for pattern in revisit_patterns)
    
    correction_count = sum(len(re.findall(pattern, predict_str, re.IGNORECASE)) 
                          for pattern in correction_patterns)
    

    sentences = re.split(r'(?<=[.!?])\s+', predict_str)
    sequence_count = 0
    
    for i in range(len(sentences) - 1):
        has_revisit = any(re.search(pattern, sentences[i], re.IGNORECASE) 
                         for pattern in revisit_patterns)
        has_correction = any(re.search(pattern, sentences[i+1], re.IGNORECASE) 
                           for pattern in correction_patterns)
        
        if has_revisit and has_correction:
            sequence_count += 1
    
    base_score = min(0.2 * initial_ref_count, 0.4)  # Up to 0.4 for initial references
    revisit_score = min(0.3 * revisit_count, 0.3)   # Up to 0.3 for revisits
    sequence_score = min(0.3 * sequence_count, 0.3) # Up to 0.3 for revisit+correction sequences
    total_score = min(base_score + revisit_score + sequence_score, 1.0)
    
    return total_score

def repetition_penalty_reward(predict_str: str) -> float:
    """
    Calculates a penalty score for repetitive content in the response.
    Higher repetition results in a lower score (penalty).
    
    The function detects:
    1. Repeated phrases and sentences
    2. Circular reasoning patterns
    3. Redundant statements with minimal variation
    
    Args:
        predict_str: The text to analyze for repetition
        
    Returns:
        float: A score between 0.0 and 1.0, where higher values indicate less repetition
    """
    import re
    from collections import Counter
    
    sentences = re.split(r'(?<=[.!?])\s+', predict_str)
    
    sentence_counter = Counter(sentences)
    exact_duplicates = sum(count - 1 for count in sentence_counter.values() if count > 1)
    
    words = predict_str.split()
    phrase_repetition = 0
    
    if len(words) >= 5:
        five_grams = [' '.join(words[i:i+5]) for i in range(len(words)-4)]
        five_gram_counter = Counter(five_grams)
        phrase_repetition = sum(count - 1 for count in five_gram_counter.values() if count > 1)
    
    circular_patterns = [
        r"(.*?)\s+But\s+(?:wait|actually|maybe|perhaps).*?\1",
        r"(.*?)\s+So\s+maybe\s+(?:.*?)\s+But\s+(?:wait|actually).*?\1",
        r"But the image (?:displays|illustrates|shows|depicts).*?But the image (?:displays|illustrates|shows|depicts)",
        r"So maybe.*?So maybe",
        r"Wait, but.*?Wait, but",
    ]
    
    circular_count = sum(len(re.findall(pattern, predict_str, re.IGNORECASE)) for pattern in circular_patterns)
    
    maybe_but_wait_count = len(re.findall(r"maybe\s+(.*?)\.\s+But\s+(?:wait|actually).*?maybe\s+\1", predict_str, re.IGNORECASE))
    
    sentence_repetition_penalty = min(0.5, 0.1 * exact_duplicates)
    phrase_repetition_penalty = min(0.3, 0.05 * phrase_repetition)
    circular_reasoning_penalty = min(0.2, 0.1 * circular_count + 0.2 * maybe_but_wait_count)
    
    total_penalty = sentence_repetition_penalty + phrase_repetition_penalty + circular_reasoning_penalty
    
    return max(0.0, 1.0 - total_penalty)


def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.05) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        self_critic_score = self_critic_reward(predict)
        image_revist_score = image_revisit_reward(predict)
        repetition_penalty_score = repetition_penalty_reward(predict)
        scores.append(
            {
                "overall": (1 - format_weight - format_weight - format_weight) * accuracy_score + format_weight * self_critic_score + format_weight * image_revist_score + format_weight * repetition_penalty_score,
                "format": (self_critic_score + image_revist_score + repetition_penalty_score) / 3,
                "accuracy": accuracy_score,
            }
        )

    return scores
