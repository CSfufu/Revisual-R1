# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Dict, List
from collections import Counter

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict: str) -> float:
    """
    Checks if answer follows the required think-then-answer format.
    Expects reasoning in <think>...</think> tags followed by a clear answer.
    
    Args:
        predict: The model's response to evaluate
        
    Returns:
        float: 1.0 if format is correct, 0.0 otherwise
    """
    # Allow different possible formats with both thinking and clear answer
    patterns = [
        # Standard think tag followed by boxed answer
        re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL),
        # Think tag followed by an answer tag
        re.compile(r"<think>.*</think>\s*<answer>.*</answer>", re.DOTALL),
        # Think tag followed by answer clearly labeled
        re.compile(r"<think>.*</think>\s*(?:answer|solution)\s*[:=]\s*.*", re.DOTALL),
        # Strict format for mathematical solutions
        re.compile(r"<think>.*</think>.*(?:Therefore|Thus|So|Hence),?\s+.*=.*", re.DOTALL)
    ]
    
    for pattern in patterns:
        if pattern.fullmatch(predict):
            return 1.0
            
    return 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    Evaluates if the final answer matches the ground truth.
    Extracts the answer from various formats.
    
    Args:
        predict: The model's response
        ground_truth: The correct answer
        
    Returns:
        float: 1.0 if answer is correct, 0.0 otherwise
    """
    # First try to extract from boxed content
    answer = extract_boxed_content(predict)
    
    # If no boxed content, try extracting from answer tags
    if not answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
    
    # If no tags, look for final conclusion with "answer is" or similar
    if not answer:
        conclusion_patterns = [
            r"(?:Therefore|Thus|So|Hence),?\s+(?:the\s+)?(?:answer|solution|result)\s+(?:is|=)\s+(.*?)(?:\.|$)",
            r"(?:answer|solution|result)\s*(?::|=)\s*(.*?)(?:\.|$)",
            r"(?:Finally|In conclusion),\s+(?:we\s+(?:get|have|find))?\s*(.*?)(?:\.|$)"
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, predict, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                break
                
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def self_critic_reward(predict_str: str) -> float:
    """
    Calculates an optimized reward score for self-reflective content in mathematical reasoning,
    measuring the quality and impact of self-critique leading to improved solutions.

    Args:
        predict_str: The mathematical reasoning text to analyze

    Returns:
        float: A score between 0.0 and 1.0 representing the quality of self-reflection.
               Higher values indicate more meaningful self-correction patterns.
    """
    import re
    from collections import Counter
    
    # Skip empty inputs
    if not predict_str or len(predict_str.strip()) < 10:
        return 0.0
        
    # --- Configuration with refined weights ---
    # Categorized by strength of self-reflection
    reflection_categories = {
        # Strong self-critique (highest value)
        "strong_critique": {
            r"\bmistake\b": 0.12,
            r"\berror\b": 0.12,
            r"\bincorrect\b": 0.12,
            r"\bwrong\b": 0.10,
            r"\bI was mistaken\b": 0.15,
            r"\bThat's not right\b": 0.15,
            r"\bThis contradicts\b": 0.15,
            r"\bI need to reconsider\b": 0.15,
            r"\bthis doesn't make sense\b": 0.14,
            r"\bI made an error\b": 0.15,
            r"\bcontradiction\b": 0.13,
            r"\bflaw\b": 0.12,
            r"\binvalid\b": 0.12,
        },
        
        # Reconsideration indicators (medium-high value)
        "reconsideration": {
            r"\blet me recalculate\b": 0.10,
            r"\blet's revise\b": 0.10,
            r"\blet me reconsider\b": 0.10,
            r"\blet's rethink\b": 0.10,
            r"\blet me check again\b": 0.09,
            r"\bdouble-check\b": 0.09,
            r"\bverify\b": 0.08,
            r"\bre-evaluate\b": 0.10,
            r"\breassess\b": 0.10,
            r"\brevisit\b": 0.08,
            r"\blet's try again\b": 0.09,
        },
        
        # Doubt/uncertainty indicators (medium value)
        "doubt": {
            r"\bwait\b": 0.06,
            r"\bhold on\b": 0.06,
            r"\bI doubt\b": 0.07,
            r"\bI question\b": 0.07,
            r"\bhowever\b": 0.05,
            r"\bbut\b": 0.04, 
            r"\balthough\b": 0.04,
            r"\bon second thought\b": 0.08,
            r"\bwait a second\b": 0.07,
            r"\bis that right\b": 0.08,
            r"\bdid I miss something\b": 0.08,
        },
        
        # Mild reflection (lower value)
        "mild_reflection": {
            r"\bperhaps\b": 0.03,
            r"\bmaybe\b": 0.03,
            r"\bI think\b": 0.02,
            r"\bI wonder\b": 0.04,
            r"\balternatively\b": 0.05,
            r"\bon the other hand\b": 0.05,
            r"\bI'm not sure\b": 0.05,
        }
    }
    
    # Flattened dictionary for simplicity in calculation
    all_patterns = {}
    for category, patterns in reflection_categories.items():
        all_patterns.update(patterns)

    # --- Sequence patterns ---
    # Patterns that indicate the start of self-reflection
    reflection_starter_patterns = [
        # Strong critique starters
        r"(?:That|This|It|The result) (?:is|seems|appears to be) (?:incorrect|wrong|mistaken|an error|a mistake|flawed)",
        r"(?:I|We) (?:made|committed) (?:an error|a mistake|a miscalculation)",
        r"(?:I|We) (?:was|were) (?:wrong|mistaken|incorrect|in error)",
        r"(?:Wait|Hold on|Actually|However|But|Oh)(?:,)? (?:that|this|it) (?:doesn't|does not) (?:seem|look|appear) (?:right|correct|accurate)",
        
        # Reconsideration starters
        r"(?:Let|I should|I need to) (?:me|) (?:reconsider|recalculate|rethink|revise|check|verify|double-check)",
        r"(?:Let's|Let us) (?:try again|recalculate|rethink|revise|check|verify|double-check)",
        
        # Doubt/uncertainty starters 
        r"(?:Wait|Hold on|Hmm|Actually|But|However)(?:,)? (?:I'm|I am|I) (?:not sure|uncertain|confused|doubtful|unsure)",
        r"(?:Wait|Hold on|Hmm|Actually|But|However)(?:,)? (?:this|that|it) (?:might|may|could) (?:be|not be) (?:right|correct|accurate)",
    ]
    
    # Patterns that indicate meaningful correction after reflection
    progress_followup_patterns = [
        # Explicit correction markers
        r"(?:The|A) correct (?:answer|solution|approach|calculation|value|result) (?:is|should be)",
        r"(?:I|We) should (?:have|instead) (?:calculated|computed|determined|found|used)",
        r"(?:Actually|In fact|Instead|Rather|Correctly)(?:,)? (?:the|we|I) (?:should|need to|must|have to)",
        r"(?:After|Upon) (?:reconsideration|recalculation|checking|verifying)(?:,)? (?:I|we) (?:see|realize|find|determine)",
        
        # Solution improvement markers
        r"(?:Now|So|Therefore|Thus)(?:,)? (?:the|a) (?:correct|accurate|proper|right) (?:approach|calculation|solution|answer)",
        r"(?:The|A) (?:error|mistake|issue|problem|flaw) (?:was|is) (?:that|because|due to)",
        r"(?:I|We) (?:missed|overlooked|forgot|didn't consider|failed to account for)",
        r"(?:This|That) (?:leads to|gives|results in|produces) (?:a|the) (?:correct|accurate|better|improved|proper)",
    ]
    
    # --- Scoring configuration ---
    sequence_bonus = 0.25  # Increased bonus for meaningful sequences
    sequence_window = 3    # Slightly increased window for followup detection
    max_sequence_count = 3 # Cap on how many sequences can contribute to score
    max_total_score = 1.0  # Maximum possible reward
    
    # --- Text preprocessing ---
    # Clean the input text: normalize whitespace, replace repeated newlines with periods
    cleaned_text = re.sub(r'\s+', ' ', predict_str)
    cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)  # Replace ellipses with single period
    
    # Split into sentences more robustly
    # Handle multiple punctuation, edge cases with quotes, etc.
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9])'
    sentences = re.split(sentence_pattern, cleaned_text)
    
    # --- Calculation ---
    # 1. Calculate Weighted Keyword Score with deduplication penalty
    keyword_score = 0.0
    text_lower = cleaned_text.lower()
    
    # Track keyword occurrences to penalize repetition
    keyword_counts = Counter()
    
    for pattern, weight in all_patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        count = len(matches)
        
        if count > 0:
            # Apply diminishing returns for repeated keywords
            if count == 1:
                keyword_score += weight
            else:
                # Diminishing returns formula: first occurrence gets full weight, 
                # subsequent ones get progressively less
                keyword_score += weight * (1 + 0.5 * (count - 1))
                
            keyword_counts[pattern] = count
    
    # 2. Calculate Sequence Score with improved detection
    sequence_score = 0.0
    used_starter_indices = set()
    sequence_count = 0
    
    for i in range(len(sentences)):
        # Skip very short sentences as they're unlikely to be meaningful reflections
        if len(sentences[i]) < 5:
            continue
            
        sentence_i = sentences[i]
        is_starter = any(re.search(pattern, sentence_i, re.IGNORECASE) for pattern in reflection_starter_patterns)
        
        if is_starter and i not in used_starter_indices:
            # Look for a followup in subsequent sentences within the window
            for j in range(i + 1, min(i + 1 + sequence_window, len(sentences))):
                sentence_j = sentences[j]
                is_followup = any(re.search(pattern, sentence_j, re.IGNORECASE) for pattern in progress_followup_patterns)
                
                if is_followup:
                    sequence_score += sequence_bonus
                    used_starter_indices.add(i)
                    sequence_count += 1
                    
                    # Cap the number of sequences that contribute to score
                    if sequence_count >= max_sequence_count:
                        break
                        
                    break  # Found a sequence, move to next potential starter
            
            # If we've reached the max count, break the outer loop too
            if sequence_count >= max_sequence_count:
                break
    
    # 3. Analyze overall document structure for reflection pattern
    # Check if reflection appears in second half of document (higher quality reflection)
    total_sentences = len(sentences)
    if total_sentences >= 4:  # Only meaningful for longer texts
        second_half_start = total_sentences // 2
        
        # Check if most self-reflection occurs in the second half (common pattern in good reasoning)
        first_half_keywords = 0
        second_half_keywords = 0
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for pattern in all_patterns if re.search(pattern, sentence_lower, re.IGNORECASE))
            
            if i < second_half_start:
                first_half_keywords += keyword_count
            else:
                second_half_keywords += keyword_count
        
        # If most reflection is in second half, apply a small bonus (structure bonus)
        if second_half_keywords > first_half_keywords:
            structure_bonus = 0.1
        else:
            structure_bonus = 0
    else:
        structure_bonus = 0
    
    # --- Combine and apply adjustments ---
    # Base score from keywords and sequences
    total_score = keyword_score + sequence_score
    
    # Add structure bonus
    total_score += structure_bonus
    
    # Ensure score is between 0 and max_total_score
    total_score = min(max_total_score, max(0.0, total_score))
    
    # Debug info can be uncommented if needed
    # print(f"Keyword score: {keyword_score}, Sequence score: {sequence_score}, Structure bonus: {structure_bonus}")
    # print(f"Final score: {total_score}")
    
    return total_score


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
        r"Wait, maybe.*?Wait, maybe",
    ]
    
    circular_count = sum(len(re.findall(pattern, predict_str, re.IGNORECASE)) for pattern in circular_patterns)
    
    maybe_but_wait_count = len(re.findall(r"maybe\s+(.*?)\.\s+But\s+(?:wait|actually).*?maybe\s+\1", predict_str, re.IGNORECASE))
    
    sentence_repetition_penalty = min(0.5, 0.1 * exact_duplicates)
    phrase_repetition_penalty = min(0.3, 0.05 * phrase_repetition)
    circular_reasoning_penalty = min(0.2, 0.1 * circular_count + 0.2 * maybe_but_wait_count)
    
    total_penalty = sentence_repetition_penalty + phrase_repetition_penalty + circular_reasoning_penalty
    
    return max(0.0, 1.0 - total_penalty)

def length_reward_l1_max(
    n_y: int,
    n_gold: int,
    alpha: float = 0.001,
    delta: float = 0.95,
) -> float:
    """
    L1-Max length reward.
    n_y    : 生成序列 token 数
    n_gold : 预算上限
    """
    raw = alpha * (n_gold - n_y) + delta
    return max(0.0, min(1.0, raw))  # clip 到 [0,1]



def compute_score(predicts: List[str], ground_truths: List[str], use_efficient: bool, n_gold: int, format_weight: float = 0.05) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        reward = accuracy_reward(predict, ground_truth)
        penalty_reward = repetition_penalty_reward(predict)
        self_critic_score = self_critic_reward(predict)
        format_score = format_reward(predict)
        scores.append(
            {
                "overall": reward * (1 - format_weight * 3) + penalty_reward * format_weight * 2 + self_critic_score * format_weight,
                "format": format_score,
                "self_critic": self_critic_score,
                "repetition_score": penalty_reward,
                "accuracy": reward,
            }
        )

    return scores