import re
from typing import Optional
import numpy as np
from math_verify import parse, verify


def numeric_or_symbolic_correctness(prediction, answer):
    if prediction is None or answer is None:
        return False
    if prediction == answer:
        return True
    # order is important
    return verify(parse('$' + answer + '$'), parse('$' + prediction + '$'))


def find_boxed_content(s, last_occurrence):
    pattern = r'\\boxed\{'
    all_matches = [m.end() for m in re.finditer(pattern, s)]
    if len(all_matches) == 0:
        return None
    if last_occurrence:
        start = all_matches[-1]
    else:
        start = all_matches[0]
    stack = 1
    i = start
    while i < len(s) and stack > 0:
        if s[i] == '{':
            stack += 1
        elif s[i] == '}':
            stack -= 1
        i += 1
    if stack == 0:
        return s[start:i - 1]  # Return the content inside the braces

    return None


def extract_between_and_with_boxes(
        x: str,
        last_occurrence: bool = False,
) -> str:
    """Extracts the boxed or delimited answer, returning empty string otherwise."""
    boxed_answer = find_boxed_content(x, last_occurrence=last_occurrence)
    if boxed_answer is not None:
        return boxed_answer
    else:
        return ''


def split_answer_separator(text: str, separator: Optional[str] = None) -> str:
    """Takes the first part of answer given a separator."""
    text = text.strip()
    if separator is not None:
        text = text.split(separator)[0].strip()
    return text


def equivalence_partition(iterable, relation):
    classes = []
    for obj in iterable:  # for each object
        # find the class it is in
        if obj is None:
            classes.append([obj])
            continue
        found = False
        for cl in classes:
            if not cl[0]:  # modification
                continue
            if relation(cl[0], obj):  # is it equivalent to this class?
                cl.append(obj)
                found = True
                break
        if not found:  # it is in a new class
            classes.append([obj])
    return classes


def compute_majority_vote_correct(processed_predictions, predictions_correctness, predictions_partition, strict_tie_breaking=True, partition_weights=None):
    max_weight = 0
    majority_answer = None
    all_majority_answers = []
    multiple_majority_answers = False  # are there >1 most-popular answers?
    for partition_index, equivalence_class in enumerate(predictions_partition):
        if not equivalence_class[0]:
            # Ignore empty strings, None, etc., corresponding to the model
            # failing to arrive at a final answer
            continue
        if partition_weights is None:
            current_partition_weight = len(equivalence_class)
        else:
            current_partition_weight = np.sum(partition_weights[partition_index])
        if current_partition_weight > max_weight:
            max_weight = current_partition_weight
            majority_answer = equivalence_class[0]
            multiple_majority_answers = False
            all_majority_answers = [majority_answer]
        elif current_partition_weight == max_weight:
            multiple_majority_answers = True
            all_majority_answers.append(equivalence_class[0])

    if multiple_majority_answers:
        # strict handling of draws (ties); see function docstring above.
        if strict_tie_breaking:
            return False
        else:
            majority_answer = np.random.choice(all_majority_answers)
    if not majority_answer:
        # No majority answer was found, which could occur if all answers are
        # None, empty string, etc.
        return False
    majority_idx = processed_predictions.index(majority_answer)
    return predictions_correctness[majority_idx]


def process_sample(sample, few_shot_separator, extract_last_occurrence):
    # few_shot_separator used to prevent model hallucinating new problems
    # extract last occurrence should be turned on in most cases
    if sample is None:
        return None
    sample = split_answer_separator(sample, few_shot_separator)
    sample = extract_between_and_with_boxes(sample, extract_last_occurrence)
    return sample


def sample_match_strict(sample, reference):
    return sample == reference


def quick_evaluate_single(dataset_type, solution_or_answer, few_shot_separator, extract_last_occurrence, match_fn, raw_prediction):
    if dataset_type == 'MATH':
        answer_processed = process_sample(solution_or_answer, few_shot_separator, extract_last_occurrence)
    else:
        answer_processed = solution_or_answer
    prediction_processed = process_sample(raw_prediction, few_shot_separator, extract_last_occurrence)
    prediction_correctness = match_fn(prediction_processed, answer_processed)
    return prediction_correctness


def evaluate_preference(dataset_type, solution_or_answer, few_shot_separator, extract_last_occurrence, match_fn, raw_prediction1, raw_prediction2, soft_pref):
    if dataset_type == 'MATH':
        answer_processed = process_sample(solution_or_answer, few_shot_separator, extract_last_occurrence)
    else:
        answer_processed = solution_or_answer
    prediction1_processed = process_sample(raw_prediction1, few_shot_separator, extract_last_occurrence)
    prediction1_correctness = match_fn(prediction1_processed, answer_processed)
    prediction2_processed = process_sample(raw_prediction2, few_shot_separator, extract_last_occurrence)
    prediction2_correctness = match_fn(prediction2_processed, answer_processed)
    if prediction1_correctness != prediction2_correctness:
        return prediction1_correctness > prediction2_correctness
    else:
        return soft_pref
    
