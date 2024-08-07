import dataclasses
from typing import List, Tuple, Optional, Iterator, Dict, Callable

import numpy as np
from fuzzywuzzy import fuzz

from alignment import score_quality, example

StringAlignment = Tuple[List[str], List[str]]


def generate_score_matrix(
        sequence_1: List[str],
        sequence_2: List[str],
        weights: Dict[str, float],
) -> np.array:
    combination_functions: Dict[str, Callable[[List[str], List[str]], np.array]] = {
        "fuzz": generate_fuzz_score_matrix,
    }

    update_functions: Dict[str, Callable[[np.array, float], np.array]] = {
        "distance": update_score_matrix_with_distance,
    }

    matrix_function_names = set(combination_functions) | set(update_functions)
    invalid_function_names = set(weights) - matrix_function_names
    if invalid_function_names:
        raise ValueError(
            f"Invalid weight(s) provided: {list(invalid_function_names)}.\n"
            f"Accepted values are: {list(matrix_function_names)}"
        )

    matrix = np.zeros((len(sequence_1), len(sequence_2)))

    for name, function in combination_functions.items():
        weight = weights.get(name, 0)
        if weight == 0:
            continue
        matrix += weight * function(sequence_1, sequence_2)

    for name, function in update_functions.items():
        weight = weights.get(name, 0)
        if weight == 0:
            continue
        matrix = function(matrix, weight)

    return matrix


def generate_fuzz_score_matrix(sequence_1: List[str], sequence_2: List[str]) -> np.array:
    score_matrix = np.zeros((len(sequence_1), len(sequence_2)))
    for i, string_1 in enumerate(sequence_1):
        for j, string_2 in enumerate(sequence_2):
            score_matrix[i, j] = fuzz.ratio(string_1, string_2) / 100
    return score_matrix


def update_score_matrix_with_distance(score_matrix: np.array, weight: float) -> np.array:
    score_matrix = score_matrix.copy()

    expected_column_index_with_max_score = 0
    for row_index, row_values in enumerate(score_matrix):
        for column_index, score in enumerate(row_values):
            penalty = weight * abs(column_index - expected_column_index_with_max_score)
            score_matrix[row_index, column_index] = max(score - penalty, 0)
        expected_column_index_with_max_score = row_values.argmax() + 1

    return score_matrix


@dataclasses.dataclass
class Span:
    start: int
    end: Optional[int] = None

    def __post_init__(self):
        if self.end is None:
            self.end = self.start + 1

    def slice(self) -> Iterator['Span']:
        for i in range(self.start, self.end + 1):
            for j in range(i + 1, self.end + 1):
                yield Span(i, j)

    def overlaps(self, other: 'Span') -> bool:
        return self.start < other.end and other.start < self.end

    def is_contiguous(self, other: 'Span') -> bool:
        if self == other:
            return False

        if other < self:
            return other.is_contiguous(self)

        return self.end == other.start

    def __eq__(self, other: 'Span') -> bool:
        return self.start == other.start and self.end == other.end

    def __lt__(self, other):
        if self == other:
            return False

        return self.start < other.start or (self.start == other.start and self.end < other.end)


@dataclasses.dataclass
class SpanAlignment:
    span_1: Span
    span_2: Span
    score: Optional[float] = None

    def overlaps(self, other: 'SpanAlignment') -> bool:
        return self.span_1.overlaps(other.span_1) or self.span_2.overlaps(other.span_2)

    def is_contiguous(self, other: 'SpanAlignment') -> bool:
        if self == other:
            return False

        if other < self:
            return other.is_contiguous(self)

        return self.span_1.is_contiguous(other.span_1) and self.span_2.is_contiguous(other.span_2)

    def slice(self) -> Iterator['SpanAlignment']:
        for span_1 in self.span_1.slice():
            for span_2 in self.span_2.slice():
                yield SpanAlignment(span_1, span_2)

    def __eq__(self, other: 'SpanAlignment') -> bool:
        return self.span_1 == other.span_1 and self.span_2 == other.span_2

    def __lt__(self, other: 'SpanAlignment') -> bool:
        if self == other:
            return False

        return self.span_1 < other.span_1 or (self.span_1 == other.span_1 and self.span_2 < other.span_2)


def choose_seed_span_alignments(score_matrix: np.array) -> List[SpanAlignment]:
    span_alignments = []
    for i, score_slice in enumerate(score_matrix):
        j = score_slice.argmax()
        span_alignment = SpanAlignment(span_1=Span(i), span_2=Span(j), score=score_slice[j])
        span_alignments.append(span_alignment)
    return span_alignments


def suggest_potential_span_alignments(
        span_alignments: List[SpanAlignment],
        length_sequence_1: int,
        length_sequence_2: int,
) -> List[SpanAlignment]:
    span_alignments = [
        SpanAlignment(Span(0, 0), Span(0, 0)),
        *sorted(span_alignments),
        SpanAlignment(Span(length_sequence_1 - 1), Span(length_sequence_2 - 1)),
    ]

    additional_span_alignments = []
    for index in range(1, len(span_alignments) - 1):
        span_alignment = span_alignments[index]
        previous_span_alignment = span_alignments[index - 1]
        next_span_alignment = span_alignments[index + 1]

        if previous_span_alignment.is_contiguous(span_alignment) and span_alignment.is_contiguous(next_span_alignment):
            continue

        new_parent_span_alignment = SpanAlignment(
            Span(previous_span_alignment.span_1.start, next_span_alignment.span_1.end),
            Span(previous_span_alignment.span_2.start, next_span_alignment.span_2.end),
        )
        for new_span_alignment in new_parent_span_alignment.slice():
            if new_span_alignment not in span_alignments[1:-1]:
                if new_span_alignment not in additional_span_alignments:
                    additional_span_alignments.append(new_span_alignment)
    return additional_span_alignments


def span_alignments_to_string_alignments(
        span_alignments: List[SpanAlignment],
        sequence_1: List[str],
        sequence_2: List[str],
) -> List[StringAlignment]:
    string_alignments = []
    for span_alignment in span_alignments:
        string_alignment = (
            sequence_1[span_alignment.span_1.start:span_alignment.span_1.end],
            sequence_2[span_alignment.span_2.start:span_alignment.span_2.end],
        )
        string_alignments.append(string_alignment)
    return string_alignments


def score_alignments(
        span_alignments: List[SpanAlignment],
        sequence_1: List[str],
        sequence_2: List[str],
        weights: Dict[str, float],
) -> List[SpanAlignment]:
    for span_alignment in span_alignments:
        # NOTE: this join method is a little lazy, and it would be better to extract the exact spacing from the
        # original strings
        string_1 = " ".join(sequence_1[span_alignment.span_1.start:span_alignment.span_1.end])
        string_2 = " ".join(sequence_2[span_alignment.span_2.start:span_alignment.span_2.end])
        score = generate_score_matrix([string_1], [string_2], weights)[0, 0]
        span_alignment.score = score
    return span_alignments


def choose_best_span_alignments(span_alignments: List[SpanAlignment]) -> List[SpanAlignment]:
    span_alignments = sorted(span_alignments, key=lambda i: i.score, reverse=True)
    chosen_alignments = []
    while span_alignments:
        chosen_alignment = span_alignments.pop(0)
        chosen_alignments.append(chosen_alignment)
        span_alignments = [
            i for i in span_alignments
            if not i.overlaps(chosen_alignment)
        ]
    return chosen_alignments


def align_sequences(
        sequence_1: List[str],
        sequence_2: List[str],
        seed_weights: Dict[str, float],
        improvement_weights: Dict[str, float],
) -> List[StringAlignment]:
    if not seed_weights:
        return []

    # choose seed span alignments
    seed_score_matrix = generate_score_matrix(sequence_1, sequence_2, seed_weights)
    seed_span_alignments = choose_seed_span_alignments(seed_score_matrix)

    if not improvement_weights:
        return span_alignments_to_string_alignments(seed_span_alignments, sequence_1, sequence_2)

    # identify potential improved span alignments
    suggested_span_alignments = suggest_potential_span_alignments(
        seed_span_alignments,
        len(sequence_1),
        len(sequence_2),
    )

    # re-score all span alignments
    all_span_alignments = score_alignments(
        suggested_span_alignments + seed_span_alignments,
        sequence_1,
        sequence_2,
        improvement_weights,
    )

    # choose best span alignments
    best_span_alignments = choose_best_span_alignments(all_span_alignments)

    return span_alignments_to_string_alignments(best_span_alignments, sequence_1, sequence_2)


if __name__ == '__main__':
    alignments = align_sequences(
        example.sequence_1,
        example.sequence_2,
        seed_weights={"fuzz": 1.0, "distance": 0.05},
        improvement_weights={"fuzz": 1.0},
    )
    score_quality.display_results(
        alignments=alignments,
        expected_alignments=example.expected_alignments,
    )
