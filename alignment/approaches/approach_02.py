import dataclasses
from typing import List, Tuple, Optional, Iterator

import numpy as np
from fuzzywuzzy import fuzz

from alignment import score_quality, example

StringAlignment = Tuple[List[str], List[str]]


def generate_score_matrix(sequence_1: List[str], sequence_2: List[str]) -> np.array:
    alignment_score_matrix = np.zeros((len(sequence_1), len(sequence_2)))
    for i, string_1 in enumerate(sequence_1):
        for j, string_2 in enumerate(sequence_2):
            alignment_score_matrix[i, j] = fuzz.ratio(string_1, string_2)
    return alignment_score_matrix


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


def choose_span_alignments(score_matrix: np.array) -> List[SpanAlignment]:
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
) -> List[SpanAlignment]:
    for span_alignment in span_alignments:
        # note: this join method is a little lazy
        string_1 = " ".join(sequence_1[span_alignment.span_1.start:span_alignment.span_1.end])
        string_2 = " ".join(sequence_2[span_alignment.span_2.start:span_alignment.span_2.end])
        span_alignment.score = fuzz.ratio(string_1, string_2)
    return span_alignments


def choose_final_span_alignments(span_alignments: List[SpanAlignment]) -> List[SpanAlignment]:
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


def align_sequences(sequence_1: List[str], sequence_2: List[str]) -> List[StringAlignment]:
    # choose seed span alignments
    seed_score_matrix = generate_score_matrix(sequence_1, sequence_2)
    seed_span_alignments = choose_span_alignments(seed_score_matrix)

    # identify potential improved span alignments
    suggested_span_alignments = suggest_potential_span_alignments(
        seed_span_alignments,
        len(sequence_1),
        len(sequence_2),
    )

    # score potential improved span alignments
    suggested_span_alignments = score_alignments(
        suggested_span_alignments,
        sequence_1,
        sequence_2,
    )

    # choose best span alignments
    final_span_alignments = choose_final_span_alignments(seed_span_alignments + suggested_span_alignments)

    return span_alignments_to_string_alignments(final_span_alignments, sequence_1, sequence_2)


if __name__ == '__main__':
    score_quality.display_results(
        alignments=align_sequences(example.sequence_1, example.sequence_2),
        expected_alignments=example.expected_alignments,
    )
