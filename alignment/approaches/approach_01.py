from typing import List, Tuple

import numpy as np
from fuzzywuzzy import fuzz

from alignment import score_quality, example

AlignmentType = Tuple[List[str], List[str]]


def generate_score_matrix(sequence_1: List[str], sequence_2: List[str]) -> np.array:
    alignment_score_matrix = np.zeros((len(sequence_1), len(sequence_2)))
    for i, string_1 in enumerate(sequence_1):
        for j, string_2 in enumerate(sequence_2):
            alignment_score_matrix[i, j] = fuzz.ratio(string_1, string_2)
    return alignment_score_matrix


def choose_alignments(
        score_matrix: np.array,
        sequence_1: List[str],
        sequence_2: List[str],
) -> List[AlignmentType]:
    alignments = []
    for i in range(len(sequence_1)):
        score_slice = score_matrix[i, :]
        j = score_slice.argmax()
        alignment = ([sequence_1[i]], [sequence_2[j]])
        alignments.append(alignment)
    return alignments


def align_sequences(sequence_1: List[str], sequence_2: List[str]) -> List[AlignmentType]:
    score_matrix = generate_score_matrix(sequence_1, sequence_2)
    alignments = choose_alignments(score_matrix, sequence_1, sequence_2)
    return alignments


if __name__ == '__main__':
    score_quality.display_results(
        alignments=align_sequences(example.sequence_1, example.sequence_2),
        expected_alignments=example.expected_alignments,
    )
