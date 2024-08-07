from typing import List, Tuple

from alignment import score_quality, example

AlignmentType = Tuple[List[str], List[str]]


def align_sequences(sequence_1: List[str], sequence_2: List[str]) -> List[AlignmentType]:
    alignments = []
    for string_1, string_2 in zip(sequence_1, sequence_2):
        alignments.append(([string_1], [string_2]))
    return alignments


if __name__ == '__main__':
    score_quality.display_results(
        alignments=align_sequences(example.sequence_1, example.sequence_2),
        expected_alignments=example.expected_alignments,
    )
