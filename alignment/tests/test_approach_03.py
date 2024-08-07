import numpy as np

from alignment.approaches.approach_03 import (
    generate_fuzz_score_matrix,
    generate_score_matrix,
    update_score_matrix_with_distance,
)


class TestFuzzScoreMatrix:
    def test_fuzz_score_matrix(self):
        sequence_1 = ["hello", "world"]
        sequence_2 = ["hollow", "word"]
        score_matrix = generate_fuzz_score_matrix(sequence_1, sequence_2)

        assert 0.5 < score_matrix[0, 0] < 1.0
        assert 0.0 < score_matrix[0, 1] < 0.5
        assert 0.0 < score_matrix[1, 0] < 0.5
        assert 0.5 < score_matrix[1, 1] < 1.0


class TestDistanceScoreMatrix:
    def test_distance_score_matrix(self):
        score_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        updated_score_matrix = update_score_matrix_with_distance(score_matrix, weight=1.0)
        assert (updated_score_matrix == score_matrix).all()

        score_matrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        updated_score_matrix = update_score_matrix_with_distance(score_matrix, weight=1.0)
        assert (updated_score_matrix != score_matrix).any()


class TestCombinedScoreMatrix:
    def test_generate_score_matrix(self):
        sequence_1 = ["hello", "world"]
        sequence_2 = ["hollow", "word"]
        weights = {"fuzz": 2.0}

        expected_score_matrix = (
            generate_fuzz_score_matrix(sequence_1, sequence_2) +
            generate_fuzz_score_matrix(sequence_1, sequence_2)
        )

        score_matrix = generate_score_matrix(sequence_1, sequence_2, weights)
        assert (score_matrix == expected_score_matrix).all()
