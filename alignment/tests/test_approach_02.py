from alignment.approaches.approach_02 import Span, SpanAlignment, suggest_potential_span_alignments


class TestSpan:

    def test_lt_gt(self):
        span_1 = Span(start=0, end=1)
        assert span_1 == span_1

        span_2 = Span(start=0, end=2)
        assert span_1 < span_2
        assert span_2 > span_1

        span_3 = Span(start=1, end=2)
        assert span_1 < span_3
        assert span_3 > span_1
        assert span_2 < span_3
        assert span_3 > span_2

    def test_overlap(self):
        span_1 = Span(0, 1)
        span_2 = Span(1, 2)
        assert not span_1.overlaps(span_2)
        assert not span_2.overlaps(span_1)

        span_1 = Span(0, 3)
        span_2 = Span(1, 2)
        assert span_1.overlaps(span_2)
        assert span_2.overlaps(span_1)

        span_1 = Span(0, 2)
        span_2 = Span(1, 3)
        assert span_1.overlaps(span_2)
        assert span_2.overlaps(span_1)

    def test_is_contiguous(self):
        span_1 = Span(0, 1)
        span_2 = Span(1, 2)
        assert span_1.is_contiguous(span_2)
        assert span_2.is_contiguous(span_1)

        span_1 = Span(0, 3)
        span_2 = Span(1, 2)
        assert not span_1.is_contiguous(span_2)
        assert not span_2.is_contiguous(span_1)

        span_1 = Span(0, 2)
        span_2 = Span(1, 3)
        assert not span_1.is_contiguous(span_2)
        assert not span_2.is_contiguous(span_1)

    def test_slice(self):
        span = Span(0, 3)
        expected_slices = [
            Span(0, 1),
            Span(0, 2),
            Span(0, 3),
            Span(1, 2),
            Span(1, 3),
            Span(2, 3),
        ]
        slices = list(span.slice())
        assert slices == expected_slices


class TestSpanAlignment:

    def test_lt_gt(self):
        span_alignment_1 = SpanAlignment(span_1=Span(start=0, end=1), span_2=Span(start=0, end=1), score=None)
        assert span_alignment_1 == span_alignment_1

        span_alignment_2 = SpanAlignment(span_1=Span(start=0, end=2), span_2=Span(start=0, end=1), score=None)
        assert span_alignment_1 < span_alignment_2
        assert span_alignment_2 > span_alignment_1

    def test_is_contiguous(self):
        span_1 = SpanAlignment(Span(0, 1), Span(0, 1))
        span_2 = SpanAlignment(Span(1, 2), Span(1, 2))
        span_3 = SpanAlignment(Span(2, 3), Span(2, 3))

        # spans are not contiguous with themselves
        assert not span_1.is_contiguous(span_1)
        assert not span_2.is_contiguous(span_2)
        assert not span_3.is_contiguous(span_3)

        # contiguous spans are contiguous regardless of order
        assert span_1.is_contiguous(span_2)
        assert span_2.is_contiguous(span_1)
        assert span_2.is_contiguous(span_3)
        assert span_3.is_contiguous(span_2)

        # non-contiguous spans are not contiguous regardless of order
        assert not span_1.is_contiguous(span_3)
        assert not span_3.is_contiguous(span_1)

    def test_slice(self):
        span_alignment = SpanAlignment(Span(0, 3), Span(0, 3))
        expected_slices_of_one_span = [
            Span(0, 1),
            Span(0, 2),
            Span(0, 3),
            Span(1, 2),
            Span(1, 3),
            Span(2, 3),
        ]
        expected_slices_of_span_alignment = [
            SpanAlignment(span_1, span_2)
            for span_1 in expected_slices_of_one_span
            for span_2 in expected_slices_of_one_span
        ]
        slices_of_span_alignment = list(span_alignment.slice())
        assert slices_of_span_alignment == expected_slices_of_span_alignment

    def test_slice_only_generates_unique_slices(self):
        span_alignment = SpanAlignment(Span(0, 10), Span(0, 10))
        slice_coordinates = [
            (s.span_1.start, s.span_1.end, s.span_2.start, s.span_2.end)
            for s in span_alignment.slice()
        ]
        assert len(slice_coordinates) == len(set(slice_coordinates))



class TestSuggestPotentialSpanAlignments:

    def test_no_suggestions(self):
        length_sequence_1 = length_sequence_2 = 1
        initial_span_alignments = [
            SpanAlignment(Span(0, 1), Span(0, 1)),
        ]

        expected_suggestions = []

        suggestions = suggest_potential_span_alignments(initial_span_alignments, length_sequence_1, length_sequence_2)
        assert suggestions == expected_suggestions

    def test_simple_leading_suggestions(self):
        length_sequence_1 = 2
        length_sequence_2 = 1
        initial_span_alignments = [
            SpanAlignment(Span(1, 2), Span(0, 1)),
        ]

        expected_suggestions = [
            SpanAlignment(Span(0, 2), Span(0, 1)),
            SpanAlignment(Span(0, 1), Span(0, 1)),
        ]

        suggestions = suggest_potential_span_alignments(initial_span_alignments, length_sequence_1, length_sequence_2)
        assert sorted(suggestions) == sorted(expected_suggestions)

    def test_simple_trailing_suggestions(self):
        length_sequence_1 = 2
        length_sequence_2 = 1
        initial_span_alignments = [
            SpanAlignment(Span(0, 1), Span(0, 1)),
        ]

        expected_suggestions = [
            SpanAlignment(Span(0, 2), Span(0, 1)),
            SpanAlignment(Span(1, 2), Span(0, 1)),
        ]

        suggestions = suggest_potential_span_alignments(initial_span_alignments, length_sequence_1, length_sequence_2)
        assert sorted(suggestions) == sorted(expected_suggestions)

    def test_sequence_1_middle_suggestions(self):
        length_sequence_1 = 3
        length_sequence_2 = 2
        initial_span_alignments = [
            SpanAlignment(Span(0, 1), Span(0, 1)),
            SpanAlignment(Span(2, 3), Span(1, 2)),
        ]

        expected_suggestions = [
            i
            for i in SpanAlignment(Span(0, 3), Span(0, 2)).slice()
            if i not in initial_span_alignments
        ]

        suggestions = suggest_potential_span_alignments(initial_span_alignments, length_sequence_1, length_sequence_2)
        assert sorted(suggestions) == sorted(expected_suggestions)

    def test_sequence_2_middle_suggestions(self):
        length_sequence_1 = 2
        length_sequence_2 = 3
        initial_span_alignments = [
            SpanAlignment(Span(0, 1), Span(0, 1)),
            SpanAlignment(Span(1, 2), Span(2, 3)),
        ]

        expected_suggestions = [
            i
            for i in SpanAlignment(Span(0, 2), Span(0, 3)).slice()
            if i not in initial_span_alignments
        ]

        suggestions = suggest_potential_span_alignments(initial_span_alignments, length_sequence_1, length_sequence_2)
        assert sorted(suggestions) == sorted(expected_suggestions)
