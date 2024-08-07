from collections import defaultdict


def count_alignments(alignments, expected_alignments):
    counts = {
        "expected": 0,
        "unexpected": 0,
        "unique": 0,
        "total": 0,
    }

    for i, alignment in enumerate(alignments):
        counts["total"] += 1

        if alignment in expected_alignments:
            counts["expected"] += 1
        else:
            counts["unexpected"] += 1

        if alignment not in alignments[:i]:
            counts["unique"] += 1

    return counts


def sort_alignments(alignments, expected_alignments):
    sorted_alignments = defaultdict(list)

    for i, alignment in enumerate(alignments):

        if alignment in expected_alignments:
            sorted_alignments["expected"].append(alignment)
        else:
            sorted_alignments["unexpected"].append(alignment)

    for alignment in expected_alignments:
        if alignment not in alignments:
            sorted_alignments["missing"].append(alignment)

    return sorted_alignments


def display_results(alignments, expected_alignments):
    counts = count_alignments(
        alignments=alignments,
        expected_alignments=expected_alignments,
    )
    sorts = sort_alignments(
        alignments=alignments,
        expected_alignments=expected_alignments,
    )

    def print_pretty(*args):
        print(*args, sep="\n", end="\n\n")

    print_pretty("🧛 COUNTS:", counts)
    print_pretty("✅ EXPECTED ALIGNMENTS:", *sorts["expected"])
    print_pretty("⛔️ UNEXPECTED ALIGNMENTS:", *sorts["unexpected"])
    print_pretty("👻 MISSING ALIGNMENTS:", *sorts["missing"])
