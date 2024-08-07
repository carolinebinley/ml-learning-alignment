sequence_1 = [
    "this is my first sentence",
    "this is my 2nd sentence",
    "this is my third sentence",
    "this is my fourth sentence",
    "this one will be deleted",
    "this is my fifth sentence",
    "this is my sixth sentence",
    "this is my seventh sentence, and this is my eighth sentence",
]

sequence_2 = [
    "This is my first sentence.",
    "This is my second sentence, and this is my third sentence.",
    "This is my fourth sentence.",
    "This is my sixth sentence.",
    "This is my fifth sentence.",
    "This is my seventh sentence.",
    "This is my eighth sentence.",
]

expected_alignments = [
    (
        [
            "this is my first sentence",
        ],
        [
            "This is my first sentence.",
        ],
    ),
    (
        [
            "this is my 2nd sentence",
            "this is my third sentence",
        ],
        [
            "This is my second sentence, and this is my third sentence.",
        ],
    ),
    (
        [
            "this is my fourth sentence",
        ],
        [
            "This is my fourth sentence.",
        ],
    ),
    (
        [
            "this is my fifth sentence",
        ],
        [
            "This is my fifth sentence.",
        ],
    ),
    (
        [
            "this is my sixth sentence",
        ],
        [
            "This is my sixth sentence.",
        ],
    ),
    (
        [
            "this is my seventh sentence, and this is my eighth sentence",
        ],
        [
            "This is my seventh sentence.",
            "This is my eighth sentence.",
        ],
    ),
]
