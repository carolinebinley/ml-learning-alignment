# README

This repository was created for a Clio ML learning session on August 7, 2024.

It contains example code for different approaches to the problem of **text
alignment**. Given different versions of the same text, how does one
algorithmically match a transformed chunk of the text across the different
versions?

These approaches range from the ultra-naive (`approach_00.py`) to the totally
overcooked (`approach_04.py`). None of them are production-ready. Some may
contain bugs, none are optimized for performance on large datasets, and all can
be improved. However, they illustrate core concepts and how the solution
complexity can grow.

## Running the code

Use `poetry` with python 3.11.9.

Within the `poetry shell`:

- Run the tests: `pytest ./alignment/tests/`
- Run a specific approach: `python ./alignment/approach_XX.py`
  - The script will use the approach to align the sequences provided in
    `example.py`
  - It will print out results showing where the approach succeeded and where it
    failed

## Extending the code

You can extend this project by increasing the complexity of the example
(`./alignments/example.py`) or by adding a new approach
(`./alignments/approach_XX.py`).

I recommend test-driven development for any new approach.
