from alignment import score_quality, example
from openai import AzureOpenAI
from typing import List, Tuple
from dotenv import load_dotenv
import os

load_dotenv()

StringAlignment = Tuple[List[str], List[str]]


def wrap_prompt_components(instruction, ex, assignment):
    return (
        f"INSTRUCTION:\n{instruction}\n\n"
        f"EXAMPLE:\n{ex}\n\n"
        f"ASSIGNMENT:\n{assignment}"
    )


def generate_simple_alignment_prompt(sequence_1: List[str], sequence_2: List[str]) -> str:
    instruction = "Generate sentence-level alignments from different versions of text."
    ex = (
        "Version 1:\n"
        "0: this is a sentence.\n"
        "1: this is another sentence.\n"
        "2: this is the last sentence.\n"
        "Version 2:\n"
        "0: This is a sentence.\n"
        "1: This is another sentence, and this is the last sentence.\n"
        "Alignments:\n"
        "this is a sentence. -> This is a sentence.\n"
        "this is another sentence. this is the last sentence. -> This is another sentence, and this is the last "
        "sentence."
    )
    assignment = (
        "Version 1:\n" +
        "\n".join(f"{i}: {sent}" for i, sent in enumerate(sequence_1)) +
        "\nVersion 2:\n" +
        "\n".join(f"{i}: {sent}" for i, sent in enumerate(sequence_2)) +
        "\nAlignments:"
    )
    return wrap_prompt_components(instruction, ex, assignment)


def generate_openai_request(prompt: str) -> str:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": prompt},
        ]
    )

    return response.choices[0].message.content


def parse_simple_alignment_response(response: str) -> List[StringAlignment]:
    string_alignments = []

    for line in response.split("\n"):
        string_1, string_2 = line.split(" -> ")
        string_alignments.append(([string_1], [string_2]))

    return string_alignments


def align_with_openai(sequence_1: List[str], sequence_2: List[str]) -> List[StringAlignment]:
    prompt = generate_simple_alignment_prompt(sequence_1, sequence_2)
    response = generate_openai_request(prompt)
    return parse_simple_alignment_response(response)


if __name__ == '__main__':
    alignments = align_with_openai(example.sequence_1, example.sequence_2)
    score_quality.display_results(
        alignments=alignments,
        expected_alignments=example.expected_alignments,
    )
