import random
import verifiers as vf
from datasets import Dataset


def load_environment(
    num_examples: int = 50,
    max_number: int = 100,
    seed: int = 42,
    **kwargs,
) -> vf.Environment:
    """
    A simple math environment that tests addition and subtraction.

    Args:
        num_examples: Number of problems to generate
        max_number: Maximum number in problems
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Generate dataset
    examples = []
    for _ in range(num_examples):
        a = random.randint(1, max_number)
        b = random.randint(1, max_number)
        op = random.choice(["+", "-"])

        if op == "+":
            answer = a + b
        else:
            # Ensure non-negative results
            if a < b:
                a, b = b, a
            answer = a - b

        question = f"What is {a} {op} {b}? Reply with just the number."
        examples.append({
            "prompt": [{"role": "user", "content": question}],
            "answer": str(answer),
        })

    dataset = Dataset.from_list(examples)

    # Scoring function
    async def correct_answer(completion, answer) -> float:
        response = completion[-1]["content"].strip()
        # Extract number from response
        numbers = [s for s in response.split() if s.lstrip("-").isdigit()]
        if numbers:
            return 1.0 if numbers[-1] == answer else 0.0
        return 0.0

    rubric = vf.Rubric(funcs=[correct_answer])

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
    )
