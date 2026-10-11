"""Author a tiny curriculum for pipeline checks, not general language training."""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/demo-data')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Choose an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    # Supervised delayed recall, language completion and small arithmetic.
    for a in range(20):
        for b in range(10):
            examples.append({'messages': [{'role': 'user', 'content': f'What is {a} + {b}?'},
                                           {'role': 'assistant', 'content': str(a + b)}]})
    for noun in ('cat', 'dog', 'bird', 'fish', 'horse', 'fox'):
        for color in ('red', 'blue', 'green', 'yellow', 'white', 'black'):
            examples.append({'messages': [{'role': 'user', 'content': f'Remember: the {noun} is {color}.'},
                                           {'role': 'assistant', 'content': 'I will remember.'},
                                           {'role': 'user', 'content': f'What color is the {noun}?'},
                                           {'role': 'assistant', 'content': f'The {noun} is {color}.'}]})
    random.Random(42).shuffle(examples)
    for name, rows in (('train', examples[30:]), ('validation', examples[:30])):
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    # An easier next-symbol sanity check, with distinct held-out documents.
    for name, repeats in (('sequence-train', range(6, 16)), ('sequence-validation', range(16, 19))):
        rows = [{'text': 'abcdefghijklmnopqrstuvwxyz ' * count} for count in repeats]
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(output)


if __name__ == '__main__':
    main()
