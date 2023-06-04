import outlines.text as text
import outlines.models as models
import openai
import re
import sympy
import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np
import backoff
from sqlitedict import SqliteDict
# print(openai.api_key)

## few-shot example data

input_output_examples = [
    {
        "input": "4 4 6 8",
        "output": "(4 + 8) * (6 - 4) = 24"
    },
    {
        "input": "2 9 10 12",
        "output": "2 * 12 * (10 - 9) = 24"
    },
    {
        "input": "4 9 10 13",
        "output": "(13 - 9) * (10 - 4) = 24"
    },
    {
        "input": "1 4 8 8",
        "output": "(8 / 4 + 1) * 8 = 24"
    },
    {
        "input": "5 5 5 9",
        "output": "5 + 5 + 5 + 9 = 24"
    },
]


cot_examples = [
    {
        "input": "4 4 6 8",
        "steps": [
            "4 + 8 = 12 (left: 4 6 12)",
            "6 - 4 = 2 (left: 2 12)",
            "2 * 12 = 24 (left: 24)"
        ],
        "output": "(6 - 4) * (4 + 8) = 24"
    },
    {
        "input": "2 9 10 12",
        "steps": [
            "12 * 2 = 24 (left: 9 10 24)",
            "10 - 9 = 1 (left: 1 24)",
            "24 * 1 = 24 (left: 24)"
        ],
        "output": "(12 * 2) * (10 - 9) = 24"
    },
    {
        "input": "4 9 10 13",
        "steps": [
            "13 - 10 = 3 (left: 3 4 9)",
            "9 - 3 = 6 (left: 4 6)",
            "4 * 6 = 24 (left: 24)"
        ],
        "output": "4 * (9 - (13 - 10)) = 24"
    },
    {
        "input": "1 4 8 8",
        "steps": [
            "8 / 4 = 2 (left: 1 2 8)",
            "1 + 2 = 3 (left: 3 8)",
            "3 * 8 = 24 (left: 24)"
        ],
        "output": "(1 + 8 / 4) * 8 = 24"
    },
    {
        "input": "5 5 5 9",
        "steps": [
            "5 + 5 = 10 (left: 5 9 10)",
            "10 + 5 = 15 (left: 9 15)",
            "15 + 9 = 24 (left: 24)"
        ],
        "output": "((5 + 5) + 5) + 9 = 24"
    },
]

# globals
completion_tokens = 0
prompt_tokens = 0

data = pd.read_csv("./data/24.csv")['Puzzles']
inputs = data[:10]  # 10 examples

# oai_model = models.text_completion.openai("gpt-4", max_tokens = 1000, temperature = 0.7)
oai_model = models.text_completion.openai(
    "gpt-3.5-turbo", max_tokens=1000, temperature=0.5)


## prompts
@text.prompt
def standard_prompt(input, examples=input_output_examples):
    '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
    {% for example in examples %}
    Input: {{ example.input }}
    Answer: {{ example.output }}
    {% endfor %}
    Input: {{input}}
    '''
    
@text.prompt
def cot_prompt(input, examples=cot_examples):
    '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
    {% for example in examples %}
    Input: {{ example.input }}
    Steps:
    {% for step in example.steps %}
    {{ step }}
    {% endfor %}
    Answer: {{ example.output }}
    {% endfor %}
    Input: {{input}}
    '''


def update_token_usage(res):
    '''
    Need this to keep track of token usage and cost
    '''
    global completion_tokens
    global prompt_tokens
    completion_tokens += res['usage']['completion_tokens']
    prompt_tokens += res['usage']['prompt_tokens']


def eval_output_game24(input: str, output: str) -> bool:
    '''
    Use regex to check if the output is correct for the given input
    '''
    expression = output.strip().split(
        '\n')[-1].lower().replace('answer: ', '').split('=')[0]
    numbers = re.findall(r'\d+', expression)
    problem_numbers = re.findall(r'\d+', input)
    if sorted(numbers) != sorted(problem_numbers):
        return False
    try:
        # print(sympy.simplify(expression))
        return sympy.simplify(expression) == 24
    except Exception as e:
        # print(e)
        return False


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def model(*args, **kwargs):
    return oai_model(*args, **kwargs)


def test_standard():
    '''
    Test the score using 5-shot standard prompting
    '''
    # constants
    n_generate = 5

    # generate all answers for all questions
    outputs = []
    for x in tqdm(inputs):
        prompt = standard_prompt(x)
        answers = model(prompt, sample=n_generate)
        outputs.append(answers)

    # check accuracy of the answers
    n_any_correct = 0
    avg_correct_arr = []
    for x, ys in zip(inputs, outputs):
        status = [eval_output_game24(x, y) for y in ys]
        if any(status):
            n_any_correct += 1

        avg_correct_arr.append(sum(status) / len(status))

    print("Standard Prompting\n------------------")
    print("n_inputs: ", len(inputs))
    print("n_inputs with atleast 1 correct answer: ", n_any_correct)
    print("avg number of correct answers per input: ",
          sum(avg_correct_arr) / len(avg_correct_arr))



def test_cot():
    '''
    Test the score using 5-shot cot prompting
    '''
    # constants
    n_generate = 5

    # generate all answers for all questions
    outputs = []
    for x in tqdm(inputs):
        prompt = cot_prompt(x)
        answers = model(prompt, sample=n_generate)
        outputs.append(answers)

    # check accuracy of the answers
    n_any_correct = 0
    avg_correct_arr = []
    for x, ys in zip(inputs, outputs):
        status = [eval_output_game24(x, y) for y in ys]
        if any(status):
            n_any_correct += 1

        avg_correct_arr.append(sum(status) / len(status))

    print("COT Prompting\n------------------")
    print("n_inputs: ", len(inputs))
    print("n_inputs with atleast 1 correct answer: ", n_any_correct)
    print("avg number of correct answers per input: ",
          sum(avg_correct_arr) / len(avg_correct_arr))



# tree of thought
propose_examples = [
    {
        "input": "2 8 8 14",
        "next_steps": [
            "2 + 8 = 10 (left: 8 10 14)",
            "8 / 2 = 4 (left: 4 8 14)",
            "14 + 2 = 16 (left: 8 8 16)",
            "2 * 8 = 16 (left: 8 14 16)",
            "8 - 2 = 6 (left: 6 8 14)",
            "14 - 8 = 6 (left: 2 6 8)",
            "14 /  2 = 7 (left: 7 8 8)",
            "14 - 2 = 12 (left: 8 8 12)"
        ]
    }
]


@text.prompt
def propose_prompt(input, examples=propose_examples):
    '''
    {% for example in examples %}
    Input: {{ example.input }}
    Possible next steps:
    {% for next_step in example.next_steps %}
    {{ next_step }}
    {% endfor %}
    {% endfor %}
    Input: {{ input }}
    Possible next steps:
    '''


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


def get_proposals(x: str, y: str = '') -> list[str]:
    current_numbers = get_current_numbers(y if y else x)
    if current_numbers == '24':
        prompt = cot_prompt.format(input=x) + 'Steps:' + y
        print("End reached cot=", [prompt])
    else:
        prompt = propose_prompt(current_numbers)

    proposals_str = model(prompt)
    proposals_arr = proposals_str.split('\n')

    # new thought has past steps encoded in it
    proposals_arr = [y + _ + '\n' for _ in proposals_arr]
    return proposals_arr


value_examples = [
    {
        "input": "10 14",
        "reasoning_steps": ["10 + 14 = 24"],
        "output": "sure"
    },
    {
        "input": "11 12",
        "reasoning_steps": ["11 + 12 = 23", "12 - 11 = 1", "11 * 12 = 132", "11 / 12 = 0.91"],
        "output": "impossible"
    },
    {
        "input": "4 4 10",
        "reasoning_steps": [
            "4 + 4 + 10 = 8 + 10 = 18",
            "4 * 10 - 4 = 40 - 4 = 36",
            "(10 - 4) * 4 = 6 * 4 = 24"
        ],
        "output": "sure"
    },
    {
        "input": "4 9 11",
        "reasoning_steps": ["9 + 11 + 4 = 20 + 4 = 24"],
        "output": "sure"
    },
    {
        "input": "5 7 8",
        "reasoning_steps": [
            "5 + 7 + 8 = 12 + 8 = 20",
            "(8 - 5) * 7 = 3 * 7 = 21",
            "I cannot obtain 24 now, but numbers are within a reasonable range"
        ],
        "output": "likely"
    },
    {
        "input": "5 6 6",
        "reasoning_steps": [
            "5 + 6 + 6 = 17",
            "(6 - 5) * 6 = 1 * 6 = 6",
            "I cannot obtain 24 now, but numbers are within a reasonable range"
        ],
        "output": "likely"
    },
    {
        "input": "10 10 11",
        "reasoning_steps": [
            "10 + 10 + 11 = 31",
            "(11 - 10) * 10 = 10",
            "10 10 10 are all too big"
        ],
        "output": "impossible"
    },
    {
        "input": "1 3 3",
        "reasoning_steps": [
            "1 * 3 * 3 = 9",
            "(1 + 3) * 3 = 12",
            "1 3 3 are all too small"
        ],
        "output": "impossible"
    }
]


@text.prompt
def value_prompt(input, examples=value_examples):
    '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
    {% for example in examples %}
    Input: {{ example.input }}
    {% for next_step in example.reasoning_steps %}
    {{ next_step }}
    {% endfor %}
    {{ example.output }}
    {% endfor %}
    Input: {{input}}
    '''


value_last_step_examples = [
    {
        "input": "4 4 6 8",
        "answer": "(4 + 8) * (6 - 4) = 24",
        "judge": "sure"
    },
    {
        "input": "2 9 10 12",
        "answer": "2 * 12 * (10 - 9) = 24",
        "judge": "sure"
    },
    {
        "input": "4 9 10 13",
        "answer": "(13 - 9) * (10 - 4) = 24",
        "judge": "sure"
    },
    {
        "input": "4 4 6 8",
        "answer": "(4 + 8) * (6 - 4) + 1 = 25",
        "judge": "impossible"
    },
    {
        "input": "2 9 10 12",
        "answer": "2 * (12 - 10) = 24",
        "judge": "impossible"
    },
    {
        "input": "4 9 10 13",
        "answer": "(13 - 4) * (10 - 9) = 24",
        "judge": "impossible"
    },
]

@text.prompt
def value_last_step_prompt(input, answer, examples=value_last_step_examples):
    '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
    {% for example in examples %}
    Input: {{ example.input }}
    Answer: {{ example.answer }}
    Judge: {{ example.judge }}
    {% endfor %}
    Input: {{input}}
    Answer: {{answer}}
    Judge:'''



def get_value_prompt(x, y):
    last_line = y.strip().split('\n')[-1]
    if 'left: ' not in last_line:  # last step
        ans = last_line.lower().replace('answer: ', '')
        # print([value_last_step_prompt.format(input=x, answer=ans)])
        return value_last_step_prompt(input=x, answer=ans)
    current_numbers = get_current_numbers(y)
    return value_prompt(current_numbers)


def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
    if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
        print(f'4 steps done but no answer found {y=}')
        return 0
    value_names = [_.split('\n')[-1] for _ in value_outputs]
    value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
    value = sum(value * value_names.count(name)
                for name, value in value_map.items())

    print(f'->{x=}')
    print(f'->{y=}')
    print(f'->{value_names=}')
    print(f'->{value=}')
    return value


cache = SqliteDict('./my_db.sqlite', autocommit=True)


def get_value(x, y, n_evaluate_sample, cache_value):
    '''
    Check compatibility of a partial output with the input using a prompt
    '''
    value_prompt = get_value_prompt(x, y)
    if cache_value and value_prompt in cache:
        return cache[value_prompt]
    value_outputs = model(value_prompt, samples=n_evaluate_sample)
    value = value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        cache[value_prompt] = value
    return value


def get_values(x, ys, n_evaluate_sample, cache_value=False):
    values = []
    local_value_cache = {}
    for y in tqdm(ys):  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            value = get_value(x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


for x in tqdm(inputs):
    print(x)
    break

n_steps = 4
n_eval = 4  # how many samples, to calculate value
n_best = 5  # how many best candidates to select

ys = ['']  # current output candidates
for step in range(n_steps):
    print(f"{step=} {len(ys)=}")
    # step 1 - generate new candidates
    # new proposals for the next step which should ideally have less number of numbers left
    # also possible that all the candidates are bad/invalid
    new_ys = [get_proposals(x, y) for y in ys]
    new_ys = list(itertools.chain(*new_ys))

    # for y in new_ys:
    # print(f"-- {y=}")

    # step 2 - evaluate the candidates
    values = get_values(x, new_ys, n_eval)

    # Sort the values in descending order
    sorted_indices = np.argsort(values)[::-1]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_ys = [new_ys[i] for i in sorted_indices]

    # Print the sorted values and corresponding new_ys
    for val, y in zip(sorted_values, sorted_ys):
        print(f"-- {val=}\t{y=}")

    # Step 3 - Select the best candidates
    best_values = sorted_values[:n_best]
    best_ys = sorted_ys[:n_best]

    for y, value in zip(best_ys, best_values):
        print(f"best -- {value=}\t{y=}")

    # step 4 - update the candidates
    ys = best_ys

answers = ys

outputs = [ys]


# check accuracy of the answers
n_any_correct = 0
avg_correct_arr = []
for x, ys in zip(inputs, outputs):
    status = [eval_output_game24(x, y) for y in ys]
    if any(status):
        n_any_correct += 1

    avg_correct_arr.append(sum(status) / len(status))
    break


print("ToT Prompting\n------------------")
print("n_inputs: ", len(inputs))
print("n_inputs with atleast 1 correct answer: ", n_any_correct)
print("avg number of correct answers per input: ",
      sum(avg_correct_arr) / len(avg_correct_arr))
