import outlines.text as text
import outlines.models as models
import openai
import re
import sympy
import pandas as pd
from tqdm import tqdm

@text.prompt
def standard_prompt(input):
    '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
    Input: 4 4 6 8
    Answer: (4 + 8) * (6 - 4) = 24
    Input: 2 9 10 12
    Answer: 2 * 12 * (10 - 9) = 24
    Input: 4 9 10 13
    Answer: (13 - 9) * (10 - 4) = 24
    Input: 1 4 8 8
    Answer: (8 / 4 + 1) * 8 = 24
    Input: 5 5 5 9
    Answer: 5 + 5 + 5 + 9 = 24
    Input: {{input}}
    '''
    
@text.prompt
def cot_prompt(input):
    '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
    Input: 4 4 6 8
    Steps:
    4 + 8 = 12 (left: 4 6 12)
    6 - 4 = 2 (left: 2 12)
    2 * 12 = 24 (left: 24)
    Answer: (6 - 4) * (4 + 8) = 24
    Input: 2 9 10 12
    Steps:
    12 * 2 = 24 (left: 9 10 24)
    10 - 9 = 1 (left: 1 24)
    24 * 1 = 24 (left: 24)
    Answer: (12 * 2) * (10 - 9) = 24
    Input: 4 9 10 13
    Steps:
    13 - 10 = 3 (left: 3 4 9)
    9 - 3 = 6 (left: 4 6)
    4 * 6 = 24 (left: 24)
    Answer: 4 * (9 - (13 - 10)) = 24
    Input: 1 4 8 8
    Steps:
    8 / 4 = 2 (left: 1 2 8)
    1 + 2 = 3 (left: 3 8)
    3 * 8 = 24 (left: 24)
    Answer: (1 + 8 / 4) * 8 = 24
    Input: 5 5 5 9
    Steps:
    5 + 5 = 10 (left: 5 9 10)
    10 + 5 = 15 (left: 9 15)
    15 + 9 = 24 (left: 24)
    Answer: ((5 + 5) + 5) + 9 = 24
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

def test_output_game24(input: str, output: str) -> bool:
    '''
    Use regex to check if the output is correct for the given input
    '''
    expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
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
    
def test_standard():
    '''
    Test the score using 5-shot standard prompting
    '''
    ## constants
    n_samples = 5
    complete = models.text_completion.openai("gpt-3.5-turbo", max_tokens = 1000, temperature = 0.7)
    
    ## generate all answers for all questions
    outputs = []
    for x in tqdm(inputs):
        prompt = standard_prompt(x)
        answers = complete(prompt, sample = n_samples)
        outputs.append(answers)
        
    ## check accuracy of the answers
    n_any_correct = 0
    avg_correct_arr = []
    for x, ys in zip(inputs, outputs):
        status = [test_output_game24(x, y) for y in ys]
        if any(status):
            n_any_correct += 1
        
        avg_correct_arr.append(sum(status) / len(status))
    
    
    
    print("Standard Prompting\n------------------")
    print("n_inputs: ", len(inputs))
    print("n_inputs with atleast 1 correct answer: ", n_any_correct)
    print("avg number of correct answers per input: ", sum(avg_correct_arr) / len(avg_correct_arr))
    

def test_cot():
    '''
    Test the score using 5-shot cot prompting
    '''
    ## constants
    n_samples = 5
    complete = models.text_completion.openai("gpt-3.5-turbo", max_tokens = 1000, temperature = 0.7)
    
    ## generate all answers for all questions
    outputs = []
    for x in tqdm(inputs):
        prompt = cot_prompt(x)
        messages = [{"role": "user", "content": prompt}]
        answers = complete(prompt, sample = n_samples)
        outputs.append(answers)
        
    ## check accuracy of the answers
    n_any_correct = 0
    avg_correct_arr = []
    for x, ys in zip(inputs, outputs):
        status = [test_output_game24(x, y) for y in ys]
        if any(status):
            n_any_correct += 1
        
        avg_correct_arr.append(sum(status) / len(status))
    
    
    
    print("COT Prompting\n------------------")
    print("n_inputs: ", len(inputs))
    print("n_inputs with atleast 1 correct answer: ", n_any_correct)
    print("avg number of correct answers per input: ", sum(avg_correct_arr) / len(avg_correct_arr))
    
    
if __name__ == "__main__":
    completion_tokens = 0
    prompt_tokens = 0

    data = pd.read_csv("./data/24.csv")['Puzzles']
    inputs = data[200:210] # 10 examples

    # test_standard()
    test_cot()