import requests as r
import json
import time
import pandas as pd
import requests
from tqdm import tqdm
import random

def load_dataset(url = "https://raw.githubusercontent.com/eth-nlped/mathdial/refs/heads/main/data/test.csv") -> pd.DataFrame:
    request = requests.get(url)

    with open("buffer.csv", 'wb') as f:
        f.write(request.content)
    
    test_full = pd.read_csv('buffer.csv')
    test = test_full[['question', 'student_incorrect_solution', 'conversation']]
    return test

def generate_model_data(response_function, length: int = 50, filename: str = "prompted_model_data.json"):
    csv_data = load_dataset("https://raw.githubusercontent.com/eth-nlped/mathdial/refs/heads/main/data/test.csv")
    
    if length < 0:
        length = len(csv_data)
    
    prompt = """
    You are a teacher supposed to help guide students towards the right answer.
    The problem the student is working on is {problem}.
    Do not tell the student exactly where they went wrong, instead, guide them towards recognizing
    their mistake so they can fix it themselves. Ask them questions about the way they went through the
    problem so that they recognize their mistakes, but keep your response concise and to the point.
    """.replace("\n", "").replace("    ", " ").strip()
        
    json_data = []
    
    for index in tqdm(range(length)):
        question = csv_data['question'][index]
        student = csv_data['student_incorrect_solution'][index]
        #teacher = csv_data['conversation'][index].split("|EOM|")[0].split(':')[-1].split(')')[-1]

        conversation = [
            item.split(':')[-1].split(')')[-1] for item in csv_data['conversation'][index].split("|EOM|")
        ]

        index = random.randint(0, len(conversation) - 1)
        if index % 2 != 0:
            index -= 1

        chat_history = conversation[0:index]
        
        try:
            better_response = response_function(student, context=prompt.format(problem=question), chat_history = chat_history)
            response = response_function(student, context=f"You have been asked to help a student with the following problem: {question}")
        except Exception as err:
            print(err)
            break
        
        json_data.append({
            "question": question,
            "student": student,
            "teacher": conversation[index],
            "chat_history": chat_history,
            "prompted-model": better_response,
            "normal-model": response
        })
        
        time.sleep(1)
        
    print(f"Got {len(json_data)} examples of data")
    with open(filename, "w") as f:
        f.write(json.dumps(json_data, indent=2))

def put_data_together(new_filename: str = "gpt-4o-mini_data.jsonl", normal_model_filename: str = "openai_normal_model_data.json", finetuned_model_filename: str = "finetuned_model_data.json") -> None:
    with open(normal_model_filename, "r") as f:
        normal_model_data = json.loads(f.read())
        
    with open(finetuned_model_filename, "r") as f:
        finetuned_model_data = json.loads(f.read())
    
    with open(new_filename, "w") as f:
        for normal_model, finetuned_model in zip(normal_model_data, finetuned_model_data):
            new_data = {
                "question": normal_model['question'],
                "student": normal_model['student'],
                "teacher": normal_model['teacher'],
                "normal-model-unprompted": normal_model['normal-model'],
                "normal-model-prompted": normal_model['prompted-model'],
                "finetuned-model-unprompted": finetuned_model['normal-model'],
                "finetuned-model-prompted": finetuned_model['prompted-model'],
                "conversation-history": normal_model['chat_history']
            }
            
            f.write(f"{json.dumps(new_data)}\n")

class openai_wrapper:
    
    def __init__(self, filename: str = None, key: str = None):
        if (key):
            self.key = key
        else:
            with open(filename, 'r') as f:
                self.key = json.loads(f.read())['OpenAI']
                
        self.model = "gpt-4o-mini"
        
    def get_response(self, text: str, context: str = "", chat_history: list = []) -> str:

        messages = [{"role": "developer", "content": context}]
        messages.append({"role": "user", "content": text})

        for i in range(0, len(chat_history)):
            role = "user" if i % 2 != 0 else "assistant"
            messages.append({"role": role, "content": chat_history[i]})

        #print(json.dumps(messages, indent=2))

        response = r.post("https://api.openai.com/v1/chat/completions",
            
            headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
            }, 
            
            json={
                "model": self.model,
                "messages": messages
            }
        )
        
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def main():

    normal = openai_wrapper("keys.json")
    finetuned = openai_wrapper("keys.json")
    finetuned.model = "ft:gpt-4o-mini-2024-07-18:personal::AX31ySoX"

    length: int = 50

    generate_model_data(normal.get_response, filename='gpt-4o-mini-normal-conversation.json', length=length)
    generate_model_data(finetuned.get_response, filename='gpt-4o-mini-finetuned-conversation.json', length=length)
    put_data_together(new_filename='gpt-4o-mini-conversation_data_v3.jsonl', normal_model_filename='gpt-4o-mini-normal-conversation.json', finetuned_model_filename='gpt-4o-mini-finetuned-conversation.json')



if __name__ == "__main__":
    main()