import json
from CohereLink import CohereLink as Cohere
import requests
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm
import time
import gc
import matplotlib.pyplot as plt
import math

def generate_bar_chart(values: list[float], stddev: list[float]):
    categories = ['Normal Model\nUnprompted', 'Normal Model\nPrompted', 'Finetuned Model\nUnprompted', 'Finetuned Model\nPrompted']

    # Create a customized bar chart
    plt.bar(categories, values, yerr=stddev, color='skyblue', edgecolor='black', width=0.6)

    # Add labels and title
    plt.xlabel('Model Types')
    plt.ylabel('Similarity Score / Normal Model Unprompted score')
    plt.title(f'{model_name} performance')

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.savefig(f"{model_name}_performance.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def get_embeds(texts: list[str]):
    
    response = session.post("https://api.cohere.com/v2/embed", json = {
                "model": embedding_model,
                "texts": texts,
                "input_type": "classification",
                "embedding_types": ["float"]
            }
        )
    
    response.raise_for_status()
    time.sleep(10)
    
    return response.json()['embeddings']['float']

def calculate_similarity(reference_embed, reference_norm, text_embed: list[float]):
    
    text_array = np.array(text_embed).flatten()
    
    value = np.dot(reference_embed, text_array) / (reference_norm * norm(text_array))
    
    del text_embed
    del text_array
    return value

def main():
    
    with open(f"{model_name}_data_v3.jsonl", "r") as f:
        lines = f.read().split("\n")
    
    if not lines[-1]:
        lines.pop(len(lines) - 1)
    
    data = {key: [] for key in json.loads(lines[0].strip()).keys()}
    
    for line in lines:
        line_json = json.loads(line.strip())
        
        for key in line_json.keys():
            data[key].append(line_json[key])
    
    length = len(lines)
    
    normal_model_unprompted_scores = []
    normal_model_prompted_scores = []
    finetuned_model_unprompted_scores = []
    finetuned_model_prompted_scores = []
    
    teacher_embeds = []
    normal_unprompted_embeds = []
    normal_prompted_embeds = []
    finetuned_unprompted_embeds = []
    finetuned_prompted_embeds = []
    
    #Get embeddings
    for i in tqdm(range(0, length, 96), desc="Embeddings"):
        
        for item in get_embeds(data['teacher'][i:i+96]):
            teacher_embeds.append(item)
        
        for item in get_embeds(data['normal-model-unprompted'][i:i+96]):
            normal_unprompted_embeds.append(item)
            
        for item in get_embeds(data['normal-model-prompted'][i:i+96]):
            normal_prompted_embeds.append(item)
            
        for item in get_embeds(data['finetuned-model-unprompted'][i:i+96]):
            finetuned_unprompted_embeds.append(item)
            
        for item in get_embeds(data['finetuned-model-prompted'][i:i+96]):
            finetuned_prompted_embeds.append(item)
    
    teacher_embeds = teacher_embeds[0:length]
    normal_unprompted_embeds = normal_unprompted_embeds[0:length]
    normal_prompted_embeds = normal_prompted_embeds[0:length]
    finetuned_unprompted_embeds = finetuned_unprompted_embeds[0:length]
    finetuned_prompted_embeds = finetuned_prompted_embeds[0:length]
    
    #For stddev
    normal_model_unprompted_sum_sq = 0
    normal_model_prompted_sum_sq = 0
    finetuned_model_unprompted_sum_sq = 0
    finetuned_model_prompted_sum_sq = 0
    
    for reference_embed, normal_model_unprompted, normal_model_prompted, finetuned_model_unprompted, finetuned_model_prompted in tqdm(zip(teacher_embeds, normal_unprompted_embeds, normal_prompted_embeds, finetuned_unprompted_embeds, finetuned_prompted_embeds)):
        reference_norm = norm(reference_embed)
        
        normal_unprompted_score = calculate_similarity(reference_embed, reference_norm, normal_model_unprompted)
        normal_prompted_score = calculate_similarity(reference_embed, reference_norm, normal_model_prompted) / normal_unprompted_score
        finetuned_unprompted_score = calculate_similarity(reference_embed, reference_norm, finetuned_model_unprompted) / normal_unprompted_score
        finetuned_prompted_score = calculate_similarity(reference_embed, reference_norm, finetuned_model_prompted) / normal_unprompted_score
        normal_unprompted_score = 1
        
        normal_model_unprompted_scores.append(normal_unprompted_score)
        normal_model_prompted_scores.append(normal_prompted_score)
        finetuned_model_unprompted_scores.append(finetuned_unprompted_score)
        finetuned_model_prompted_scores.append(finetuned_prompted_score)
        
        del reference_embed
        del reference_norm
        
        gc.collect()
        
    normal_model_unprompted_score = sum(normal_model_unprompted_scores) / length
    normal_model_prompted_score = sum(normal_model_prompted_scores) / length
    finetuned_model_unprompted_score = sum(finetuned_model_unprompted_scores) / length
    finetuned_model_prompted_score = sum(finetuned_model_prompted_scores) / length
        
    for noru, norp, fineu, finep in zip(normal_model_unprompted_scores, normal_model_prompted_scores, finetuned_model_unprompted_scores, finetuned_model_prompted_scores):
        normal_model_unprompted_sum_sq += (noru - normal_model_unprompted_score) ** 2
        normal_model_prompted_sum_sq += (norp - normal_model_prompted_score) ** 2
        finetuned_model_unprompted_sum_sq += (fineu - finetuned_model_unprompted_score) ** 2
        finetuned_model_prompted_sum_sq += (finep - finetuned_model_prompted_score) ** 2
    
    scores = [normal_model_unprompted_score, normal_model_prompted_score, finetuned_model_unprompted_score, finetuned_model_prompted_score]
    stddev = [math.sqrt(normal_model_unprompted_sum_sq/length), math.sqrt(normal_model_prompted_sum_sq/length), math.sqrt(finetuned_model_unprompted_sum_sq/length), math.sqrt(finetuned_model_prompted_sum_sq/length)]
        
    with open(f"{model_name}_performance.json", "w") as f:
        f.write(json.dumps({
            "normal-model-unprompted": (normal_model_unprompted_score),
            "normal-model-prompted": normal_model_prompted_score,
            "finetuned-model-unprompted": finetuned_model_unprompted_score,
            "finetuned-model-prompted": finetuned_model_prompted_score
        }, indent=2))
        
    generate_bar_chart(scores, stddev)
        
        

if __name__ == "__main__":
    
    with open("keys.json", "r") as f:
        key = json.loads(f.read())['Cohere_2']
    
    embedding_model = "embed-english-v3.0"
    model_name = "gpt-4o-mini-conversation"
    
    with requests.Session() as session:
        session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"bearer {key}",
                'accept': 'application/json',
            })
        main()