import json
import openai
import anthropic
import requests
import time
from typing import Optional
import os

CONFIG_PATH = os.path.join("game_theory_llm/config/conf.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

api_call_counts = {
    'gpt-4': {'calls': 0, 'successes': 0},
    'gpt-3.5': {'calls': 0, 'successes': 0},
    'gpt-4o-mini': {'calls': 0, 'successes': 0},
    'gpt-o1-mini': {'calls': 0, 'successes': 0},
    'qwen': {'calls': 0, 'successes': 0},
    'gemini': {'calls': 0, 'successes': 0},
    'claude-sonnet': {'calls': 0, 'successes': 0},
}


def call_model(model_name, prompt, max_retries=5, retry_delay=20):
    for attempt in range(max_retries):
        try:
            if model_name == 'gpt-4':
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    max_tokens=1500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=config["openai"]["gpt4_api_key"]
                )
                api_call_counts['gpt-4']['calls'] += 1
                api_call_counts['gpt-4']['successes'] += 1
                return response["choices"][0]["message"]["content"]

            elif model_name == 'gpt-3.5':
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    max_tokens=1500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=config["openai"]["gpt35_api_key"]
                )
                api_call_counts['gpt-3.5']['calls'] += 1
                api_call_counts['gpt-3.5']['successes'] += 1
                return response["choices"][0]["message"]["content"]

            elif model_name == 'gpt-4o-mini':
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    max_tokens=1500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=config["openai"]["gpt4mini_api_key"]
                )
                api_call_counts['gpt-4o-mini']['calls'] += 1
                api_call_counts['gpt-4o-mini']['successes'] += 1
                return response["choices"][0]["message"]["content"]

            elif model_name == 'gpt-o1-mini':
                response = openai.ChatCompletion.create(
                    model="o1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    api_key=config["openai"]["gpt01mini_api_key"]
                )
                api_call_counts['gpt-o1-mini']['calls'] += 1
                api_call_counts['gpt-o1-mini']['successes'] += 1
                return response["choices"][0]["message"]["content"]

            elif model_name == 'qwen':
                CLIENT = Client(config["qwen"]["endpoint"])
                result = CLIENT.submit(prompt, None, api_name="/model_chat").result()
                answer = result[1][0][1]
                api_call_counts['qwen']['calls'] += 1
                api_call_counts['qwen']['successes'] += 1
                return answer

            elif model_name == 'gemini':
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="+config['gemini']['api_key']
                headers = {"Content-Type": "application/json"}
                data = {"contents": [{"parts": [{"text": prompt}]}]}

                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    api_call_counts['gemini']['calls'] += 1
                    api_call_counts['gemini']['successes'] += 1
                    return response.json()['candidates'][0]['content']['parts'][0]['text']
                elif response.status_code in [429, 503]:
                    print(f"Server error (code: {response.status_code}). Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay * 2)
                    continue
                else:
                    print(f"API call failed with status code: {response.status_code}")

            elif model_name == 'claude-sonnet':
                client = anthropic.Anthropic(api_key=config["claude"]["api_key"])
                response = client.messages.create(
                    model="claude-3-sonnet",
                    max_tokens=1500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                api_call_counts['claude-sonnet']['calls'] += 1
                api_call_counts['claude-sonnet']['successes'] += 1
                return response.content

        except Exception as e:
            print(f"Error in API call (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None

    return None


if __name__ == "__main__":
    prompt = "hello, world!"
    result = call_model("gemini", prompt)
    print(result)
