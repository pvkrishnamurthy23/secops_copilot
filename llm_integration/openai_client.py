import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def ask_openai(prompt):
    response = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content 