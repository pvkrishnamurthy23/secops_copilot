import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def ask_openai(prompt):
    response = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content 

def test_openai_connection():
    """Test the OpenAI API key and connection by listing available models."""
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not found in environment variables.")
        return False
    try:
        openai.api_key = OPENAI_API_KEY
        models = openai.models.list()
        #model_count = sum(1 for _ in models)
        #print(f"Connection successful! {model_count} models available.")
        print(f"Connection successful!")
        return True
    except Exception as e:
        print(f"OpenAI connection failed: {e}")
        return False

def generate_openai_response(prompt):
    """Generate a response from OpenAI using the ask_openai helper."""
    try:
        response = ask_openai(prompt)
        print(f"OpenAI response: {response}")
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

if __name__ == "__main__":
    if test_openai_connection():
        generate_openai_response("for the following link help.calibo.com, summarise the content on the page?")
