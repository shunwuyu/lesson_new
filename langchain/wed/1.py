import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

print(os.getenv("OPENAI_API_KEY"))