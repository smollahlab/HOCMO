import os
import openai
import pandas as pd
import numpy as np
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv

openai.api_key = os.getenv('OPENAI_API_KEY')



