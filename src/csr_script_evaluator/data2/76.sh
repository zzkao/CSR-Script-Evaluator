#!/bin/bash

# Training

# Inference / Demonstration
python -c "
from gptcache import cache
from gptcache.adapter import openai

cache.init()
cache.set_openai_key()

# OpenAI call with cache
response = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[{
    'role': 'user',
    'content': 'what's github'
  }],
)
print(response)
"

python -c "
from gptcache import cache
from gptcache.processor.pre import get_prompt
from gptcache.adapter import openai

cache.init(pre_embedding_func=get_prompt)
cache.set_openai_key()

# Example with embedding function
response = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[{
    'role': 'user',
    'content': 'what's github'
  }],
)
print(response)
"

# Testing / Evaluation
```