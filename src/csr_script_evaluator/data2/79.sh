#!/bin/bash

# Training
# No training commands in README

# Inference / Demonstration
mistral-chat $MISTRAL_MODEL --instruct --max_tokens 256 --temperature 0.35
python -c "from mistral_inference.transformer import Transformer; from mistral_inference.generate import generate; from mistral_common.tokens.tokenizers.mistral import MistralTokenizer; from mistral_common.protocol.instruct.messages import UserMessage; from mistral_common.protocol.instruct.request import ChatCompletionRequest; tokenizer = MistralTokenizer.from_model('mistralai/Mistral-7B-Instruct-v0.3'); model = Transformer.from_folder('.'); prompt = 'How expensive would it be to ask a window cleaner to clean all windows in Paris. Make a reasonable guess in US Dollar.'; messages = [UserMessage(content=prompt)]; completion_request = ChatCompletionRequest(messages=messages); tokens = tokenizer.encode_chat_completion(completion_request).tokens; out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id); result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0]); print(result)"

# Testing / Evaluation
# No testing/evaluation commands in README
```