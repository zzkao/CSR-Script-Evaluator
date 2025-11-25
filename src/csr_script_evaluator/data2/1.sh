#!/bin/bash

# Training
# No training commands found in repository

# Inference / Demonstration
python examples/storm_examples/run_storm_wiki_gpt.py --output-dir ./results/gpt --retriever you --do-research --do-generate-outline --do-generate-article --do-polish-article
python examples/storm_examples/run_storm_wiki_gpt.py --output-dir ./results/gpt --retriever you --do-research --do-generate-outline --do-generate-article --do-polish-article --topic "Bitcoin"
python examples/storm_examples/run_storm_wiki_open_models.py --output-dir ./results/open --retriever bing --do-research --do-generate-outline --do-generate-article --do-polish-article
python examples/storm_examples/run_storm_wiki_open_models.py --url https://en.wikipedia.org/wiki/Example --output-dir ./results/curated --retriever bing --do-research --do-generate-outline --do-generate-article --do-polish-article

# Testing / Evaluation
# No specific testing commands found in repository
```