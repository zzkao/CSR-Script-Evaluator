#!/bin/bash

# Training
python -m graphrag.index --root ./ragtest --init
python -m graphrag.index --root ./ragtest

# Inference / Demonstration
python -m graphrag.query --root ./ragtest --method global "What are the top themes in this story?"
python -m graphrag.query --root ./ragtest --method local "Who is Scrooge and what are his main relationships?"

# Testing / Evaluation
pytest
```