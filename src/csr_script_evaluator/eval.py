from .script_evaluator import ScriptEvaluator
import subprocess, sys, os
import argparse
from importlib import resources

def read_test_script():
    with resources.open_text("csr_script_evaluator.data", "hi.txt") as f:
        content = f.read()
    return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", type=str, required=True, help="Your Anthropic API key")
    parser.add_argument("-n", "--number", type=int, required=False, help="Run the exact test script number. Use if autodetect isn't working")
    args = parser.parse_args()

    APIKEY = args.apikey
    TEST_NUMBER = args.number

    print("Current Python:", sys.executable)
    subprocess.run("pwd", shell=True)
    print(read_hi())
    
    print(APIKEY)
