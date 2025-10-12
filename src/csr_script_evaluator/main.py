from .script_evaluator import ScriptEvaluator
from .command_executor import CommandExecutor
import subprocess
import argparse
from importlib import resources
from .state import *
import sys
import json

def read_test_script_commands(bash_file):
        commands = []
        current_cmd = []

        with resources.open_text("csr_script_evaluator.data", bash_file) as f:
            for line in f:
                line = line.strip()
                # skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # if line ends with \, continue building the command
                if line.endswith("\\"):
                    current_cmd.append(line[:-1].strip())
                else:
                    current_cmd.append(line)
                    # end of a full command
                    commands.append(" ".join(current_cmd))
                    current_cmd = []

        return commands

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, required=False, help="Force run the exact test script number")
    parser.add_argument("--apikey", type=str, required=True, help="Input your anthropic API key")
    parser.add_argument("-v", "--verbose", action='store_true', help="See full history of command execution")
    parser.add_argument("-t", "--timeout", type=int, required=False, help="Set command timeout length in seconds")
    args = parser.parse_args()

    APIKEY = args.apikey
    test_number = args.number
    timeout = args.timeout

    # Autodetect script number
    current_directory = (subprocess.run("pwd", shell=True, capture_output=True, text=True).stdout.strip())
    repo_name = current_directory.rsplit('/', 1)[-1]
    with resources.open_text("csr_script_evaluator.data", "CSRBench100.txt") as f:
        repo_to_num = {(line.strip()).rsplit('/', 1)[-1]: i for i, line in enumerate(f, start=1)}
    if not test_number and repo_name not in repo_to_num:
        sys.exit("Error: Automatic script number detection failed because the current repository is not listed in CSRBench100.txt. Are you in the repo's directory? \nHint: Check your current directory with pwd. If you are in the repo, please manually input test script number using csr-eval -n.")
    if not test_number:
        test_number = repo_to_num[repo_name]
    if not timeout:
        timeout = 60

    # Parse commands
    commands = read_test_script_commands(f"{test_number}.sh")


    # Run commands
    if args.verbose:
        print('RUNNING TEST SCRIPTS')
    evaluator = ScriptEvaluator(api_key=APIKEY)
    executor = CommandExecutor(commands=commands, timeout=timeout)
    stdouts, stderrs = executor.run()

    history = []
    success = 0
    for i in range(len(commands)):
        command = commands[i]
        stdout = stdouts[i]
        stderr = stderrs[i]

        if '__TIMEOUT__' in stderr: # Autopass on timeouts
            success += 1
            continue
        else:
            eval = evaluator.query(bash_script=command, stdout=stdout, stderr=stderr)
            history.append(State(Action(command=command), 
                                output=f"{stdout} {stderr}",
                                eval=("SUCCESS" if eval else "FAILED")
                                ))
        if eval:
            success += 1
        
    total = len(commands)
    if args.verbose:
        for i in history:
            print(i.to_dict())
    
    # Logging
    log_data = {
        "summary": {"success": success, "total": total},
        "states": [s.to_dict() for s in history]
    }
    with open("results.json", "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"Final Result: {success} / {total}")
    