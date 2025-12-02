import subprocess
import random
import string
import re

class CommandExecutor():
    def __init__(self, commands, timeout=60):
        self.commands = commands
        self.proc = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.rand_str = ''.join(random.choices(string.ascii_letters, k=20))
        self.timeout = timeout

    def _parser(self, string):
        stack = None
        output = []
        chunk = []
        lines = string.splitlines()

        for line in lines:
            if not stack:
                stack = line

            elif stack == line:
                output.append( "\n".join(chunk) )
                chunk = []
                stack = None
            else:
                chunk.append(line)
        return output

    def _clean_bash_prefix(self, ls):
        return [re.sub(r"bash: line \d+:\s*", "", text) for text in ls]


    def run(self):
        self.proc.stdin.write("" + "\n")
        self.proc.stdin.flush()
        for i, cmd in enumerate(self.commands):
            print(f'RUNNING {cmd}')
            self.proc.stdin.write(f"echo \"{self.rand_str}{i}\"; echo \"{self.rand_str}{i}\" >&2\n")
            self.proc.stdin.flush()

            # Send the command
            self.proc.stdin.write(f"timeout {self.timeout}s {cmd}; code=$?; [ $code -eq 124 ] && echo '__TIMEOUT__' >&2\n") # Run cmd and enforce timeout
            self.proc.stdin.flush()


            self.proc.stdin.write(f"echo \"{self.rand_str}{i}\"; echo \"{self.rand_str}{i}\" >&2\n")
            self.proc.stdin.flush()

        # Close the subprocess
        outputs, errors = self.proc.communicate() 

        stdout = self._parser(outputs)
        stderr = self._clean_bash_prefix(self._parser(errors))

        return stdout, stderr