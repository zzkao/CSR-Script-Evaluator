from typing import Optional

class Action:
    def __init__(self, command: str,  description: Optional[str] = None, ):
        self.command = command
        self.description = description
    
    def to_dict(self):
        return {"command": self.command, "description": self.description}

    def __str__(self):
        parts = []

        if self.command and self.command.strip():
            parts.append(f"command:\n{self._indent(self.command)}")
        if self.description and self.description.strip():
            parts.append(f"description:\n{self._indent(self.description)}")

        return "\n".join(parts)

    @staticmethod
    def _indent(text, spaces=2):
        return "\n".join(" " * spaces + line for line in text.splitlines())

class State:
    def __init__(self, action: Action, output: str, eval: str):
        self.action = action
        self.output = output
        self.eval = eval
    
    def to_dict(self):
        if self.eval:
            return {"action": self.action.to_dict(), "output": self.output, "eval": self.eval}
        else:
            return {"action": self.action.to_dict(), "output": self.output}

    def __str__(self):
        if self.eval:
            return (
                "Action:\n"
                f"{self._indent(str(self.action))}\n"
                "Output:\n"
                f"{self._indent(self.output)}\n"
                "Evaluation:\n"
                f"{self._indent(self.eval)}"
            )
        else:
            return (
                "Action:\n"
                f"{self._indent(str(self.action))}\n"
                "Output:\n"
                f"{self._indent(self.output)}"
            )
    
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _indent(text, spaces=2):
        return "\n".join(" " * spaces + line for line in text.splitlines())
    