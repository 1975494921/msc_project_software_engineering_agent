from abc import abstractmethod
from typing import Annotated, TypedDict


class Node:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, state):
        pass

class State(TypedDict):
    messages: list
    complexity: str
    llm_query: str
    project_dir: str
    code_history: dict
    debug_history: dict
    structure: list
    current_index: int

    debugging: bool
    debug_timeout: int
    max_debugging_num: int
