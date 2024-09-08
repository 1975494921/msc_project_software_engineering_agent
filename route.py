from typing import Literal

from abstract import State


def route_planning_codegeneration(state: State) -> Literal["planning_node", "code_generation_node", "__end__"]:
    if state['current_index'] == len(state['structure']):
        return "__end__"

    return "code_generation_node"


def code_generation_node_route(state: State) -> Literal["debugging_node", "write_code_node"]:
    code_key = list(state['structure'][state['current_index']].keys())[0]

    if state['debugging']:
        if code_key in state['debug_history'] and state['debug_history'][code_key]['num'] > state['max_debugging_num']:
            return "write_code_node"

        return "debugging_node"

    return "write_code_node"


def debugging_route(state: State) -> Literal["write_code_node", "code_generation_node"]:
    code_key = list(state['structure'][state['current_index']].keys())[0]

    if not state['debugging']:
        return "write_code_node"

    else:
        if state['debug_history'][code_key]['passed']:
            return "write_code_node"

        elif state['debug_history'][code_key]['num'] > state['max_debugging_num']:
            return "write_code_node"

        else:
            return "code_generation_node"
