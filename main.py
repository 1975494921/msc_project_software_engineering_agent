import os
import sys
import langgraph

from langgraph.graph import StateGraph

from abstract import State
from generation_node import DetectComplexityNode, StructureGenerationNode, CodeGenerationNode
from route import route_planning_codegeneration, code_generation_node_route, debugging_route
from utility_node import PlanningNode, WriteCodeNode, DebuggingNode

class DualOutput:
    def __init__(self, file, console):
        self.file = file
        self.console = console

    def write(self, message):
        self.file.write(message)
        self.console.write(message)

    def flush(self):
        self.file.flush()
        self.console.flush()

def build_graph(llm_query: str):
    builder = StateGraph(State)

    builder.add_node("detect_complexity_node", DetectComplexityNode(llm_query))
    builder.add_node("structure_generation_node", StructureGenerationNode())
    builder.add_node("planning_node", PlanningNode())
    builder.add_node("code_generation_node", CodeGenerationNode())
    builder.add_node("write_code_node", WriteCodeNode())
    builder.add_node("debugging_node", DebuggingNode())

    builder.add_edge("detect_complexity_node", "structure_generation_node")
    builder.add_edge("structure_generation_node", "planning_node")
    builder.add_conditional_edges("planning_node", route_planning_codegeneration)
    builder.add_conditional_edges("debugging_node", debugging_route)
    builder.add_conditional_edges("code_generation_node", code_generation_node_route)
    builder.add_edge("write_code_node", "planning_node")

    builder.set_entry_point("detect_complexity_node")

    return builder.compile()

if __name__ == '__main__':
    save_dir = 'generate_software'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    query = ''
    while query == '':
        query = input('Please provide a development request: ')

    project_dir = ''
    while project_dir == '':
        project_dir = input('Please provide a directory to save the project: ')

    project_dir = os.path.join(save_dir, project_dir)

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    # Open a file to write the print statements to
    with open(os.path.join(project_dir, 'logs.txt'), 'w') as f:
        # Create a DualOutput object
        dual_output = DualOutput(f, sys.stdout)

        # Redirect standard output to the DualOutput object
        sys.stdout = dual_output

        print(
            'This is a graph-based software engineering agent that can create a whole complex software project (Author Junting Li)\n')

        initial_state = {
            'messages': [],
            'complexity': '',
            'llm_query': '',
            'project_dir': project_dir,
            'code_history': {},
            'debug_history': {},
            'structure': None,
            'current_index': None,

            'debugging': True,
            'debug_timeout': 10,
            'max_debugging_num': 10,
        }

        app_compiled = build_graph(query)
        app_compiled.invoke(initial_state, {"recursion_limit": 1000})

        sys.stdout = dual_output.console
