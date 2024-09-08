import copy
import json

from abstract import Node, State
from llm import get_llm_response
from regular_expression import filter_code_block


class CodeGenerationNode(Node):
    def __init__(self):
        pass

    def __call__(self, state: State):
        code_key = list(state['structure'][state['current_index']].keys())[0]

        generate_code = self.code_generation_multifile(state, code_key)
        generate_code = filter_code_block(generate_code)

        if code_key not in state['code_history']:
            state['code_history'][code_key] = []

        state['code_history'][code_key].append(generate_code)

        return state

    def code_generation_multifile(self, state, code_key):
        llm_query = state['llm_query']
        structure_explanation = ''
        counter = 1
        for part in state['structure']:
            filename, explanation = list(part.keys())[0], list(part.values())[0]
            structure_explanation += f"{counter}.{filename}: {explanation}\n"
            counter += 1

        query = f"{llm_query}\nThe project is separate into the following part:\n{structure_explanation}"
        already_written_keys = [key for key in state['code_history'].keys() if key != code_key]

        if len(already_written_keys) > 0:
            query += "\nSome codes have been written:\n"
            for key in already_written_keys:
                query += f"{key}:\n{state['code_history'][key][-1]}\n" + '#### split code ####' + '\n'

            query += '\nYou can import the code above with the filenames and use them in the current part.'

        query += f"\nWhen writting codes, please takes into account the parts that are yet to be written, which they may need the current code to implement their functionality. Please write the code for the {code_key}:"

        messages = [
            {"role": "system",
             "content": "You are a python coding assistant that will generate code based on the user's request. Response should be a python code snippet only. If this code only contains auxiliary functions or components (classes), try to test to run them or create instance in the __main__ function."},
            {"role": "user", "content": query},
        ]

        if state['debugging']:
            if (code_key in state['debug_history'] and
                    state['debug_history'][code_key]['num'] > 0 and
                    not state['debug_history'][code_key]['passed']):
                error_msg = state['debug_history'][code_key]['errors'][-1]
                last_generated_code = state['code_history'][code_key][-1]
                messages.append({"role": "assistant", "content": last_generated_code})
                messages.append({"role": "user",
                                 "content": f"There are some errors in the last generated code:\n{error_msg}.\n Please correct the code snippet."})

        response = get_llm_response(messages)
        print('-' * 100)
        print(f"{code_key}:")
        print(f'query:\n{query}\n')
        print('-' * 100)
        print(f'response:\n{response}')
        print('-' * 100)

        return response


class StructureGenerationNode(Node):
    def __init__(self):
        pass

    def __call__(self, state: State):
        complexity = state['complexity']
        llm_query = state['llm_query']

        if complexity == 'single':
            # Generate single file code
            exit(0)
        else:
            # Generate multi file code
            structure = self.generate_structure(llm_query)

        state['structure'] = structure

        return state

    def generate_structure(self, query) -> list:
        messages = [
            {"role": "system",
             "content": "You are tasked with serving as a Python project structure assistant. You will receive development requirements from users and break down the project into distinct, manageable Python files. Each file needs to detail in describing in text that what will be implemented for this section, with the filenames ending in .py used as title. You do not need to write code. Organize these parts from the simplest to the most complex, ensuring that each part can import functionality from the preceding ones. This structure should avoid including any markdown files or folders, the response should not include introduction or conclusion. The structure can be not limited to the most basic functions and can include more components related to it. Keep in mind that earlier parts cannot import later parts, but later parts can import earlier parts. If main file exists, which use all of the other parts to implement the main functionality always be the last part."},
            {"role": "user", "content": query},
        ]

        response = get_llm_response(messages)
        print(response)
        print('-' * 100)

        messages = [
            {"role": "system",
             "content": "You are tasked to reconstruct the enumerated parts in a text to a python list. The title of each part is usually a filename and the content is its description. For the output, each element is a dict, with only an item that the key is the filename (.py) of a particular part and the value is its description. Please ignore any textual explanations that don't belong to any enumerated parts. Also the response needs to be able to be loaded directly into Python using json.load."},
            {"role": "user", "content": response},
        ]

        response = get_llm_response(messages)
        response = filter_code_block(response)
        print(response)
        print('-' * 100)

        list_obj = json.loads(response)
        return list_obj


class DetectComplexityNode(Node):
    def __init__(self, llm_query: str):
        self.llm_query = llm_query

    def __call__(self, state: State):
        complexity = self.get_complexity(self.llm_query)

        state['llm_query'] = self.llm_query
        print('The complexity of the project structure:', complexity)
        if 'single' in complexity:
            state['complexity'] = 'single'

        else:
            state['complexity'] = 'multi'

        return state

    def get_complexity(self, query):
        messages = [
            {"role": "system",
             "content": "You are a coding assistant where the user will enter a development request and you need to help the user decide whether to use a single file or multi-file structure based on complexity. Response single-file or multi-file only."},
            {"role": "user", "content": query},
        ]

        response = get_llm_response(messages)
        while (('single' not in response and 'multi' not in response) or
               ('single' in response and 'multi' in response)):
            messages_temp = copy.deepcopy(messages)
            messages_temp.append({"role": "assistant", "content": response})
            messages_temp.append({"role": "user",
                                  "content": "Your last response is not clear. Please provide a single-file or multi-file response only."})

            response = get_llm_response(messages_temp)

        return response
