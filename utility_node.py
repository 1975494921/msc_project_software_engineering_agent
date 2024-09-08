from abstract import Node, State
from virtual_env import run_debugging


class WriteCodeNode(Node):
    def __init__(self):
        pass

    def __call__(self, state: State):
        code_name = list(state['structure'][state['current_index']].keys())[0]
        code = state['code_history'][code_name][-1]

        with open(f"{state['project_dir']}/{code_name}", 'w') as f:
            f.write(code)


class DebuggingNode(Node):
    def __init__(self):
        pass

    def __call__(self, state: State):
        code_key = list(state['structure'][state['current_index']].keys())[0]

        if code_key not in state['debug_history']:
            state['debug_history'][code_key] = {'errors': [], 'passed': False, 'num': 1}

        else:
            state['debug_history'][code_key]['num'] += 1

        output = run_debugging(state['code_history'], code_key)

        error = output['error']
        if error == '' or 'timeout' in error:
            passed = True
            state['debug_history'][code_key]['passed'] = True

        else:
            passed = False
            state['debug_history'][code_key]['errors'].append(error)

        print('-' * 100)
        print(f"Debugging {code_key}:")
        print(f'Passed: {passed}, {output}')
        print('-' * 100)

        return state


class PlanningNode(Node):
    def __init__(self):
        pass

    def __call__(self, state: State):
        if state['current_index'] == len(state['structure']):
            return state

        if state['current_index'] is None:
            state['current_index'] = 0

        else:
            state['current_index'] += 1

        return state
