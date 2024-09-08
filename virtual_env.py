import multiprocessing
import sys
import types
import io
import traceback


def create_module(module_name, code):
    """Create a module based on a name and code string."""
    module = types.ModuleType(module_name)
    module.__code__ = compile(code, module_name, 'exec')
    exec(module.__code__, module.__dict__)
    return module


def module_runner(queue, code_dict, main_module_name):
    try:
        created_modules = {}
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()

        sys.stdout = stdout
        sys.stderr = stderr

        for name, code in code_dict.items():
            module = types.ModuleType(name)
            exec(compile(code, name, 'exec'), module.__dict__)
            created_modules[name] = module
            sys.modules[name] = module

        main_module = created_modules[main_module_name]

        # Force the __name__ of the main module to be '__main__'
        main_module.__name__ = '__main__'

        # Execute the main module's code with the updated __name__
        exec(code_dict[main_module_name], main_module.__dict__)

        # Flush stdout and stderr into the queue
        stdout.flush()
        stderr.flush()

        queue.put({'output': stdout.getvalue(), 'error': stderr.getvalue()})

    except Exception as e:
        # Capture the detailed error information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        error_message = f"Error: {str(e)}\n\nTraceback:\n{tb_str}"
        queue.put({'output': stdout.getvalue(), 'error': error_message})

    finally:
        # Restore stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        stdout.close()
        stderr.close()

def run_virtual_env(code_dict, main_module_name):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=module_runner, args=(queue, code_dict, main_module_name))
    process.start()

    partial_output = {'error': '', 'output': ''}

    try:
        # Check if the process finishes within the timeout
        process.join(timeout=3)  # Set a reasonable timeout

        # Retrieve any available output before checking if the process is still alive
        while not queue.empty():
            try:
                result = queue.get_nowait()
                partial_output['output'] += result.get('output', '')
                partial_output['error'] += result.get('error', '')
            except queue.Empty:
                break

        if process.is_alive():
            # If the process is still alive after the timeout, terminate it
            process.terminate()
            partial_output['error'] += 'Process terminated due to timeout.'

    except Exception as e:
        partial_output['error'] += f"Exception occurred: {str(e)}"

    finally:
        process.join()  # Ensure the process is properly cleaned up

    return partial_output


def run_debugging(code_history, run_module):
    code_dict = {}
    for file_name, code_list in code_history.items():
        module_name = file_name.split('.')[0]
        code = code_list[-1]
        code_dict[module_name] = code

    run_module = run_module.split('.')[0]

    return run_virtual_env(code_dict, run_module)


if __name__ == '__main__':
    code_dict = {
        'a': """

def add(a, b):
    return a + b
""",
        'b': """
from a import add
print(add(1, 2))

"""
    }

    output = run_virtual_env(code_dict, 'a')
    print("Output:\n" + output['output'], end='')
    print('-' * 100)
    print("Error:\n" + output['error'], end='')
    print('-' * 100)
    print(output['error'] == '')