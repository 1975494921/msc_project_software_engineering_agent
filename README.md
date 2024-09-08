# Graph-based Software Engineering Agent

Author: Junting Li

This agent is a graph-based software engineering agent that can be used to generate a whole software with only one sentence. It also include the automatic code debugging process, make sure the code is correct and efficient.

Before using this agent, you need to install the packages in the `requirements.txt` file. You can simply run the following command to install all the packages:

```bash

pip install -r requirements.txt

```

In addition, please put your OpenAI API key in the `llm.py` file, so that the agent can use the OpenAI API to generate the software.

To use this agent, you can simply run the following command:

```python
python main.py
```

Then you can input the sentence you want to generate the software, as well as the folder name to store, the agent will generate the whole software for you.

The following code indicates some parameters you can set in the `main.py` file:

```python
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
```

You can set the `debugging` parameter to `True` or `False` to enable or disable the debugging process. You can also set the `debug_timeout` and `max_debugging_num` to control the maximum time to debug a single code and the maximum number of debugging times, to prevent infinite loop.