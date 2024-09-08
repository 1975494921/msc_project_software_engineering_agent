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
