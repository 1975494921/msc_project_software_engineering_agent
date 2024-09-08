import re


def filter_code_block(text):
    pattern = r"```(\w+)\n(.*?)```"
    result = re.findall(pattern, text, re.DOTALL)
    if result:
        return result[0][-1]

    return text
