import openai

api_key = ...

openai.api_key = api_key

def get_llm_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        stream=False
    )

    return response.choices[0].message["content"]

if __name__ == '__main__':
    messages = [
        {
            "role": "system",
            "content": "You are a Python developer working on a project. You need to write a Python function that takes a list of integers and returns the sum of the list."
        },
        {
            "role": "user",
            "content": "I need to write a Python function that takes a list of integers and returns the sum of the list."
        }
    ]

    response = get_llm_response(messages)
    print(response)
