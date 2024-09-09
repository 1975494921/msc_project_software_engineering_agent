from openai import OpenAI

api_key = "..."

client = OpenAI(
    api_key=api_key,
)


def get_llm_response(messages):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=False
    )
    message_content = response.choices[0].message.content
    return message_content


if __name__ == '__main__':
    messages = [
        {
            "role": "system",
            "content": "You are a Python developer working on a project. You need to write a Python function that takes a list of integers and returns the sum of the list."
        },
        {
            "role": "user",
            "content": "Hello"
        }
    ]

    response = get_llm_response(messages)
    print(response)
