import openai
import json
path = "ans_llm.json"


openai.api_key ="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"


def gpt_check(idx, c_set):
    """
    idx: c_set 中 的 correspondence 的index
    c_set: correspondence set of view
    """
    # check the cache
    with open(path, "r") as f:
        c_dic = json.load(f)
        if str(idx) in c_dic.keys():
            return c_dic[str(idx)]

    prompt = f"For a schema match task,Do Schema1:{c_set[idx][0][0]} attribute:{c_set[idx][0][1]}  Schema2:{c_set[idx][1][0]} attribute:{c_set[idx][1][1]}, please answer with 'yes or no'"
    
    response = openai.Completion.create(
    engine="text-davinci-002",  # You can specify the engine you want to use.
    prompt=prompt,
    max_tokens=150,  # Adjust this based on the length of responses you want.
    temperature=0.7,  # You can adjust the temperature for creativity.
    stop=None  # You can specify a list of words to stop generation if needed.
    )

    if "yes" in response.choices[0].text.lower():
        return "yes"
    else:
        return "no"
    