import openai
from openai import OpenAI
from transformers import pipeline

openai.api_key = "YOUR_API_KEY"
client = OpenAI(api_key=openai.api_key)

def ask(api_messages, temperature, model):
    origin = model.split(':')[0]
    specific_model = model.split(':')[1]
    
    text_prompt = "\n\n".join([message["role"] + ":\n" + message["content"] for message in api_messages])
    
    if origin == "oa":
        completion = client.chat.completions.create(
            model=specific_model,
            messages=api_messages,
            temperature=temperature
        )
        response = completion.choices[0].message.content

    elif origin == "hf":
        pipe = pipeline("text-generation", model=specific_model)
        result = pipe(text_prompt)
        full_text = result[0]['generated_text']
        response = full_text[len(text_prompt):]

    elif origin == "ans":
        response = specific_model

    else:
        raise ValueError(f"Unknown origin: {origin}")

    response = response.strip()
    return response
