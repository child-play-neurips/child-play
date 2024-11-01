import openai
from openai import OpenAI
from transformers import pipeline
# from google.cloud import aiplatform

# aiplatform.init(project="crafty-hall-429513-t0")

openai.api_key = ""

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
    
    elif origin == "vertex":
        endpoint = aiplatform.Endpoint(endpoint_name=f"projects/your-gcp-project-id/locations/us-central1/endpoints/{specific_model}")
        response = endpoint.predict(instances=[text_prompt], parameters={"temperature": temperature})
        response = response.predictions[0]

    elif origin == "ans":
        response = specific_model

    else:
        raise ValueError(f"Unknown origin: {origin}")

    response = response.strip()
    return response
