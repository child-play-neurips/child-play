# google cloud imports    
# import vertexai
# from vertexai.generative_models import GenerativeModel
# vertexai.init(project="PROJECT_ID", location="us-central1")

# Mistral imports
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
# from mistral_api import key_mistral

# client = MistralClient(api_key=key_mistral)

# OpenAi imports
from utils.gpt_api import key_openai
from openai import OpenAI

client = OpenAI(api_key=key_openai)

# Hugging face imports
from transformers import pipeline

def ask(api_messages, temperature, model):
    origin = model.split(':')[0]
    specific_model = model.split(':')[1]
    
    text_prompt = "\n\n".join([message["role"] + ":\n" + message["content"] for message in api_messages])
    
    # OpenAI
    if origin == "oa":
        completion = client.chat.completions.create(
            model=specific_model,
            messages=api_messages,
            temperature=temperature
        )

        response = completion.choices[0].message.content

    # Hugging Face
    elif origin == "hf":
        pipe = pipeline("text-generation", model=specific_model)
        result = pipe(text_prompt)
        full_text = result[0]['generated_text']
        
        response = full_text[len(text_prompt):]
        
    elif origin == "ans":
        response = specific_model
    
    # Mistral
    # elif origin == "mi":
    #     model = specific_model
    #     chat_response = client.chat(
    #         model=model,
    #         messages=[ChatMessage(role="user", content=api_messages)]
    #     )

    #     response = chat_response.choices[0].message.content

    # Google Cloud
    # elif origin == "gc":
    #     model = GenerativeModel(model_name="gemini-1.0-pro-002")

    #     model_response = model.generate_content(text_prompt)

    #     response = model_response.text

    else:
        raise ValueError(f"Unknown origin: {origin}")

    response = response.strip()

    return response

