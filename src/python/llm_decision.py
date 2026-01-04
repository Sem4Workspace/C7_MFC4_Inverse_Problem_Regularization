# # import openai
# # import json

# # def query_llm(prompt, model="gpt-4"):
# #     """
# #     Queries the LLM with a structured prompt and returns parsed JSON output.
# #     """

# #     response = openai.ChatCompletion.create(
# #         model=model,
# #         messages=[
# #             {"role": "system", "content": "You are a scientific reasoning assistant."},
# #             {"role": "user", "content": prompt}
# #         ],
# #         temperature=0.0
# #     )

# #     content = response["choices"][0]["message"]["content"]

# #     try:
# #         decision = json.loads(content)
# #     except json.JSONDecodeError:
# #         raise ValueError("LLM output is not valid JSON")

# #     return decision

# from openai import OpenAI
# import json

# client = OpenAI()  # uses OPENAI_API_KEY from environment

# def query_llm(prompt, model="gpt-4o-mini"):
#     """
#     Queries the LLM with a structured prompt and returns parsed JSON output.
#     """

#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a scientific reasoning assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.0
#     )

#     content = response.choices[0].message.content

#     try:
#         decision = json.loads(content)
#     except json.JSONDecodeError:
#         raise ValueError(
#             "LLM output is not valid JSON. Output was:\n" + content
#         )

#     return decision



# import os
# import json
# from google import genai

# # Create Gemini client
# client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# def query_llm(prompt, model="gemini-1.0-pro"):
#     """
#     Queries Gemini LLM and returns structured JSON decision.
#     """

#     response = client.models.generate_content(
#         model=model,
#         contents=prompt,
#         config={
#             "temperature": 0.0,
#             "response_mime_type": "application/json"
#         }
#     )

#     content = response.text

#     try:
#         decision = json.loads(content)
#     except json.JSONDecodeError:
#         raise ValueError(
#             "Gemini output is not valid JSON:\n" + content
#         )

#     return decision

from groq import Groq
import json
import os
os.getenv("GROQ_API_KEY")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def query_llm(prompt, model="llama-3.3-70b-versatile"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    content = response.choices[0].message.content
    return json.loads(content)
