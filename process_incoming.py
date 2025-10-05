import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
from openai import OpenAI
from config import api_key

# client = OpenAI(api_key=api_key)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        #"model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    
    response = r.json()
    return response

# def inference_openai(prompt):
#     response = client.responses.create(
#         model="gpt-5",
#         input=prompt
#     )
#     return response.output_text

def inference_openai(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-5",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )
    return response.choices[0].message.content

# --------- RAG utility and public API ---------
_DF_CACHE = None


def _load_embeddings_df():
    global _DF_CACHE
    if _DF_CACHE is None:
        _DF_CACHE = joblib.load('embeddings.joblib')
    return _DF_CACHE


def _to_minutes(seconds):
    return round(float(seconds) / 60.0, 2)


def _build_prompt(context_df, incoming_query):
    return f"""You are an AI assistant helping students in the *Sigma Web Development Course*.
Below are the subtitle chunks from the course videos. Each chunk contains the following:
- Video title
- Video number
- Start time (in seconds)
- End time (in seconds)
- The spoken text during that time

{context_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

-------------------------------------------------------------------------------------------------------------
User Query:
{incoming_query}

Instructions for Response:
1. If the user's question is related to the course or the video content:
   - Identify which video explains the concept.
   - Clearly mention the video title and video number.
   - Convert the timestamps from seconds to **minutes (rounded to two decimals)**.
   - Explain naturally and helpfully what is being taught at that point in the video.
   - Politely guide the user to watch that specific portion of the video for better understanding.
   - Respond in a human, conversational tone (avoid robotic or overly formal language).

2. If the user asks something **unrelated to the Sigma Web Development Course or its content**, reply politely:
   > "Sorry, I can only answer questions related to the Sigma Web Development Course."

3. If the user asks **who made you** or any similar question, reply:
   > "I was created by Karan Verma, who is 21 years old and currently pursuing his B.Tech final year in Bhopal."

4. Do **not mention** or refer to the subtitle chunk format, data structure, or any internal representation in your response.

Your goal is to provide a helpful, natural explanation like a friendly human tutor.
"""


def answer_query(incoming_query: str) -> str:
    df = _load_embeddings_df()
    if df is None or len(df) == 0:
        raise RuntimeError("Embeddings are not available. Please generate 'embeddings.joblib'.")

    question_embedding = create_embedding([incoming_query])[0]
    similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
    top_results = 5
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.loc[max_indx]

    try:
        new_df = new_df.copy()
        new_df['start_min'] = new_df['start'].apply(_to_minutes)
        new_df['end_min'] = new_df['end'].apply(_to_minutes)
    except Exception:
        pass

    prompt = f'''I am teaching web devlopment in my sigma web development course. Here are video subtitle chunk containing video title, video number, start time in seconds, end time in seconds, the text at that time:\n\n{new_df[["title" , "number", "start", "end", "text"]].to_json(orient="records")}\n\n-------------------------------------------------------------------------------------------------------------\n{incoming_query}\nUser ask this question related to the video chunk, you have to answer in human way(dont mention the above format, its just for you) where is and how much content is taught in which video (in which and at what timestamp) and guide the user to go to that particular video and return time in minutes not seconds. If user asks unrelated question, tell him that you can only answer question related to the course\n'''

    try:
        response_text = inference_openai(prompt)
    except Exception:
        local = inference(prompt)
        response_text = local.get("response", "") if isinstance(local, dict) else str(local)

    try:
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
    except Exception:
        pass

    try:
        with open("response.text", "w", encoding="utf-8") as f:
            f.write(response_text)
    except Exception:
        pass

    return response_text

# Removed interactive CLI to prevent terminal prompts when running the project
