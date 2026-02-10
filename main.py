from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import math
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

load_dotenv()
app = FastAPI()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

index = faiss.read_index("social_index.faiss")
df = pd.read_csv("knowledge_base/Jan-01-2025_Dec-31-2025_4283743298543699.csv")
df = df.fillna(0)

class Query(BaseModel):
    query: str
    top_k: int = 5




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
def search_posts(query: Query):

    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query.query
    ).data[0].embedding

    q_embedding = np.array([q_embedding]).astype("float32")

    distances, indices = index.search(q_embedding, 15)  # retrieve more first

    candidates = []

    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]

        engagement = (
            row["Likes"] +
            row["Comments"] +
            row["Shares"] +
            row["Saves"]
        )

        engagement_rate = float(engagement / max(row["Reach"], 1))

        semantic_score = float(1 / (1 + dist))  # convert L2 distance to similarity

        final_score = float((0.7 * semantic_score) + (0.3 * engagement_rate))

        candidates.append({
            "description": str(row["Description"]),
            "engagement_rate": float(engagement_rate),
            "post_type": str(row["Post type"]),
            "publish_time": str(row["Publish time"]),
            "semantic_score": float(semantic_score),
            "final_score": float(final_score)
        })

    # sort by final score
    ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    return {"results": ranked[:query.top_k]}



@app.post("/analyze")
def analyze_posts(query: Query):

    # reuse search logic
    search_response = search_posts(query)
    results = search_response["results"]

    context = "\n\n".join([
        f"""
        Description: {r['description']}
        Engagement Rate: {r['engagement_rate']}
        Post Type: {r['post_type']}
        Published: {r['publish_time']}
        """
        for r in results
    ])

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior social media data strategist."},
            {"role": "user", "content": f"""
            Analyze the following high-performing posts:

            {context}

            Provide:
            1. Common themes
            2. Tone patterns
            3. Post type effectiveness
            4. Strategic recommendations
            Return structured bullet points.
            """}
        ]
    )

    return {
        "analysis": completion.choices[0].message.content,
        "supporting_posts": results
    }
