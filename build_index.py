import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("knowledge_base/Jan-01-2025_Dec-31-2025_4283743298543699.csv")

# Clean + compute engagement
df["engagement"] = (
    df["Likes"] +
    df["Comments"] +
    df["Shares"] +
    df["Saves"]
)

df["engagement_rate"] = df["engagement"] / df["Reach"].replace(0, 1)

texts = []

for _, row in df.iterrows():
    text = f"""
    Description: {row['Description']}
    Post Type: {row['Post type']}
    Engagement Rate: {row['engagement_rate']}
    """
    texts.append(text)

# Generate embeddings
embeddings = []

for text in texts:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "social_index.faiss")

print("Index built successfully.")
