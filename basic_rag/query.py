
from sentence_transformers import SentenceTransformer
import ollama
import faiss
import pickle

query="How many paid leaves do employees get?"
embedding_model=SentenceTransformer("all-MiniLM-L6-v2")

query_embedding=embedding_model.encode([query])

#loaing  the faiss indexes
index=faiss.read_index("index/faiss.index")

#loading the maping
with open("index/chunks.pkl","rb") as f:
    maping_data=pickle.load(f)

#retriving the relavent information
k=3
distance,idx=index.search(query_embedding,k)
print(distance)
print(idx)

retrieved_chunks=[maping_data[id] for id in idx[0]]
# print(context)

context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are a helpful question-answering assistant.
Answer the user's question using only the information provided in the context.
If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."
Context:
{context}
Question:
{query}
Answer:
"""
# print(prompt)
response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# print(response)
answer = response["message"]["content"]

print("\nFinal Answer:")
print(answer)




