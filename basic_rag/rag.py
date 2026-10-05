
from sentence_transformers import SentenceTransformer
import faiss
import ollama


file_path = r"D:\geneartive_ai\RAG\basic_rag\data\COMPANY_POLICY.txt"
with open(file_path,"r",encoding="utf-8") as f:
    file=f.read()

# print(file)

# cleaning the data
cleaned_data=file.strip();

#chunking of data
chunk_size=300
overlap=50
chunks={}
start_index=0
end_index=len(cleaned_data)
i=0

while start_index<end_index:
    #  Calculate the end of the current window
    window_end = min(start_index + chunk_size, end_index)

    # Extract the chunk
    chunk = cleaned_data[start_index:window_end]

    # Label and store the chunk
    label = f"chunk{i}"
    chunks[label] = chunk

    # Move the start index forward, keeping the overlap
    start_index += chunk_size - overlap
    i += 1

# print("Total chunks:", len(chunks))

# for label, chunk in chunks.items():
#     print(f"\n{label}")
#     print(chunk)



embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

data=list(chunks.values())
# print(data)
embeddings=embedding_model.encode(data)
# print("Number of chunks:", len(data))
# print("Embedding shape:", embeddings.shape)
# print(embeddings[0])

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("Number of vectors:", index.ntotal)

# Create mapping: FAISS ID → Original chunk text

id_to_chunk = {
    i: chunk
    for i, chunk in enumerate(data)
}

# print(id_to_chunk)

# print(data)

query = "How many paid leaves do employees get?"
query_embedding = embedding_model.encode([query])

print("Query embedding shape:", query_embedding.shape)

# Number of chunks we want
k = 3

# Search FAISS
distances, ids = index.search(query_embedding, k)

print("\nFAISS IDs:")
print(ids)

print("\nDistances:")
print(distances)

# Convert FAISS IDs back to original chunks
retrieved_chunks = [
    id_to_chunk[idx]
    for idx in ids[0]
]

# Display retrieved chunks
# print("\nRetrieved Chunks:")

# for i, chunk in enumerate(retrieved_chunks):
#     print(f"\nRetrieved Chunk {i}")
#     print(chunk)

#context creation
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
print(prompt)
response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response)
answer = response["message"]["content"]

print("\nFinal Answer:")
print(answer)