from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os



file_path = r"D:\geneartive_ai\RAG\basic_rag\data\COMPANY_POLICY.txt"

with open(file_path, "r", encoding="utf-8") as f:
    file = f.read()


# 2. CLEANING

cleaned_data = file.strip()


# 3. CHUNKING

chunk_size = 300
overlap = 50

chunks = {}

start_index = 0
end_index = len(cleaned_data)
i = 0

while start_index < end_index:

    window_end = min(
        start_index + chunk_size,
        end_index
    )

    chunk = cleaned_data[start_index:window_end]

    label = f"chunk{i}"

    chunks[label] = chunk

    start_index += chunk_size - overlap

    i += 1


print("Total chunks:", len(chunks))


# 4. EMBEDDING

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

data = list(chunks.values())

embeddings = embedding_model.encode(data)

print("Embedding shape:", embeddings.shape)


# 5. CREATE FAISS INDEX

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# 6. CREATE ID → CHUNK MAPPING

id_to_chunk = {}

for i, chunk in enumerate(data):
    id_to_chunk[i] = chunk


# 7. CREATE INDEX DIRECTORY

os.makedirs("index", exist_ok=True)


# 8. SAVE FAISS INDEX

faiss.write_index(
    index,
    "index/faiss.index"
)


# 9. SAVE CHUNK MAPPING

with open("index/chunks.pkl", "wb") as f:
    pickle.dump(id_to_chunk, f)


print("\nIndexing completed successfully!")

print("Saved:")
print("→ index/faiss.index")
print("→ index/chunks.pkl")