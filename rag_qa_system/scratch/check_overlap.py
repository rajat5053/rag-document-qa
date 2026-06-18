import pickle

chunks_path = "d:/Projects/RAG System/rag_qa_system/data/chunks.pkl"
with open(chunks_path, "rb") as fh:
    chunks = pickle.load(fh)

print("Chunk 2 word count:", len(chunks[2].split()))
print("Chunk 3 word count:", len(chunks[3].split()))

print("\nLast 10 words of Chunk 2:")
print(" ".join(chunks[2].split()[-10:]))

print("\nFirst 10 words of Chunk 3:")
print(" ".join(chunks[3].split()[:10]))
