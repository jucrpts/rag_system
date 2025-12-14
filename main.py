import sys
import subprocess

# Auto-install dependencies if missing
try:
    import langchain
    import sentence_transformers
    import faiss
    import transformers
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"Missing dependencies: {e}. Installing from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed. Please run the script again.")
    sys.exit(0)

import os

# Load our document
with open("my_knowledge.txt") as f:
    knowledge_text = f.read()

# 1. Initialize the Text Splitter
# This splitter is smart. It tries to split on paragraphs ("\n\n"),
# then newlines ("\n"), then spaces (" "), to keep semantically
# related text together as much as possible.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,  # Max size of a chunk
    chunk_overlap=20, # Overlap to maintain context between chunks
    length_function=len
)

# 2. Create the chunks
chunks = text_splitter.split_text(knowledge_text)

print(f"We have {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---\n{chunk}\n")



from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
# 'all-MiniLM-L6-v2' is a fantastic, fast, and small model.
# It runs 100% on your local machine.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Embed all our chunks
# This will take a moment as it "reads" and "understands" each chunk.
chunk_embeddings = model.encode(chunks)

print(f"Shape of our embeddings: {chunk_embeddings.shape}")


import faiss
import numpy as np

# Get the dimension of our vectors (e.g., 384)
d = chunk_embeddings.shape[1]

# 1. Create a FAISS index
# IndexFlatL2 is the simplest, most basic index. It calculates
# the exact distance (L2 distance) between our query and all vectors.
index = faiss.IndexFlatL2(d)

# 2. Add our chunk embeddings to the index
# We must convert to float32 for FAISS
index.add(np.array(chunk_embeddings).astype('float32'))

print(f"FAISS index created with {index.ntotal} vectors.")



from transformers import pipeline

# 1. Load a "Question-Answering" or "Text-Generation" model
# We'll use a small, instruction-tuned model from Google.
generator = pipeline('text2text-generation', model='google/flan-t5-small')

# --- This is our RAG pipeline function ---
def answer_question(query):
    # 1. RETRIEVE
    # Embed the user's query
    query_embedding = model.encode([query]).astype('float32')

    # Search the FAISS index for the top k (e.g., k=2) most similar chunks
    k = 2
    distances, indices = index.search(query_embedding, k)

    # Get the actual text chunks from our original 'chunks' list
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    # 2. AUGMENT
    # This is the "magic prompt." We combine the retrieved context
    # with the user's query.
    prompt_template = f"""
    Answer the following question using *only* the provided context.
    If the answer is not in the context, say "I don't have that information."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # 3. GENERATE
    # Feed the augmented prompt to our generative model
    answer = generator(prompt_template, max_length=100)
    print(f"--- CONTEXT ---\n{context}\n")
    return answer[0]['generated_text']



query_1 = "What is the WFH policy?"
print(f"Query: {query_1}")
print(f"Answer: {answer_question(query_1)}\n")




