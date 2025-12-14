# 📚 (Retrieval-Augmented Generation) 

This project implements a **fully local Retrieval-Augmented Generation (RAG) system** using Python. It reads a knowledge file, splits it into semantic chunks, stores embeddings in a FAISS vector index, retrieves relevant context for a query, and generates grounded answers using a language model.

---

## 🚀 Features

* Fully local (no OpenAI or paid APIs)
* Automatic dependency installation
* Smart text chunking with overlap
* Semantic search using FAISS
* Context-aware answer generation
* Hallucination control (answers only from context)

---

## 🧠 RAG Pipeline Overview

1. Load knowledge base from `my_knowledge.txt`
2. Split text into overlapping chunks
3. Generate embeddings using Sentence Transformers
4. Store vectors in FAISS
5. Retrieve relevant chunks for a query
6. Generate answer using retrieved context

---

## 📁 Project Structure

* main.py
* my_knowledge.txt
* requirements.txt
* README.md

---

## 🛠️ Tech Stack

* Python 3.9+
* LangChain
* Sentence-Transformers (`all-MiniLM-L6-v2`)
* FAISS
* Hugging Face Transformers
* Google FLAN-T5 (Small)

---

## 📦 Setup & Installation

### Clone the Repository

Clone the repository and move into the project directory.

### Add Knowledge File

Create a file named `my_knowledge.txt` and add your domain-specific content.

### Run the Script

Run the following command:

python main.py

If dependencies are missing, they will be automatically installed from `requirements.txt`.

---

## 🧪 Example Query

Example query used in the script:

What is the WFH policy?

### Sample Output

The model retrieves relevant context and generates a grounded answer based only on the retrieved text.

---

## 🔐 Hallucination Prevention

The model is explicitly instructed to answer **only using the provided context**. If the answer is not found, it responds with:

I don't have that information.

This ensures reliable and trustworthy results.

---

## 📈 Customization Ideas

* Increase chunk size for larger documents
* Adjust number of retrieved chunks (k value)
* Replace FAISS index type for scalability
* Swap FLAN-T5 with a different local model
* Add a CLI or Streamlit interface

---

## 🎯 Use Cases

* Internal company knowledge assistant
* Policy and documentation Q&A
* Learning RAG fundamentals
* Offline AI assistants
* Enterprise knowledge search

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Jayaaditya

If you find this useful, consider giving the repository a ⭐
