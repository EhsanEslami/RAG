# Natural Language Toolkit : https://www.nltk.org/
import nltk
from nltk.tokenize import sent_tokenize
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
import json
# source : https://blog.demir.io/advanced-rag-implementing-advanced-techniques-to-enhance-retrieval-augmented-generation-systems-0e07301e46f4

nltk.download('punkt')  # Ensure the punkt tokenizer is downloaded


def sliding_window(text, window_size=3):
    """
    Generate text chunks using a sliding window approach (Python sliding window) and Natural Language Toolkit (NLTK)

    Args:
    text (str): The input text to chunk.
    window_size (int): The number of sentences per chunk.

    Returns:
    list of str: A list of text chunks.
    """
    sentences = sent_tokenize(text)
    return [' '.join(sentences[i:i+window_size]) for i in range(len(sentences) - window_size + 1)]


# Example usage
text = "This is the first sentence. Here comes the second sentence. And here is the third one. Finally, the fourth sentence."
chunks = sliding_window(text, window_size=3)
for chunk in chunks:
    print(chunk)
    print("-----")
    # here, you can convert the chunk to embedding vector
    # and, save it to a vector database


def convert_to_embeddings(chunks):
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for each chunk
    embeddings = model.encode(chunks)

    # Initialize ChromaDB using LangChain's Chroma integration
    embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = Chroma(
        embedding_function=embedding_function, 
        collection_name="text_chunks")

    # Add text chunks and their embeddings to the collection using LangChain's Chroma integration
    documents = [Document(
        page_content=chunk, 
        metadata={"embedding": json.dumps(embedding.tolist())}) 
        for chunk, embedding in zip(chunks, embeddings)]

    vector_store.add_documents(documents)

    print("Embeddings have been successfully stored in the ChromaDB collection.")
    return vector_store


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
