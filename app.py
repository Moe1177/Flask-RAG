import os
import json
import threading
import traceback
from dotenv import load_dotenv
from datetime import datetime

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from pymongo import MongoClient

from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from git import Repo
from openai import OpenAI
from langchain.schema import Document
from pinecone import Pinecone

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------------------------------------------------------------------------

MONGO_URI = os.environ.get('MONGO_URI')
if not MONGO_URI:
    raise Exception("MONGO_URI not set in environment variables")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.get_default_database()
chats_collection = db.chats


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise Exception("PINECONE_API_KEY not set in environment variables")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise Exception("OPENROUTER_API_KEY not set in environment variables")

# GitHub repository URL – default provided, can be overridden in .env
GITHUB_REPO_URL = os.environ.get("GITHUB_REPO_URL")
if not GITHUB_REPO_URL:
    raise Exception("GITHUB_REPO_URL not set in environment variables")

REPO_PATH = f"/codebase/SOEN341_Winter2025"

# Use the GitHub repo URL as the Pinecone namespace
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", GITHUB_REPO_URL)

# ---------------------------------------------------------------------------

# Supported file extensions and ignored directories
SUPPORTED_EXTENSIONS = {'.tsx', '.jsx'}
IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor', ".idea", ".env", ".venv"}

# ---------------------------------------------------------------------------

# Global vectorstore variable
vectorstore = PineconeVectorStore(
    index_name="codebase-rag", 
    embedding=HuggingFaceEmbeddings()
)

# Initialize Pinecone client and connect to the index
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("codebase-rag")

# Setup OpenRouter client for RAG calls
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ---------------------------------------------------------------------------

def clone_or_update_repo():
    """Clones the repo if not present; otherwise, pulls latest changes."""
    if os.path.exists(REPO_PATH):
        try:
            repo = Repo(REPO_PATH)
            repo.remotes.origin.pull()
            print("Repository updated successfully.")
        except Exception as e:
            print("Failed to pull latest changes:", str(e))
    else:
        repo_name = GITHUB_REPO_URL.split('/')[-1]
        repo_path = f"/codebase/{repo_name}"
        Repo.clone_from(GITHUB_REPO_URL, repo_path)
        print("Repository cloned successfully.")

def get_file_content(file_path, REPO_PATH):
    """
    Returns the relative file name and its content.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        rel_path = os.path.relpath(file_path, REPO_PATH)
        return {"name": rel_path, "content": content}
    except Exception as e:
        print(f"Error reading file: {file_path}: {str(e)}")
        return None

def get_main_files_content(REPO_PATH):
    """
    Walks the repository and retrieves contents of supported files,
    ignoring directories defined in IGNORED_DIRS.
    """
    try:
        files_content = []
        for root, _, files in os.walk(REPO_PATH):
            if any(ignored in root for ignored in IGNORED_DIRS):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file_path)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, REPO_PATH)

                    if file_content:
                        files_content.append(file_content)
    except Exception as e:
        print(f"Error reading files: {str(e)}")
        return None
    return files_content
    

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def initialize_vectorstore():
    """
    Clones the repository, extracts file contents, builds Documents,
    and creates/updates the Pinecone vector store.
    """
    global vectorstore
    try:
        clone_or_update_repo()
        print("Cloning repository and initializing vectorstore...")
        files_content = get_main_files_content(REPO_PATH)
        documents = []
        for file in files_content:
            doc = Document(
                page_content=f"{file['name']}\n\n{file['content']}",
                metadata={"text": file['content'], "source": file['name']}
            )
            documents.append(doc)
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=HuggingFaceEmbeddings(),
            index_name="codebase-rag",
            namespace=NAMESPACE
        )
        print("Vectorstore initialized successfully.")
    except Exception as e:
        print("Error initializing vectorstore:", e)
        traceback.print_exc()


initialize_vectorstore()


def perform_rag(query, model="deepseek/deepseek-r1-distill-llama-70b:free"):
    """
    Performs Retrieval Augmented Generation:
      1. Computes query embedding.
      2. Retrieves top matches from Pinecone.
      3. Constructs an augmented query.
      4. Calls the OpenRouter API to generate a response.
    """
    try:
        raw_query_embedding = get_huggingface_embeddings(query)
        top_matches = pinecone_index.query(
            vector=raw_query_embedding.tolist(),
            top_k=3,
            include_metadata=True,
            namespace=NAMESPACE
        )
        # Extract context from the matched documents
        contexts = [item['metadata']['text'] for item in top_matches['matches']]
        augmented_query = "\n" + "\n\n-------\n\n".join(contexts[:10]) + \
                          "\n-------\n\n\n\n\nMY QUESTION:\n" + query

        system_prompt = (
            "You have ultimate knowledge over this codebase.\n"
            "You are an AI agent that helps users navigate the website to help them do what they need to do.\n"
            "You do not answer any questions about the backend, about any configuration file, or anything that has nothing to do with a functionality from the website.\n"
            "A good user request would be for example how can I create a channel or how can I log in.\n"
            "A bad user request would be how was this component of the website built, or what technology was used in this part of the website.\n"
            "Also do not release any sensitive information. Do not hallucinate. Answer the user's question by following the previous instructions. Consider the entire context provided to answer the user's question.\n"
            "If any invasive questions are asked, ONLY reply that this information cannot be given out due to privacy, and security reasons.\n"
            "You should not even give out one word of information on tech stack or anything else. Make all answers clear, and concise and easy to understand for the user.\n"
            "Also, make the answers straight to the point as well as be polite."
        )

        llm_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
        return llm_response.choices[0].message.content
    except Exception as e:
        print("Error in perform_rag:", e)
        traceback.print_exc()
        return "Sorry, I encountered an error processing your request."


@app.route('/webhook', methods=['POST'])
def github_webhook():
    """
    Receives GitHub webhook events and triggers an update of embeddings.
    In production, consider validating the webhook signature.
    """
    payload = request.json
    print("Received GitHub webhook payload:", json.dumps(payload, indent=2))
    threading.Thread(target=update_embeddings, args=(payload,)).start()
    return jsonify({'status': 'processing'}), 202

def update_embeddings(github_payload):
    """
    Updates the vector store by re-cloning the repository and rebuilding embeddings.
    For a more efficient production solution, perform incremental updates.
    """
    try:
        print("Updating embeddings based on GitHub payload...")
        initialize_vectorstore()
        print("Embeddings update complete.")
    except Exception as e:
        print("Error updating embeddings:", e)
        traceback.print_exc()


@socketio.on('chat_message')
def handle_chat_message(data):
    """
    Receives chat messages from the client, processes them via RAG,
    stores conversation in MongoDB, and returns the response.
    Expected data: { 'message': '<user message>' }
    """
    user_message = data.get('message')

    if not user_message:
        emit('chat_response', {'response': 'Invalid message.'})
        return

    chats_collection.insert_one({
        'sender': 'user',
        'message': user_message,
        'timestamp': datetime.now()
    })

    response_text = perform_rag(user_message)

    chats_collection.insert_one({
        'sender': 'bot',
        'message': response_text,
        'timestamp': datetime.now()
    })

    emit('chat_response', {'response': response_text})


if __name__ == '__main__':
    # In production, disable debug and use an async WSGI server if possible.
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
