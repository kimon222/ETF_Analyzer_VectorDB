import os
import glob
from dotenv import load_dotenv
import gradio as gr

# imports for langchain, Chroma, and plotly
from langchain_community.document_loaders import DirectoryLoader, TextLoader  # updated import for TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# Load environment variables in a file called .env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Read in documents using LangChain's loaders
folders = [folder for folder in glob.glob("knowledge-base/*") if os.path.isdir(folder)]

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

# Text splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Debugging: Check if chunks are loaded properly
print(f"Total chunks: {len(chunks)}")

# Ensure that we have chunks to process
if len(chunks) == 0:
    raise ValueError("No chunks were created. Please check the document loading and splitting steps.")

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")

# Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
embeddings = OpenAIEmbeddings()

# Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch
db_name = "vector_db"  # Specify the db_name
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create our Chroma vectorstore!
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Get one vector and find how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

# Prework
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]

# Map document types to colors
colors = ['blue' if t == 'about_etfs' else
          'green' if t == 'investment_strategies' else
          'orange' if t == 'financial_regulations' else
          'purple' if t == 'etf_comparison' else
          'red' for t in doc_types]

# Handle perplexity for TSNE: perplexity must be less than n_samples
perplexity_value = min(len(vectors) - 1, 30)

# TSNE for dimensionality reduction
tsne = TSNE(n_components=3, perplexity=perplexity_value, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Text: {d[:100]}..." for d in documents],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D ETF Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
