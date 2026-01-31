# app.py
# CPU-only Multi-RAG Agent
# Embeddings: Ollama nomic-embed-text
# LLM: google/flan-t5-base (HF Spaces compatible)

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# -------------------------------
# 1. Embeddings (Ollama - CPU)
# -------------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -------------------------------
# 2. Documents (Multi RAG sources)
# -------------------------------
tech_docs = [
    Document(page_content="RAG combines retrieval with generation."),
    Document(page_content="Transformers use self-attention mechanisms.")
]

api_docs = [
    Document(page_content="Authentication uses JWT tokens."),
    Document(page_content="Rate limit is 100 requests per minute.")
]

policy_docs = [
    Document(page_content="Refunds take 5-7 business days."),
    Document(page_content="Support email is support@example.com.")
]

# -------------------------------
# 3. Vector Stores
# -------------------------------
vs_tech = Chroma.from_documents(tech_docs, embeddings, collection_name="tech")
vs_api = Chroma.from_documents(api_docs, embeddings, collection_name="api")
vs_policy = Chroma.from_documents(policy_docs, embeddings, collection_name="policy")

# -------------------------------
# 4. RAG Tools
# -------------------------------
tech_tool = Tool(
    name="TechRAG",
    func=lambda q: vs_tech.similarity_search(q, k=3),
    description="For technical and ML-related questions"
)

api_tool = Tool(
    name="APIRAG",
    func=lambda q: vs_api.similarity_search(q, k=3),
    description="For API usage and backend questions"
)

policy_tool = Tool(
    name="PolicyRAG",
    func=lambda q: vs_policy.similarity_search(q, k=3),
    description="For policies, refunds, and support questions"
)

tools = [tech_tool, api_tool, policy_tool]

# -------------------------------
# 5. CPU LLM (HF Spaces)
# -------------------------------
model_id = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------------
# 6. Agent
# -------------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# -------------------------------
# 7. Run
# -------------------------------
if __name__ == "__main__":
    while True:
        query = input("\nAsk something (or 'exit'): ")
        if query.lower() == "exit":
            break
        response = agent.run(query)
        print("\nAnswer:", response)
