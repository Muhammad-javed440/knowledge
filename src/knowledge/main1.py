# 
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import os
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage

# Get the GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Define the Google Embedder Config
google_embedder = {
    "provider": "google",
    "config": {
        "model": "models/text-embedding-004",
        "api_key": GEMINI_API_KEY,
    }
}

# Create a knowledge source
content = "Users name is Muhammad javed. He is 45 years old and lives in Lahore, Pakistan. Working as Chief Data Scientist at CancerClarity."
string_source = StringKnowledgeSource(content=content)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash-exp",
    api_key=GEMINI_API_KEY,
    temperature=0,
)

# Create an agent with the knowledge store
agent = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory="You are a master at understanding people and their preferences.",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm,
    embedder=google_embedder
)

# Define a task
task = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

# Ensure the storage directory exists
os.makedirs("./my_crew2/short_term1", exist_ok=True)

# Create the Crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,   # To view items
    memory=True,  # Fixed typo
    # Long-term memory for persistent storage across sessions
    long_term_memory=LongTermMemory(
        storage=LTMSQLiteStorage(db_path="./my_crew2/long_term/long_term_memory_storage1.db")
    ),
    # Short-term memory for current context using RAG
    short_term_memory=ShortTermMemory(
        storage=RAGStorage(
            embedder_config=google_embedder,
            type="short_term",
            path="./my_crew2/short_term1/"
        )
    ),
    # Entity Memory for tracking key information about entities
    entity_memory=EntityMemory(
        storage=RAGStorage(
            embedder_config=google_embedder,
            type="short_term",
            path="./my_crew2/entity1/"
        )
    ),
    process=Process.sequential,
    knowledge_sources=[string_source],
    embedder=google_embedder
)

# Function to execute the crew task
def kickoff1():
    result = crew.kickoff(inputs={"question": "What city does Muhammad Javed live in and how old is he?"})
    print(result)

# Run the function
kickoff1()
