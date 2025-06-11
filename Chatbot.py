from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from neo4j import GraphDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.prompts import MessagesPlaceholde
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# --------------- Setup Environment & Models ---------------
load_dotenv()

# Load LangChain LLM (Groq Llama 3)
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# Neo4j settings (replace as needed)
uri = "bolt://localhost:<PORT>"  # Replace <PORT> with your Neo4j port
user = "<username>"  # Replace with your Neo4j username
password = "<password>"  # Replace with your Neo4j password
graphDB = GraphDatabase.driver(uri, auth=(user, password))

# Load DataFrame
file_path = "tcc_ceds_music.csv"
data = pd.read_csv(file_path, encoding="utf-8")
#remove the first column
data = data.iloc[:, 1:]

# Load FAISS index
vector_DB = faiss.read_index("songs.index")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --------------- Helper: Vector Search Function ---------------
def vector_search(query, data, model, index, top_n=10):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_n)
    return data.iloc[I[0]]

# --------------- LangChain Tools ---------------
@tool
def cypher_query(query: str) -> str:
    """
    Run a Cypher query in Neo4j and return results.
    Use this for specific song names, artist queries, or structured data searches.
    """
    try:
        with graphDB.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]
            return str(records) if records else "No results found"
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool
def semantic_search(query: str) -> str:
    """
    Use FAISS vector search for mood, vibe, or semantic music requests.
    Use this when users ask for music based on feelings, emotions, or abstract concepts.
    """
    try:
        results = vector_search(query, data, embedding_model, vector_DB)
        recommendations = results[['track_name', 'artist_name', 'genre']].to_dict(orient='records')
        return str(recommendations)
    except Exception as e:
        return f"Error in semantic search: {str(e)}"

# --------------- Agent System Prompt ---------------
system_prompt = """
You are a music recommendation assistant. Users will ask for songs by mood, genre, artist, or song name.

Guidelines:
- If the request mentions mood, vibe, feelings, or abstract concepts (like "happy songs", "sad music", "energetic tracks"), use the semantic_search tool.
- If the request mentions a specific song name, artist name, or needs structured data queries, use the cypher_query tool.
- Always explain your recommendations in a friendly way and provide context about why these songs match their request.
- Format your responses to be helpful and engaging.

Tools available:
- cypher_query: For specific song/artist searches and structured queries
- semantic_search: For mood-based and semantic music discovery
"""

# --------------- LangChain Agent Setup ---------------
tools = [cypher_query, semantic_search]

# Create the correct prompt template for tool-calling agent
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Use create_tool_calling_agent instead of create_structured_chat_agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=3,
    return_intermediate_steps=True
)

# --------------- Main Chat Loop ---------------
print("🎵 MusicBot is ready! Type your request (type 'quit' or 'exit' to stop):")
print("Try asking for:")
print("- Mood-based: 'I want happy songs' or 'recommend sad music'")
print("- Artist-specific: 'songs by [artist name]'")
print("- Genre-based: 'jazz recommendations'")
print("-" * 50)

conversation_history = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "exit"]:
        print("🎵 Thanks for using MusicBot! Goodbye!")
        break
    
    try:
        # Add user message to history
        conversation_history.append(f"Human: {user_input}")
        
        # Get response from agent
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": conversation_history
        })
        
        # Extract and display the output
        output = response.get("output", "Sorry, I couldn't process your request.")
        print(f"\n🎵 MusicBot: {output}")
        
        # Add bot response to history
        conversation_history.append(f"Assistant: {output}")
        
        # Keep conversation history manageable (last 10 exchanges)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please try rephrasing your request.")

# Close Neo4j connection
graphDB.close()