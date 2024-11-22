from fetchai.crypto import Identity
from fetchai.registration import register_with_agentverse
from fetchai.communication import parse_message_from_agent, send_message_to_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from flask import Flask, request
import logging
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)

# Initialize global variables
research_identity = None

def perform_research(query):
    """
    Perform research using Tavily and OpenAI
    """
    try:
        search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4-1106-preview"
        )
        
        search_results = search.invoke(query)
        
        summary_prompt = f"""Based on the following search results, provide a comprehensive answer about {query}:
        
        {search_results}
        
        Please provide a clear and concise summary."""
        
        response = llm.invoke(summary_prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"Error in research: {str(e)}")
        return f"Error performing research: {str(e)}"

@app.route('/webhook', methods=['POST'])
def webhook():
    """Following the documentation's webhook pattern"""
    try:
        data = request.get_data().decode("utf-8")
        logger.info(f"Received data: {data}")
        
        # Parse the incoming message
        message = parse_message_from_agent(data)
        logger.info(f"Parsed message: {message}")
        
        # Get the sender's address
        sender = message.sender
        
        # Get the query from payload
        query = message.payload.get("query", "")
        logger.info(f"Processing query: {query}")
        
        if not query:
            return {"status": "error: no query provided"}
        
        # Perform the research
        result = perform_research(query)
        
        # Send response back to the requesting AI
        response = send_message_to_agent(
            research_identity,
            sender,
            {
                "result": result
            }
        )
        logger.info(f"Sent response: {response}")
        
        return {"status": "Agent message processed"}
        
    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        return {"status": f"error: {str(e)}"}

@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "healthy", "agent_address": research_identity.address if research_identity else None}

def init_app():
    """Initialize the application"""
    global research_identity
    
    try:
        # Create identity
        research_identity = Identity.from_seed(os.getenv("RESEARCH_AI_KEY"), 0)
        logger.info(f"Research AI started with address: {research_identity.address}")
        
        # Simplified readme following documentation
        readme = """
        <description>I help perform web research and provide detailed information about any topic</description>
        <use_cases>
            <use_case>Research and summarize any topic</use_case>
            <use_case>Answer complex questions with research</use_case>
        </use_cases>
        <payload_requirements>
        <description>What is needed in the request</description>
        <payload>
            <requirement>
                <parameter>query</parameter>
                <description>The topic or question you want researched</description>
            </requirement>
        </payload>
        </payload_requirements>
        """
        
        # Register with Agentverse
        register_with_agentverse(
            research_identity,
            os.getenv("WEBHOOK_URL", "https://research-ai-1-f7zf.onrender.com/webhook"),
            os.getenv("AGENTVERSE_API_KEY"),
            "Research Assistant",
            readme
        )
        logger.info("Registration successful")
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

# Initialize app when imported
load_dotenv()
init_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))