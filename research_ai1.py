# app/main.py
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

def init_app():
    """Initialize the application"""
    global research_identity
    
    try:
        # Check for required environment variables
        required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "RESEARCH_AI_KEY", "AGENTVERSE_API_KEY"]
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"Missing required environment variable: {var}")
        
        # Create identity
        research_identity = Identity.from_seed(os.getenv("RESEARCH_AI_KEY"), 0)
        logger.info(f"Research AI started with address: {research_identity.address}")
        
        # Setup webhook
        webhook_url = os.getenv("WEBHOOK_URL", "https://research-ai.onrender.com/webhook")
        logger.info(f"Webhook URL: {webhook_url}")
        
        # Register with Agentverse
        register_with_agentverse(
            identity=research_identity,
            url=webhook_url,
            agentverse_token=os.getenv("AGENTVERSE_API_KEY"),
            agent_title="Research and Summary AI",
            readme=get_readme()
        )
        logger.info("Registration successful")
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

def get_readme():
    """Get the readme for agent registration"""
    return """
    <description>
    A powerful AI research assistant that performs comprehensive web research and provides detailed summaries on any topic. 
    Utilizes advanced search capabilities and natural language processing to deliver accurate, well-structured information.
    </description>
    <use_cases>
        <use_case>Research and summarize any topic or question</use_case>
        <use_case>Provide detailed analysis of current events</use_case>
        <use_case>Generate comprehensive information reports</use_case>
        <use_case>Answer complex questions with web research</use_case>
    </use_cases>
    <keywords>
        research, information, knowledge, summary, analysis, search, web research, query
    </keywords>
    <payload_requirements>
        <description>Requirements for research requests</description>
        <payload>
            <requirement>
                <parameter>query</parameter>
                <description>The question or topic you want researched</description>
                <type>string</type>
                <required>true</required>
            </requirement>
        </payload>
    </payload_requirements>
    """

def perform_research(query):
    """
    Perform research using Tavily and OpenAI
    """
    try:
        # Initialize tools and LLM
        search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4-1106-preview"
        )
        
        # Search for information
        logger.info(f"Performing search for query: {query}")
        search_results = search.invoke(query)
        
        # Generate summary
        summary_prompt = f"""Based on the following search results, provide a comprehensive answer about {query}:
        
        {search_results}
        
        Please provide a clear and concise summary."""
        
        logger.info("Generating summary")
        response = llm.invoke(summary_prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"Error in research: {str(e)}")
        return f"Error performing research: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent_address": research_identity.address if research_identity else None}

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint for receiving queries"""
    try:
        # Parse incoming message
        data = request.get_data().decode("utf-8")
        logger.info(f"Research AI received raw data: {data}")
        
        message = parse_message_from_agent(data)
        logger.info(f"Research AI parsed message: {message}")
        
        # Extract query and return address
        query = message.payload.get("query", "")
        sender_address = message.sender
        logger.info(f"Processing query from {sender_address}: {query}")
        
        if not query:
            logger.error("No query provided in message payload")
            return {"status": "error", "message": "No query provided"}, 400
        
        # Perform research
        result = perform_research(query)
        logger.info(f"Research result generated: {result[:100]}...")
        
        # Send result back
        logger.info(f"Attempting to send result back to {sender_address}")
        try:
            response = send_message_to_agent(
                research_identity,
                sender_address,
                {
                    "status": "success",
                    "result": result
                }
            )
            logger.info(f"Successfully sent response: {response}")
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
            raise
            
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}, 500

# Initialize app when imported
load_dotenv()
init_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

