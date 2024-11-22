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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def perform_research(query):
    """
    Simple function to perform research using Tavily
    """
    try:
        # Initialize tools and LLM
        search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4-1106-preview"
        )
        
        # First, search for information
        search_results = search.invoke(query)
        
        # Then, use ChatGPT to summarize the results
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
    try:
        # Parse incoming message
        data = request.get_data().decode("utf-8")
        logger.info(f"Research AI received raw data: {data}")
        
        message = parse_message_from_agent(data)
        logger.info(f"Research AI parsed message: {message}")
        
        # Extract query and return address from message
        query = message.payload.get("query", "")
        sender_address = message.sender  # Get the sender's address
        logger.info(f"Processing query from {sender_address}: {query}")
        
        if not query:
            logger.error("No query provided in message payload")
            return {"status": "error", "message": "No query provided"}, 400
        
        # Perform research
        result = perform_research(query)
        logger.info(f"Research result generated: {result[:100]}...")  # Log first 100 chars
        
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

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "RESEARCH_AI_KEY", "AGENTVERSE_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")
    
    # Create identity from seed
    research_identity = Identity.from_seed(os.getenv("RESEARCH_AI_KEY"), 0)
    logger.info(f"Research AI started with address: {research_identity.address}")
    
    # Setup webhook
    webhook_url = "http://localhost:5001/webhook"
    logger.info(f"Webhook URL: {webhook_url}")
    
    # Enhanced readme for better discovery
    readme = """
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
    
    # Register with Agentverse
    register_with_agentverse(
        identity=research_identity,
        url=webhook_url,
        agentverse_token=os.getenv("AGENTVERSE_API_KEY"),
        agent_title="Research and Summary AI",
        readme=readme
    )
    logger.info("Registration successful")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5001)