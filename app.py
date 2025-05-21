import uvicorn
import os
import nest_asyncio
from dotenv import load_dotenv
import asyncio
from fastapi import FastAPI, HTTPException, Form, Response
from fastapi.responses import StreamingResponse, PlainTextResponse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from pymongo import MongoClient
from datetime import datetime, timedelta
from twilio.twiml.messaging_response import MessagingResponse
import re

# Apply nest_asyncio for running inside Jupyter Notebook (optional, remove if not using notebook)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Load API keys securely
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
SMS_ORG_UNIQUE_ID = os.getenv("SMS_ORG_UNIQUE_ID", "default_sms_org")

# --- MongoDB Client Initialization ---
client = MongoClient(MONGO_URI)
db = client["Chatbots_Appointments"]
collection = db["user_queries"]
organization_collection = db["organizations"]
patient_data_collection = db["Appointments_Booked"]
# --- End MongoDB Client Initialization ---

qdrant_collection = "Chabot_Queries"

# Configure Google Gemini API
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.6)

# Initialize Qdrant Client
from qdrant_client.models import VectorParams, Distance

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Dictionary to store per-user chat history for SMS sessions
sms_sessions: dict[str, list[AIMessage | HumanMessage]] = {}


# --- Organization Info Fetching ---
def get_organization_info(unique_id):
    org_info = organization_collection.find_one(
        {"unique_id": unique_id},
        {
            "_id": 0,
            "unique_id": 1,
            "organization_name": 1,
            "organization_description": 1,
            "organization_address": 1,
            "organization_phone_number": 1,
            "organization_email": 1,
            "organization_google_address": 1,
            "organization_expertise": 1
        }
    )
    return org_info if org_info else {
        "unique_id": unique_id,
        "organization_name": "Medical AI Assistant",
        "organization_description": "An AI-powered assistant for medical queries & Appointments Booking.",
        "organization_address": "Not available",
        "organization_phone_number": "Not available",
        "organization_email": "Not available",
        "organization_google_address": "Not available",
        "organization_expertise": "general medical queries"
    }

def fetch_organization_info(x):
    """Fetches organization info based on unique_id and adds it to the input dictionary."""
    unique_id = x.get("unique_id")
    if not unique_id:
         unique_id = x.get("SMS_ORG_UNIQUE_ID", "default_org_id")
         print(f"Warning: unique_id not found in input, using fallback: {unique_id}")

    org_info = get_organization_info(unique_id)
    current_time = datetime.now()
    min_allowed_datetime = current_time + timedelta(hours=1)
    formatted_min_time = min_allowed_datetime.strftime("%H:%M")
    formatted_min_date = current_time.strftime("%Y-%m-%d")

    x["organization_name"] = org_info.get("organization_name", "Medical AI Assistant")
    x["organization_description"] = org_info.get("organization_description", "An AI-powered assistant for medical queries & Appointments Booking.")
    x["organization_address"] = org_info.get("organization_address", "Not available")
    x["organization_phone_number"] = org_info.get("organization_phone_number", "Not available")
    x["organization_email"] = org_info.get("organization_email", "Not available")
    x["organization_expertise"] = org_info.get("organization_expertise", "general medical queries")
    x["organization_google_address"] = org_info.get("organization_google_address", "Not available")

    x["formatted_min_time"] = formatted_min_time
    x["formatted_min_date"] = formatted_min_date

    return x

fetch_org_runnable = RunnableLambda(fetch_organization_info)
# --- End Organization Info Fetching ---


# --- Embedding Model and RAG ---
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY
)

def get_related_text(query, unique_id, qdrant_collection_name, top_k=4):
    """Fetch relevant stored text from Qdrant."""
    if not unique_id:
         unique_id = SMS_ORG_UNIQUE_ID
         print(f"Warning: unique_id not found for RAG, using fallback: {unique_id}")

    query_embedding = embeddings_model.embed_query(query)
    collection_to_query = qdrant_collection_name if qdrant_collection_name else "new_practice"

    try:
        search_results = qdrant_client.query_points(
            collection_name=collection_to_query,
            query=query_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="unique_id", match=MatchValue(value=unique_id))]
            ),
            limit=top_k
        )

        related_texts = [
            result.payload.get("text", "No Context available") for result in search_results.points
        ]
        print(f"Related Chunks Found for unique_id '{unique_id}': {len(related_texts)}")

        return " ".join(related_texts) if related_texts else "No relevant context found."
    except Exception as e:
         print(f"Error fetching from Qdrant for unique_id '{unique_id}': {e}")
         return "Error retrieving context."

fetch_related_text_runnable = RunnableLambda(
    lambda x: {
        **x,
        "related_text": get_related_text(x["query"], x.get("unique_id"), x.get("qdrant_collection", "new_practice"))
    }
)
# --- End Embedding Model and RAG ---


# --- Chat History Formatting ---
def format_chat_history(chat_history_list: list[AIMessage | HumanMessage]):
    """Format the chat history list for the LLM prompt."""
    formatted_history = ""
    for message in chat_history_list:
        if isinstance(message, HumanMessage):
            formatted_history += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"AI: {message.content}\n"
    return formatted_history if chat_history_list else "No previous conversation history."
# --- End Chat History Formatting ---


# --- Appointment Detail Extraction Function ---
def extract_appointment_details(chat_history_list):
    """Extract appointment details from the chat history."""
    print("Extracting appointment details from chat history...")
    chat_text = format_chat_history(chat_history_list)
    
    # Define regex patterns for extraction
    name_pattern = r"(?:Name|name):\s*([^,\n]+)"
    phone_pattern = r"(?:Phone\s*(?:Number|number)|phone):\s*([^,\n]+)"
    reason_pattern = r"(?:Reason|reason):\s*([^,\n]+)"
    date_pattern = r"(?:Date|date|Appointment\s*Date):\s*([^,\n]+)"
    time_pattern = r"(?:Time|time|Appointment\s*Time):\s*([^,\n]+)"
    
    # Extract values using regex
    name_match = re.search(name_pattern, chat_text)
    phone_match = re.search(phone_pattern, chat_text)
    reason_match = re.search(reason_pattern, chat_text)
    date_match = re.search(date_pattern, chat_text)
    time_match = re.search(time_pattern, chat_text)
    
    # Create dictionary of appointment details
    appointment_details = {
        "name": name_match.group(1).strip() if name_match else None,
        "phone": phone_match.group(1).strip() if phone_match else None,
        "reason": reason_match.group(1).strip() if reason_match else None,
        "date": date_match.group(1).strip() if date_match else None,
        "time": time_match.group(1).strip() if time_match else None
    }
    
    print(f"Extracted appointment details: {appointment_details}")
    
    # Check if required fields are present
    required_fields = ["name", "phone", "reason", "date", "time"]
    if all(appointment_details.get(field) for field in required_fields):
        return appointment_details
    else:
        print("Missing required appointment details")
        return None
# --- End Appointment Detail Extraction Function ---


# --- Save Appointment Function ---
def save_appointment(appointment_details, organization_id):
    """Save appointment details to database."""
    try:
        # Prepare patient data for database
        appointment_data = {
            "name": appointment_details.get("name"),
            "phone": appointment_details.get("phone"),
            "reason": appointment_details.get("reason"),
            "appointment_date": appointment_details.get("date"),
            "appointment_time": appointment_details.get("time"),
            "organization_id": organization_id,
            "created_at": datetime.now(),
            "status": "confirmed"
        }
        
        # Store in MongoDB
        result = patient_data_collection.insert_one(appointment_data)
        print(f"Saved appointment in MongoDB with ID: {result.inserted_id}")
        
        return True, "Your appointment has been successfully booked."
    except Exception as e:
        print(f"Error saving appointment in database: {e}")
        return False, "There was an error saving your appointment. Please try again later."
# --- End Save Appointment Function ---


# --- LLM Prompts and Chains ---
intent_prompt = PromptTemplate.from_template("""
You are an AI assistant tasked with detecting the intent of a conversation. The conversation history is provided below.

### Conversation History:
{chat_history}

### User Query:
User Query: "{query}"

Analyze the entire conversation history and the user's most recent query to determine the intent:

1.  **Appointment Intent**: If the user is starting or continuing the process of booking, modifying, Confirming, or canceling an appointment, **including phrases like "book a demo," "schedule a meeting," "set up a consultation," or "arrange a session,"** and has not yet provided all necessary details (Name, Email, Phone Number, Reason for Appointment, Preferred Date, Preferred Time), respond with "appointment."

2.  **General Query**: If the user query is unrelated to appointments—such as general questions, greetings (e.g., "hi", "hello", "hey there"), introductory queries (e.g., "who are you?", "how can you help me?"), or other unrelated requests—respond with "general".

**Note:** If the user is only asking for **general information about a doctor** — such as availability, specialization, or consultation timings — and the query does **not express any intent to book or schedule an appointment, session, demo, meeting, or consultation**, treat it as a **general** query.

**Also Note:** If the user is asking about the **process, requirements, or details of how appointments work**, but has **not expressed an intention to actually book** one, treat it as a **general** query as well.

**Focus primarily on determining whether the user is engaged in an appointment-related conversation or continuing the appointment process.**
 Respond with only one word: "appointment" or "general".
""")

intent_chain = LLMChain(
    prompt=intent_prompt,
    llm=llm,
)


response_prompt = PromptTemplate.from_template("""
    You are an AI assistant for {organization_name}.
    {organization_description}

    **Expertise:** {organization_expertise}
    You are Sana, a specialized mental health and chronic disease management chatbot assistant, developed by EMRChain. You interact in a text-based chat environment for a healthcare office, providing responses with warmth, empathy, and professionalism.
    🌟 Start of Conversation Greeting if someone say (Hy, Hello, Hi there!), (First Message Only):
    Hello! I'm Sana, your mental health and chronic disease management assistant. How can I support you today?
    📌 Scope Boundaries:
    Only answer questions related to:
    Mental health (e.g., depression, anxiety, stress management)
    Chronic diseases (e.g., diabetes, hypertension, asthma, arthritis)
    If asked anything outside your scope, respond exactly as follows:
    English:
    I specialize in mental health and chronic disease management. Please ask about related concerns.
    Urdu:
    میں ذہنی صحت اور دائمی بیماریوں کے انتظام کے بارے میں معلومات فراہم کرتا/کرتی ہوں۔ براہ کرم متعلقہ سوال پوچھیں۔
    📘 Response Rules:
    Accuracy:
    Respond to health-related questions using clear, factual, and evidence-based information in one concise line.
    Tone:
    Keep responses empathetic, conversational, and professional. Avoid jargon and overly technical language.
    Scope Enforcement:
    Do not engage in topics outside your expertise. Use predefined out-of-scope messages.
    Conversation Ending:
    Politely wrap up with:
    Thank you! Is there anything else I can help you with today?
    ✅ Goal:
    Maintain a friendly, helpful tone while ensuring strict focus on mental health and chronic disease support. Respond like a supportive digital health assistant in a clinical setting.
    **Note:** Ensure to ask the user for any missing details before proceeding with the booking process.
    ### Context:
    {related_text}

    ### Conversation History:
    {chat_history}

    ### User Query:
    {query}

    **Response Guidelines:**
    - Ensure answers align with {organization_expertise}.
    - Provide answers in a **natural, paragraph-style format** without using bullet points, numbering, special symbols, or structured lists.
    - Ensure responses are brief yet sufficiently detailed to provide meaningful insight.
    - Always present responses as continuous text rather than a list.
    - For **For general topics**, first define the term concisely before relating it to {organization_name}'s expertise.


    **Handling Queries:**
    - If the query falls **within {organization_expertise} and Context is available**, provide a precise response using Context.
    - If **Context is not available but the query falls within {organization_expertise}**, first define the term concisely before relating it to {organization_name}'s expertise. Encourage the user to contact {organization_name} for specific details.

    - If the query includes symptoms that match known conditions within {organization_expertise}, confidently state that these symptoms are indicative of a specific condition (e.g., PCOS, fibroids, endometriosis) within the organization's domain. Clearly name the possible condition(s) based on the symptoms provided and recommend professional consultation at {organization_name} for proper evaluation and treatment.
    - If the user mentions a specific condition or disease within {organization_expertise}, briefly explain the condition and mention its common causes and contributing factors. Then, relate it to {organization_name}'s services and advise the user to consult the hospital for a detailed diagnosis and personalized care.

    - If the query is **outside {organization_expertise}**, politely inform the user that the topic is beyond {organization_name}'s expertise and recommend consulting a relevant expert or trusted source.
""")

response_chain = LLMChain(
    prompt=response_prompt,
    llm=llm,
)


appointment_prompt = PromptTemplate.from_template("""
The user is either starting or continuing the process of booking an appointment. Based on the conversation history and the user's latest query, assist them in proceeding with the booking.

For booking an appointment, you need to collect the following information:
1. **Name**
2. **Phone Number**
3. **Reason for Appointment**
4. **Preferred Date** (Must be {formatted_min_date} or later)
5. **Preferred Time** (Must be at least 1 hour from now: {formatted_min_time})

**Response Guidelines:**
- **USE BULLET POINTS to ask for information** - When asking for multiple pieces of information, always use bullet points.
- **Be clear and organized** - Make it easy for users to understand what information you need.
- If the user hasn't provided any information yet, use this exact format for your first response:

"I'll help you book an appointment. Please provide the following details:
- Your full name
- Phone number
- Reason for appointment
- Preferred date ({formatted_min_date} or later)
- Preferred time (must be after {formatted_min_time} today)"

If some of this information has already been provided, request the missing details or confirm the information provided so far.

If the user selects a date **before {formatted_min_date}**, say:
*"The appointment date must be today or later."*

If the user selects a time **before {formatted_min_time} on today's date**, inform them:
*"Appointments must be scheduled at least 1 hour from now."*

### Final Confirmation:
Once all required details are collected, confirm with the user:
"I have the following details for your appointment:
- Name: {{name}}
- Phone Number: {{phone_number}}
- Reason: {{reason}}
- Appointment Date: {{appointment_date}}
- Appointment Time: {{appointment_time}}

Is this correct? Please confirm by saying 'yes' or 'confirm' to book your appointment."

Once the user confirms the details, inform them that their appointment has been booked successfully:
"Thank you! Your appointment has been successfully booked. We look forward to seeing you on {{appointment_date}} at {{appointment_time}}."

### Conversation History:
{chat_history}

### User Query:
User Query: {query}
""")

appointment_chain = LLMChain(
    prompt=appointment_prompt,
    llm=llm,
)
# --- End LLM Prompts and Chains ---


# --- Appointment Processing Function ---
def process_appointment_confirmation(x):
    """Process appointment confirmation and save to database."""
    query = x["query"].strip().lower()
    chat_history_list = x["chat_history_list"]
    
    # Check if input looks like a confirmation (yes, confirm, etc.)
    if query in ["yes", "confirm", "correct", "that's correct", "that is correct", "book it", "book the appointment"]:
        # Extract appointment details from chat history
        appointment_details = extract_appointment_details(chat_history_list)
        if not appointment_details:
            return {
                "text": "I couldn't find your complete appointment details. Please provide your booking information again."
            }
        
        # Save appointment to database
        success, message = save_appointment(
            appointment_details, 
            x.get("unique_id", SMS_ORG_UNIQUE_ID)
        )
        
        if success:
            confirmation_message = f"Thank you! Your appointment has been successfully booked. We look forward to seeing you on {appointment_details.get('date')} at {appointment_details.get('time')}."
            return {"text": confirmation_message}
        else:
            return {"text": message}
    else:
        # If not a confirmation, let the appointment chain handle it
        return appointment_chain.invoke(x)
# --- End Appointment Processing Function ---


# --- Router Chain Structure ---
router_chain = RunnableSequence(
    RunnableLambda(lambda x: {
        "query": x.get("query", ""),
        "unique_id": x.get("unique_id", SMS_ORG_UNIQUE_ID),
        "chat_history_list": x.get("chat_history_list", [])
    }),
    fetch_org_runnable,

    # Step 2: Format chat history for prompts
    RunnableLambda(lambda x: {
        **x,
        "chat_history": format_chat_history(x["chat_history_list"])
    }),

    # Step 3: Fetch related text
    fetch_related_text_runnable,

    # Step 4: Detect intent
    RunnableLambda(lambda x: {
         **x,
        "intent": intent_chain.invoke({
            "query": x["query"],
            "chat_history": x["chat_history"]
        })['text'].strip().lower()
    }),

    # Step 5: Branch based on intent
    RunnableBranch(
        (lambda x: x["intent"] == "appointment", 
         RunnableLambda(process_appointment_confirmation)),
        (lambda x: x["intent"] == "general", response_chain),
        response_chain
    )
)
# --- End Router Chain ---


# --- Web Endpoint (Using Streaming Response) ---
class QueryInput(BaseModel):
    query: str
    unique_id: str
    history: list[dict]

async def response_generator(inputs):
    """Generator to stream response back to web client."""
    try:
        chat_history_list = []
        for msg_dict in inputs.get("history", []):
             if msg_dict.get("type") == "human":
                 chat_history_list.append(HumanMessage(content=msg_dict.get("content", "")))
             elif msg_dict.get("type") == "ai":
                 chat_history_list.append(AIMessage(content=msg_dict.get("content", "")))

        chain_inputs = {
            "query": inputs["query"],
            "unique_id": inputs["unique_id"],
            "chat_history_list": chat_history_list
        }

        response = router_chain.invoke(chain_inputs)

        if isinstance(response, dict):
            response_text = response.get('text', '')
            if not response_text and 'answer' in response:
                 response_text = response['answer']
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = "Sorry, an unexpected response format was received."
            print(f"Warning: Web endpoint received unexpected chain response type: {type(response)}")

        for char in response_text:
            yield char
            await asyncio.sleep(0.00000001)

    except Exception as e:
        yield f"Error: {str(e)}"
        print(f"Error in /generate_answer endpoint: {e}")


@app.post("/generate_answer")
async def process_query(input_data: QueryInput):
    """Web endpoint to process chat queries with streaming."""
    try:
        inputs = {
            "query": input_data.query,
            "unique_id": input_data.unique_id,
            "history": input_data.history
        }
        return StreamingResponse(response_generator(inputs), media_type="text/plain")
    except Exception as e:
        print(f"Error processing web query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# --- End Web Endpoint ---


# --- SMS Endpoint for Twilio ---
@app.post("/sms")
async def handle_sms(From: str = Form(...), Body: str = Form(...)):
    """Receives incoming SMS from Twilio, processes it, and sends a reply."""
    session_id = From
    user_query = Body

    print(f"Received SMS from {session_id}: {user_query}")

    session_history_list = sms_sessions.get(session_id, [])
    if session_id not in sms_sessions:
        sms_sessions[session_id] = session_history_list
        print(f"Created new SMS session for {session_id}")
    else:
         print(f"Loaded SMS session for {session_id} with {len(session_history_list)} previous messages.")

    # Add the current human message to the history list
    session_history_list.append(HumanMessage(content=user_query))

    inputs = {
        "query": user_query,
        "unique_id": SMS_ORG_UNIQUE_ID,
        "chat_history_list": session_history_list,
    }

    final_response_text = "Sorry, I couldn't process your request at the moment." # Default error response

    try:
        print(f"Invoking router chain for SMS (session: {session_id})...")
        chain_response = router_chain.invoke(inputs)
        print(f"Router chain returned (session: {session_id}): {chain_response}")

        if isinstance(chain_response, dict):
            response_text = chain_response.get('text', '')
            if not response_text and 'answer' in chain_response:
                 response_text = chain_response['answer']
        elif isinstance(chain_response, str):
            response_text = chain_response
        else:
             response_text = "An unexpected response format was received from the processing chain."
             print(f"Warning: SMS endpoint received unexpected chain response type: {type(chain_response)}")

        final_response_text = response_text
        # Add the AI's response to the history list for the next turn
        session_history_list.append(AIMessage(content=final_response_text))
        print(f"SMS session {session_id}: Appended standard AI response to history.")

        # Generate TwiML response
        resp = MessagingResponse()
        resp.message(final_response_text)

        xml_response_content = str(resp)
        print(f"SMS session {session_id}: Generated TwiML: {xml_response_content}")

        # ✅ SMS Change: Return PlainTextResponse for success (Status 200 OK)
        return PlainTextResponse(content=xml_response_content, media_type="application/xml")

    except Exception as e:
        print(f"SMS session {session_id}: An error occurred during SMS processing: {e}")
        error_resp = MessagingResponse()
        error_resp.message("Sorry, I encountered an error. Please try again later.")
        error_xml_content = str(error_resp)
        print(f"SMS session {session_id}: Generated Error TwiML: {error_xml_content}")
        # ✅ SMS Change: Return explicit Response with status_code 500 for errors
        return Response(content=error_xml_content, media_type="application/xml", status_code=500)


# To run the app:
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)