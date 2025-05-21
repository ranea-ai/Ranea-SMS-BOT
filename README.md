# Ranea: Healthcare AI Assistant
A specialized mental health and chronic disease management chatbot assistant, focused on providing warm, empathetic, and professional healthcare guidance through both web and SMS interfaces.
## 🌟 Features

- **Specialized Healthcare Focus**: Mental health and chronic disease management expertise
- **Multi-Channel Support**: Seamless experiences via web interface and SMS
- **Intelligent Conversation**: Context-aware responses with memory of previous interactions
- **Appointment Booking**: Complete workflow from request to confirmation
- **RAG-Powered Knowledge Base**: Organization-specific information retrieval
- **Multi-Organization Support**: Customizable for different healthcare providers

## 📚 Tech Stack

- **Backend Framework**: FastAPI
- **Language Model**: Google Gemini 2.5 Flash
- **Vector Database**: Qdrant for semantic search
- **Document Database**: MongoDB for organization data and appointments
- **SMS Integration**: Twilio
- **Embedding Model**: Google Text Embedding-004
- **LangChain**: For orchestrating LLM workflows

## 🔄 System Workflow

### Architecture Overview

The system uses a sophisticated chain of components to process user queries:

1. **Query Reception**: Via web or SMS endpoint
2. **Context Enhancement**: Organization info and conversation history added
3. **Knowledge Retrieval**: RAG system fetches relevant information
4. **Intent Classification**: Determines if query is appointment-related or general
5. **Response Generation**: Tailored response based on intent and context
6. **Delivery**: Streaming response (web) or formatted SMS reply

![System Workflow](https://via.placeholder.com/800x400?text=System+Workflow+Diagram)

### Detailed Process Flow

#### Initial Setup
```
Load environment variables → Initialize databases → Configure LLM and embeddings
```

#### Conversation Processing
```
Receive query → Load conversation history → Fetch organization info → 
Retrieve relevant knowledge → Detect intent → Route to appropriate handler → 
Generate response → Deliver to user
```

#### Appointment Booking Flow
```
Detect appointment intent → Collect required information → Validate date/time → 
Confirm details → Save to database → Send confirmation
```

## 👤 Client Experience

### Web Interface

Users experience:
- Real-time streaming responses
- Contextual, organization-specific conversations
- Guided appointment booking with validation
- Professional, empathetic healthcare guidance

### SMS Interface

Users can:
- Text natural language queries from any mobile device
- Book appointments through a guided text conversation
- Receive healthcare information with the same AI capabilities
- Maintain conversation context across multiple messages

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MongoDB instance
- Qdrant instance
- Google AI API key
- Twilio account (for SMS functionality)

### Environment Variables

Create a `.env` file with the following variables:

```
GOOGLE_API_KEY=your_google_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
MONGO_URI=your_mongodb_connection_string
SMS_ORG_UNIQUE_ID=default_organization_id
```

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/healthcare-chatbot.git
   cd healthcare-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the server:
   ```
   python main.py
   ```

The server will start at `http://localhost:8000`

## 📝 API Endpoints

### Web Chat API

```
POST /generate_answer
```

Request body:
```json
{
  "query": "User's message here",
  "unique_id": "organization_identifier",
  "history": [
    {"type": "human", "content": "Previous user message"},
    {"type": "ai", "content": "Previous AI response"}
  ]
}
```

Returns a streaming response with the AI's reply.

### SMS Endpoint

```
POST /sms
```

Accepts Twilio webhook parameters:
- `From`: Sender's phone number
- `Body`: Message content

Returns TwiML for Twilio to process.

## 💾 Database Structure

### MongoDB Collections

- **organizations**: Healthcare provider information
- **user_queries**: Stores conversation history
- **Appointments_Booked**: Appointment details

### Qdrant Collection

- **Chabot_Queries**: Vector embeddings for organization knowledge

## 📊 Project Structure

```
├── main.py                # Main application file
├── .env                   # Environment variables
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## 🔧 Configuration Options

### Organization Configuration

Each healthcare organization is defined by:
- Unique identifier
- Organization name and description
- Address and contact information
- Area of expertise
- Knowledge base for RAG system

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Anthropic for LLM research
- Google for Gemini model and embeddings
- LangChain for providing the tools to build LLM applications
- Qdrant for vector search capabilities
- MongoDB for document storage
- Twilio for SMS integration
