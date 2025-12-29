# Demo Agent Generation App

A web application for creating AI voice agents using Retell AI. This app allows you to configure agent details (name, persona, purpose, use case, company) and automatically creates agents via the Retell API.

## Features

- **Webhook-Based Agent Creation**: Automatically creates Retell agents when `call_analyzed` events are received
- **Prompt Library**: Pre-built prompt templates for different use cases
- **OpenAI Integration**: Generate custom prompts using OpenAI GPT models
- **Retell API Integration**: Automatically create voice agents with Retell AI
- **n8n Integration**: Receives agent information from n8n workflows

## Setup

### Prerequisites

- Python 3.8+
- Retell API Key
- OpenAI API Key

### Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
RETELL_API_KEY=your_retell_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
AGENT_API_KEY=sk_live_<your-api-key>  # Bearer token for agent creation API
AGENT_API_BASE_URL=https://api.braincxai.com  # Optional, defaults to this
N8N_WEBHOOK_URL=https://your-n8n-instance.com/webhook/your-webhook-id  # n8n webhook URL for sending user contact info + agent URL
```

### Running the Application

1. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. Start the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The server will be available at `http://localhost:8001`

## API Endpoints

### `GET /`
Health check endpoint

### `GET /prompts`
Get available prompt templates

### `POST /webhook`
Receive `call_analyzed` webhook events from Retell (via n8n) and create new agents

**Expected Webhook Structure (from Retell call_analyzed event):**
```json
{
  "event": "call_analyzed",
  "extracted_data": {
    "agent_name": "Sarah",
    "agent_persona": "Friendly and professional",
    "agent_purpose": "Customer support",
    "company_name": "Acme Corp",
    "voice_gender": "feminine",
    "user_first_name": "Joe",
    "user_last_name": "Morgan",
    "user_email": "joe.morgan@gmail.com",
    "user_phone": "+1 555-123-4567"
  }
}
```

The endpoint extracts agent information and user contact info from the `extracted_data` field, creates a new agent, and sends user contact info + agent URL to n8n for GHL contact creation.

## Use Cases

The app supports the following use cases with pre-configured prompts:

- **Customer Service**: Helpful customer support agents
- **Sales**: Sales representatives for product promotion
- **Appointment Scheduling**: Agents for booking appointments
- **Technical Support**: Technical issue resolution agents
- **General**: General-purpose AI assistants

## Architecture

- **Backend**: FastAPI (Python) - Handles webhook requests, Retell integration, OpenAI integration
- **Webhook Endpoint**: Receives `call_analyzed` events from Retell (via n8n) and creates agents automatically

## Notes

- The webhook endpoint listens for `call_analyzed` events from Retell (forwarded via n8n)
- When the Demo Genie agent collects information and the call ends, Retell sends a `call_analyzed` webhook
- n8n forwards this webhook to `http://localhost:8001/webhook` (via ngrok)
- The app extracts agent info from `extracted_data` and creates a new Retell agent
- The app uses GPT-4o for prompt generation and GPT-4o-mini for agent responses
- Default voice is set to "11labs-Cimo" but can be customized



