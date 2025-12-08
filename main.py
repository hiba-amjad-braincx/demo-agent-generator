from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import openai
from retell import Retell
import json

# Load .env from current directory (root folder)
load_dotenv()

app = FastAPI(title="Demo Agent Generation App")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
RETELL_API_KEY = os.getenv("RETELL_API_KEY")
if Retell:
    retell_client = Retell(api_key=RETELL_API_KEY)
else:
    retell_client = None
    import requests

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt library
PROMPT_LIBRARY = {
    "customer_service": {
        "system_prompt": "You are a helpful and professional customer service representative. Your goal is to assist customers with their inquiries, resolve issues, and provide excellent service."
    },
    "sales": {
        "system_prompt": "You are a knowledgeable and friendly sales representative. Your goal is to understand customer needs, present relevant solutions, and guide them through the purchase process."
    },
    "appointment_scheduling": {
        "system_prompt": "You are a professional appointment scheduler. Your goal is to help customers book appointments efficiently, confirm details, and provide reminders."
    },
    "technical_support": {
        "system_prompt": "You are a technical support specialist. Your goal is to diagnose technical issues, provide step-by-step solutions, and ensure customer problems are resolved."
    },
    "general": {
        "system_prompt": "You are a helpful AI assistant. Your goal is to provide accurate information and assist users with their requests."
    }
}


class AgentInfo(BaseModel):
    name: str
    persona: str
    purpose: str
    use_case: str
    company_name: str
    prompt_template: Optional[str] = None
    voice_id: Optional[str] = "11labs-Cimo"
    language: Optional[str] = "en-US"
    llm_id: Optional[str] = None  # Use existing LLM if provided


class WebhookData(BaseModel):
    agent_name: Optional[str] = None
    persona: Optional[str] = None
    purpose: Optional[str] = None
    use_case: Optional[str] = None
    company_name: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow extra fields


class PostCallAnalysisField(BaseModel):
    type: str
    name: str
    description: str
    examples: Optional[List[str]] = None


@app.get("/")
async def root():
    return {"message": "Demo Agent Generation API", "status": "running"}


@app.get("/prompts")
async def get_prompt_library():
    """Get available prompt templates"""
    return {"prompts": PROMPT_LIBRARY}


@app.get("/list-retell-llms")
async def list_retell_llms():
    """List all available Retell LLMs"""
    try:
        import requests
        base_url = "https://api.retellai.com"
        headers = {
            "Authorization": f"Bearer {RETELL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(f"{base_url}/list-retell-llms", headers=headers)
        response.raise_for_status()
        return {"llms": response.json()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Retell LLMs: {str(e)}")


@app.post("/generate-prompt")
async def generate_prompt(agent_info: AgentInfo):
    """Generate a custom prompt using OpenAI based on agent info"""
    try:
        # Determine which template to use based on use_case
        template_key = agent_info.use_case.lower().replace(" ", "_")
        if template_key not in PROMPT_LIBRARY:
            template_key = "general"
        
        base_prompt = PROMPT_LIBRARY[template_key]["system_prompt"]
        
        # Enhance prompt with OpenAI
        enhancement_prompt = f"""
        Based on the following agent information, create a detailed and personalized system prompt in the format "You are [description]":
        
        Agent Name: {agent_info.name}
        Persona: {agent_info.persona}
        Purpose: {agent_info.purpose}
        Use Case: {agent_info.use_case}
        Company: {agent_info.company_name}
        
        Base Template: {base_prompt}
        
        Create a comprehensive system prompt that:
        1. Starts with "You are" (not "I am" or "Hello, I am")
        2. Describes the agent in third person
        3. Incorporates the agent's personality, purpose, and company context
        4. Is specific, engaging, and aligned with the use case
        
        Return only the prompt text, no additional formatting. The prompt should be a system instruction, not an introduction.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a prompt engineering expert. Create system prompts for AI voice agents. The prompt MUST start with 'You are' and describe the agent in third person. Do NOT create first-person introductions or greetings. The prompt should be a system instruction, not a conversation starter."},
                {"role": "user", "content": enhancement_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        generated_prompt = response.choices[0].message.content.strip()
        
        return {
            "generated_prompt": generated_prompt,
            "base_template": template_key,
            "agent_info": agent_info.model_dump()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")


@app.post("/create-agent")
async def create_agent(agent_info: AgentInfo):
    """Create a Retell agent with the provided information"""
    try:
        import requests
        
        # Use existing LLM ID if provided, otherwise create new LLM
        if agent_info.llm_id:
            llm_id = agent_info.llm_id
        else:
            # Generate or use provided prompt
            if agent_info.prompt_template:
                system_prompt = agent_info.prompt_template
            else:
                # Generate prompt
                prompt_response = await generate_prompt(agent_info)
                system_prompt = prompt_response["generated_prompt"]
            
            # Create new Retell LLM using direct API call
            base_url = "https://api.retellai.com"
            headers = {
                "Authorization": f"Bearer {RETELL_API_KEY}",
                "Content-Type": "application/json"
            }
            llm_payload = {
                "general_prompt": system_prompt,
                "general_tools_enabled": True,
                "begin_message": f"Hello! I'm {agent_info.name} from {agent_info.company_name}. How can I help you today?",
                "model": "gpt-4o-mini",
                "temperature": 0
            }
            
            print(f"[LOG] Creating Retell LLM for agent: {agent_info.name}")
            llm_response = requests.post(f"{base_url}/create-retell-llm", json=llm_payload, headers=headers)
            
            if not llm_response.ok:
                error_detail = llm_response.text
                try:
                    error_json = llm_response.json()
                    error_detail = error_json.get("message") or error_json.get("error") or str(error_json)
                except:
                    pass
                raise HTTPException(
                    status_code=llm_response.status_code,
                    detail=f"Error creating Retell LLM: {error_detail}"
                )
            
            llm_data = llm_response.json()
            llm_id = llm_data.get("retell_llm_id") or llm_data.get("llm_id")
            
            if llm_response.ok and llm_id:
                print(f"[LOG] LLM created successfully: {llm_id}")
            else:
                print(f"[LOG] LLM creation failed: Status {llm_response.status_code}")
            
            if not llm_id:
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM created but no llm_id returned. Response: {llm_data}"
                )
        
        # Validate llm_id
        if not llm_id or not isinstance(llm_id, str):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid llm_id: {llm_id}"
            )
        
        # Create Retell Agent using direct API call
        base_url = "https://api.retellai.com"
        headers = {
            "Authorization": f"Bearer {RETELL_API_KEY}",
            "Content-Type": "application/json"
        }
        agent_payload = {
            "response_engine": {
                "type": "retell-llm",
                "llm_id": llm_id
            },
            "voice_id": agent_info.voice_id,
            "agent_name": agent_info.name,
            "language": agent_info.language
        }
        
        print(f"[LOG] Creating Retell agent: {agent_info.name}")
        print(f"[LOG] Using LLM ID: {llm_id}")
        print(f"[LOG] Agent payload: name={agent_info.name}, voice={agent_info.voice_id}, language={agent_info.language}")
        
        agent_response = requests.post(f"{base_url}/create-agent", json=agent_payload, headers=headers)
        
        # Better error handling to see what Retell API returns
        if not agent_response.ok:
            error_detail = agent_response.text
            try:
                error_json = agent_response.json()
                error_detail = error_json.get("message") or error_json.get("error") or error_json.get("detail") or str(error_json)
            except:
                pass
            print(f"[LOG] Agent creation failed: Status {agent_response.status_code}, Error: {error_detail}")
            raise HTTPException(
                status_code=agent_response.status_code,
                detail=f"Retell API error: {error_detail}"
            )
        
        agent_data = agent_response.json()
        agent_id = agent_data.get("agent_id")
        agent_name = agent_data.get("agent_name")
        voice_id = agent_data.get("voice_id")
        
        print(f"[LOG] Agent created successfully: {agent_id} ({agent_name})")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "llm_id": llm_id,
            "voice_id": voice_id,
            "message": f"Agent '{agent_info.name}' created successfully!"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")


@app.post("/webhook")
async def receive_webhook(request: Request):
    """Receive call_analyzed webhook from Retell via n8n and create new agent"""
    try:
        data = await request.json()
        
        # Debug: log the incoming data structure
        print(f"[DEBUG] Received webhook data keys: {list(data.keys())}")
        
        # n8n wraps the webhook data, so check both top-level and body
        body_data = data.get("body") or data
        
        # Check if this is a call_analyzed event
        event_type = (
            body_data.get("event") or 
            body_data.get("event_type") or
            data.get("event") or 
            data.get("event_type")
        )
        
        print(f"[DEBUG] Event type: {event_type}")
        print(f"[DEBUG] Data keys: {list(data.keys())}")
        if "body" in data:
            print(f"[DEBUG] Body keys: {list(body_data.keys())}")
        
        # If it's not call_analyzed, ignore it (or handle other events if needed)
        if event_type != "call_analyzed":
            return JSONResponse(
                status_code=200,
                content={"message": f"Event {event_type} received but not processed"}
            )
        
        # Extract post-call analysis data from Retell webhook
        # extracted_data can be at top level or in analysis
        analysis = body_data.get("analysis") or {}
        extracted_data = (
            body_data.get("extracted_data") or  # Top level (most common)
            analysis.get("extracted_data") or  # Nested in analysis
            body_data.get("post_call_analysis") or 
            {}
        )
        
        print(f"[DEBUG] Analysis keys: {list(analysis.keys()) if analysis else 'No analysis'}")
        print(f"[DEBUG] Extracted data keys: {list(extracted_data.keys()) if extracted_data else 'No extracted_data'}")
        
        # If no extracted_data found, log the structure for debugging
        if not extracted_data:
            print(f"[DEBUG] WARNING: No extracted_data found!")
            print(f"[DEBUG] body_data keys: {list(body_data.keys())}")
            if analysis:
                print(f"[DEBUG] analysis keys: {list(analysis.keys())}")
        
        # Map the extracted fields to our AgentInfo structure
        # Fields from Demo Genie agent: agent_name, agent_persona, agent_usecase, company_name, other_instructions
        agent_name = extracted_data.get("agent_name") or ""
        agent_persona = extracted_data.get("agent_persona") or ""
        agent_usecase = extracted_data.get("agent_usecase") or ""
        company_name = extracted_data.get("company_name") or ""
        other_instructions = extracted_data.get("other_instructions") or ""
        
        # Map use_case to our template keys
        use_case_key = map_use_case_to_key(agent_usecase)
        
        agent_data = {
            "name": agent_name,
            "persona": agent_persona,
            "purpose": agent_usecase,  # Use the full use case text as purpose
            "use_case": use_case_key,  # Use mapped key for template selection
            "company_name": company_name
        }
        
        # Handle other_instructions - can be added to persona
        if other_instructions:
            if agent_data["persona"]:
                agent_data["persona"] += f". {other_instructions}"
            else:
                agent_data["persona"] = other_instructions
        
        # Validate required fields
        if not agent_data["name"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required field: agent_name not found in extracted data"}
            )
        
        if not agent_data["purpose"] and not agent_data["use_case"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required field: agent_usecase or purpose not found in extracted data"}
            )
        
        # If purpose is missing but use_case exists, use use_case as purpose
        if not agent_data["purpose"]:
            agent_data["purpose"] = agent_data["use_case"]
        
        # Set defaults for missing optional fields
        if not agent_data["persona"]:
            agent_data["persona"] = "Friendly and professional"
        if not agent_data["company_name"]:
            agent_data["company_name"] = "Demo Company"
        
        # Create agent info object
        agent_info = AgentInfo(**agent_data)
        
        # Create the agent
        result = await create_agent(agent_info)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Agent created successfully from call_analyzed webhook",
                "extracted_data": extracted_data,
                "agent_data": result
            }
        )
    
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error processing webhook: {str(e)}",
                "traceback": traceback.format_exc()
            }
        )


def map_use_case_to_key(use_case_text: str) -> str:
    """Map use case text to our template keys"""
    if not use_case_text:
        return "general"
    
    use_case_lower = use_case_text.lower()
    
    if "customer service" in use_case_lower or "customer support" in use_case_lower:
        return "customer_service"
    elif "sales" in use_case_lower or "selling" in use_case_lower:
        return "sales"
    elif "appointment" in use_case_lower or "scheduling" in use_case_lower or "booking" in use_case_lower:
        return "appointment_scheduling"
    elif "technical support" in use_case_lower or "tech support" in use_case_lower or "support" in use_case_lower:
        return "technical_support"
    else:
        return "general"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

