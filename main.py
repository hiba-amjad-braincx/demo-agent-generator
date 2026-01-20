from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from typing import Optional
import os
import json

# Import models
from models import AgentInfo

# Import workflow functions (includes transcript extraction)
try:
    from workflow import (
        run_workflow,
        WorkflowInput,
        extract_structured_data_from_transcript,
        ExtractedStructuredData
    )
    WORKFLOW_AVAILABLE = True
    TRANSCRIPT_EXTRACTOR_AVAILABLE = True
    print("[LOG] Workflow functions imported successfully")
except ImportError as e:
    print(f"[ERROR] Workflow functions not available: {e}")
    WORKFLOW_AVAILABLE = False
    TRANSCRIPT_EXTRACTOR_AVAILABLE = False
    run_workflow = None
    WorkflowInput = None
    extract_structured_data_from_transcript = None
    ExtractedStructuredData = None

# Load .env from current directory (root folder)
load_dotenv()
RETELL_API_KEY = os.getenv("RETELL_API_KEY")
AGENT_API_KEY = os.getenv("AGENT_API_KEY")  # Bearer token for agent creation API (format: sk_live_<your-api-key>)
AGENT_API_BASE_URL = os.getenv("AGENT_API_BASE_URL", "https://api.braincxai.com")  # Configurable base URL for API
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")  # n8n webhook URL for sending user contact info + agent URL

app = FastAPI(title="Demo Agent Generation App")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Demo Agent Generation API", "status": "running"}


def get_bearer_token() -> str:
    """Get Bearer token from environment variable.
    
    Returns:
        str: Bearer token for authentication (format: sk_live_<your-api-key>)
    """
    if not AGENT_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "AGENT_API_KEY environment variable is required. "
                "Set AGENT_API_KEY=sk_live_<your-api-key> in your .env file."
            )
        )
    return AGENT_API_KEY


async def send_to_n8n(user_first_name: Optional[str], user_last_name: Optional[str], user_email: Optional[str], user_phone: Optional[str], agent_url: Optional[str]):
    """Send user contact info and agent URL to n8n webhook
    
    Args:
        user_first_name: First name of the user
        user_last_name: Last name of the user
        user_email: Email address of the user
        user_phone: Phone number of the user
        agent_url: URL of the created agent
    """
    if not N8N_WEBHOOK_URL:
        print(f"[WARNING] N8N_WEBHOOK_URL not configured, skipping n8n webhook call")
        return
    
    if not agent_url:
        print(f"[WARNING] No agent_url provided, skipping n8n webhook call")
        return
    
    try:
        import requests
        
        payload = {
            "user_first_name": user_first_name,
            "user_last_name": user_last_name,
            "user_email": user_email,
            "user_phone": user_phone,
            "agent_url": agent_url
        }
        
        print(f"[LOG] Sending data to n8n webhook: {N8N_WEBHOOK_URL}")
        print(f"[LOG] Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.ok:
            print(f"[LOG] Successfully sent data to n8n webhook")
        else:
            print(f"[WARNING] n8n webhook returned status {response.status_code}: {response.text}")
            
    except Exception as e:
        # Don't fail agent creation if n8n webhook fails
        print(f"[ERROR] Error sending data to n8n webhook: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")


@app.post("/generate-prompt")
async def generate_prompt(agent_info: AgentInfo, structured_data: Optional[ExtractedStructuredData] = None):
    """Generate a custom prompt using OpenAI Agent Builder SDK workflow (required)"""
    # Check if workflow is available
    if not WORKFLOW_AVAILABLE or not run_workflow:
        error_msg = "Workflow functions are required but not available. Please ensure workflow.py is accessible."
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] WORKFLOW_AVAILABLE: {WORKFLOW_AVAILABLE}, run_workflow: {run_workflow}")
        raise HTTPException(
            status_code=503,
            detail=error_msg
        )
    
    try:
        print(f"[LOG] Using Agent Builder SDK workflow for prompt generation")
        print(f"[LOG] Workflow available: {WORKFLOW_AVAILABLE}")
        print(f"[LOG] Agent details - Name: {agent_info.name}, Company: {agent_info.company_name}, Purpose: {agent_info.purpose}")
        
        # Construct input_as_text from individual fields
        # website_info = f"\nWebsite URL: {agent_info.website_url}" if agent_info.website_url else ""
        
        # Build structured data section if available
        structured_lines = []
        if structured_data and structured_data.category:
            cat_name = {
                "A": "Task Handling",
                "B": "Lead / Information Collection",
                "C": "Information & Support",
                "Z": "Other / Custom"
            }.get(structured_data.category, "Unknown")
            structured_lines.append(f"AGENT CATEGORY: {structured_data.category} — {cat_name}")

            if structured_data.category == "A" and structured_data.task_type:
                structured_lines.append(f"Task Type: {structured_data.task_type}")
                if structured_data.required_fields:
                    structured_lines.append(f"Required Fields: {', '.join(structured_data.required_fields)}")
                if structured_data.availability_handling:
                    structured_lines.append(f"Availability Handling: {structured_data.availability_handling}")
                if structured_data.human_escalation is not None:
                    structured_lines.append(f"Offer Human Transfer: {'Yes' if structured_data.human_escalation else 'No'}")

            elif structured_data.category == "B":
                if structured_data.collection_goal:
                    structured_lines.append(f"Collection Goal: {structured_data.collection_goal}")
                if structured_data.required_fields:
                    structured_lines.append(f"Required Fields: {', '.join(structured_data.required_fields)}")

            elif structured_data.category == "C":
                if structured_data.information_type:
                    structured_lines.append(f"Information Type: {structured_data.information_type}")
                if structured_data.user_information:
                    structured_lines.append(f"Collect User Info: {structured_data.user_information}")

            elif structured_data.category == "Z" and structured_data.interaction_type:
                structured_lines.append(f"Interaction Style: {structured_data.interaction_type}")

        structured_data_section = "\n".join(structured_lines) if structured_lines else ""
        if structured_data_section:
            structured_data_section = "\n\n=== STRUCTURED AGENT CONFIGURATION ===\n" + structured_data_section
            print(f"[LOG] Including structured data - Category: {structured_data.category}")
        
        # Format other_instructions as a list if provided
        other_instructions_text = ""
        if agent_info.other_instructions:
            if isinstance(agent_info.other_instructions, list):
                # Format as bulleted list
                other_instructions_text = "\nother_instructions:\n" + "\n".join([f"- {rule}" for rule in agent_info.other_instructions])
            else:
                # Fallback for string (backward compatibility)
                other_instructions_text = f"\nother_instructions: {agent_info.other_instructions}"
        
        input_as_text = f"""Generate a production-quality system prompt for a Voice AI agent with the following details:

agent_name: {agent_info.name}
persona: {agent_info.persona}
purpose: {agent_info.purpose}
company_name: {agent_info.company_name}{other_instructions_text}{structured_data_section}

Generate a system prompt that begins with "You are..." and follows all best practices from the knowledge base."""
        
        workflow_input = WorkflowInput(input_as_text=input_as_text)
        
        print(f"[LOG] Starting Agent Builder workflow execution...")
        print(f"[LOG] Workflow input prepared with all required variables")
        
        # Use workflow function to generate prompt
        import time
        workflow_start_time = time.time()
        try:
            workflow_result = await run_workflow(workflow_input)
            workflow_elapsed = time.time() - workflow_start_time
            print(f"[LOG] Total workflow execution time: {workflow_elapsed:.2f} seconds ({workflow_elapsed/60:.2f} minutes)")
            if not workflow_result or "output_text" not in workflow_result:
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow returned unexpected result format: {workflow_result}"
                )
            generated_prompt = workflow_result["output_text"].strip()
        except HTTPException:
            raise
        except Exception as workflow_error:
            print(f"[ERROR] Workflow execution error: {str(workflow_error)}")
            import traceback
            print(f"[ERROR] Workflow traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error executing workflow: {str(workflow_error)}"
            )
        print(f"[LOG] Prompt generated successfully via Agent Builder workflow (length: {len(generated_prompt)} characters)")
        
        return {
            "generated_prompt": generated_prompt,
            "agent_info": agent_info.model_dump()
        }
    
    except HTTPException:
        # Re-raise HTTPException as-is (FastAPI will handle it properly)
        raise
    except Exception as e:
        print(f"[ERROR] Error generating prompt: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating prompt: {str(e)}")


@app.post("/create-agent")
async def create_agent(
    agent_info: AgentInfo, 
    structured_data: Optional[ExtractedStructuredData] = None,
    user_first_name: Optional[str] = None,
    user_last_name: Optional[str] = None,
    user_email: Optional[str] = None,
    user_phone: Optional[str] = None
):
    """Create an agent with the provided information using the new agent creation API
    
    Args:
        agent_info: Agent configuration
        structured_data: Optional structured data from transcript extraction (not used when called via HTTP endpoint)
        user_first_name: Optional first name of the user who will receive the demo agent
        user_last_name: Optional last name of the user who will receive the demo agent
        user_email: Optional email address of the user who will receive the demo agent
        user_phone: Optional phone number of the user who will receive the demo agent
    """
    try:
        import requests
        
        # Generate or use provided prompt
        if agent_info.prompt_template:
            system_prompt = agent_info.prompt_template
        else:
            # Generate prompt with structured data
            prompt_response = await generate_prompt(agent_info, structured_data)
            system_prompt = prompt_response["generated_prompt"]
        
        # Get Bearer token for authentication
        bearer_token = get_bearer_token()
        
        # Build agent creation payload for new API
        agent_payload = {
            "name": agent_info.name,
            "system_prompt": system_prompt,
            "llm_model": "gpt-4o-mini",
            "temperature": 0,
            "locale": agent_info.language or "en-US",
            "voice_provider": "elevenlabs",
            "elevenlabs_voice_id": agent_info.voice_id or "esunmKO3sPWJNo7W1w0t",  # Default to female voice
            "elevenlabs_model": "eleven_monolingual_v1",
            "is_active": True,
            "pii_policy": "off",
            "stt_provider": "deepgram",
            "stt_model": "nova3_mono_en",
            "stt_language": agent_info.language or "en-US"
        }
        
        # Create agent using new API with Bearer token authentication
        agent_api_url = f"{AGENT_API_BASE_URL}/agents"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
        
        print(f"[LOG] Creating agent via new API: {agent_info.name}")
        print(f"[LOG] API URL: {agent_api_url}")
        print(f"[LOG] Agent payload: name={agent_info.name}, voice={agent_payload['elevenlabs_voice_id']}")
        
        agent_response = requests.post(agent_api_url, json=agent_payload, headers=headers)
        
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
                detail=f"Agent API error: {error_detail}"
            )
        
        agent_data = agent_response.json()
        agent_id = agent_data.get("id")
        agent_name = agent_data.get("name")
        agent_url = agent_data.get("agent_url")
        embed_code = agent_data.get("embed_code")
        
        print(f"[LOG] Agent created successfully: {agent_id} ({agent_name})")
        if agent_url:
            print(f"[LOG] Agent URL: {agent_url}")
        
        # Send user contact info + agent URL to n8n webhook (non-blocking)
        if user_first_name or user_last_name or user_email or user_phone:
            await send_to_n8n(user_first_name, user_last_name, user_email, user_phone, agent_url)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_url": agent_url,
            "embed_code": embed_code,
            "message": f"Agent '{agent_info.name}' created successfully!"
        }
    
    except HTTPException:
        raise
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
        
        # Get basic extracted fields from infra (other_instructions is now extracted from transcript)
        agent_name = extracted_data.get("agent_name") or ""
        agent_persona = extracted_data.get("agent_persona") or ""
        agent_purpose = extracted_data.get("agent_purpose") or ""
        company_name = extracted_data.get("company_name") or ""
        # website_url = extracted_data.get("website_url") or ""
        voice_gender = extracted_data.get("voice_gender") or ""
        
        # Extract user contact info (separate from agent info)
        user_first_name = extracted_data.get("user_first_name") or None
        user_last_name = extracted_data.get("user_last_name") or None
        user_email = extracted_data.get("user_email") or None
        user_phone = extracted_data.get("user_phone") or None
        
        if user_first_name or user_last_name or user_email or user_phone:
            print(f"[LOG] User contact info extracted:")
            print(f"[LOG]   - First Name: {user_first_name}")
            print(f"[LOG]   - Last Name: {user_last_name}")
            print(f"[LOG]   - Email: {user_email}")
            print(f"[LOG]   - Phone: {user_phone}")
        
        # Map voice_gender to ElevenLabs voice_id
        voice_id = "esunmKO3sPWJNo7W1w0t"  # Default to female voice
        if voice_gender:
            voice_gender_lower = voice_gender.lower().strip()
            if "masculine" in voice_gender_lower or "male" in voice_gender_lower:
                voice_id = "UgBBYS2sOqTuMpoF3BR0"  # Male voice
            elif "feminine" in voice_gender_lower or "female" in voice_gender_lower:
                voice_id = "esunmKO3sPWJNo7W1w0t"  # Female voice
            print(f"[LOG] Voice gender extracted: {voice_gender} → voice_id: {voice_id}")
        else:
            print(f"[LOG] No voice_gender found, using default: {voice_id}")
        
        # Extract structured data from transcript using AI
        structured_data = None
        if TRANSCRIPT_EXTRACTOR_AVAILABLE and extract_structured_data_from_transcript:
            try:
                # Get transcript from webhook (could be in transcript, call_transcript, or analysis.transcript)
                transcript = (
                    body_data.get("transcript") or
                    body_data.get("call_transcript") or
                    analysis.get("transcript") or
                    body_data.get("call_transcript_processed") or
                    ""
                )
                
                if transcript:
                    print(f"[LOG] Extracting structured data from transcript (length: {len(transcript)} chars)")
                    basic_fields = {
                        "agent_name": agent_name,
                        "agent_persona": agent_persona,
                        "agent_purpose": agent_purpose,
                        "company_name": company_name
                        # "website_url": website_url
                        # other_instructions is now extracted from transcript, not from infra
                    }
                    structured_data = await extract_structured_data_from_transcript(transcript, basic_fields)
                    print(f"[LOG] Structured data extracted - Category: {structured_data.category}")
                    if structured_data.other_instructions:
                        print(f"[LOG] Other instructions extracted from transcript ({len(structured_data.other_instructions)} rules):")
                        for idx, rule in enumerate(structured_data.other_instructions, 1):
                            print(f"[LOG]   {idx}. {rule}")
                else:
                    print(f"[WARNING] No transcript found in webhook data")
            except Exception as e:
                print(f"[ERROR] Error extracting structured data from transcript: {str(e)}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        # Map to AgentInfo structure
        # Use other_instructions from extracted structured_data, fallback to empty list
        other_instructions = (structured_data.other_instructions if structured_data and structured_data.other_instructions else []) or []
        
        agent_data = {
            "name": agent_name,
            "persona": agent_persona,
            "purpose": agent_purpose,
            "other_instructions": other_instructions,
            "company_name": company_name,
            # "website_url": website_url if website_url else None,
            "voice_id": voice_id
        }
        
        # Store structured_data for later use in workflow
        agent_data["_structured_data"] = structured_data
        
        # Validate required fields
        if not agent_data["name"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required field: agent_name not found in extracted data"}
            )
        
        if not agent_data["purpose"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required field: agent_purpose not found in extracted data"}
            )
        
        # Set defaults for missing optional fields
        if not agent_data["persona"]:
            agent_data["persona"] = "Friendly and professional"
        if not agent_data["company_name"]:
            agent_data["company_name"] = "Demo Company"
        
        # Create agent info object (remove _structured_data as it's not part of AgentInfo)
        structured_data = agent_data.pop("_structured_data", None)
        agent_info = AgentInfo(**agent_data)
        
        # Create agent with user contact info (will be sent to n8n after agent creation)
        result = await create_agent(
            agent_info, 
            structured_data,
            user_first_name=user_first_name,
            user_last_name=user_last_name,
            user_email=user_email,
            user_phone=user_phone
        )
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

