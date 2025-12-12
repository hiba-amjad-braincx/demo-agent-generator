"""
Agent Builder Workflow Functions
Contains the workflow logic for generating prompts using OpenAI Agent Builder SDK
Also includes AI-powered transcript extraction for structured data
"""

try:
    from agents import FileSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace
    AGENT_BUILDER_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "Agent Builder SDK is required for workflow.py. "
        "Please install the correct 'agents' package. "
        f"Error: {e}"
    )

import os
import json
import openai
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# Tool definitions
file_search = FileSearchTool(
    vector_store_ids=[
        "vs_6938802e677481919fbe6729f6271735"
    ]
)

prompt_generator = Agent(
    name="Prompt generator",
    instructions="""You are a system designed to generate production-quality system prompts for Voice AI agents.
Your role is to transform structured configuration data into a polished system prompt that fully defines how the final Voice AI agent behaves.

You produce prompts that: 
Begin strictly with "You are…" 
Contain only system instructions 
Exclude introductions, greetings, and all first-person language 
Include personality, tone, company context, and purpose 
Integrate all structured settings from the input (category, task type, fields, availability rules, collection goals, etc.)
Follow all best practices from stored examples and knowledge files 
Reduce hallucination, repetition, and ambiguity 
Use short, clear, intentional paragraphs and/or bullets

ABSOLUTE RULES 
Never use emojis. 
Include negative instructions to prevent mistakes ("Do not…", "Avoid…"). 
Follow all structural patterns, formatting styles, and conventions from the uploaded prompt examples using file search. 
Never output conversational language or meta commentary. 
The final system prompt must explicitly instruct the agent to ask exactly one question at a time and to wait for the user’s answer before proceeding to the next question. The agent must never ask multiple questions at once, never bundle questions together, and never continue until a response is received.

RELIABLE INFORMATION COLLECTION RULES 
These rules must always be embedded into every system prompt you generate. 
1. Name Collection + Spelling 
Before assisting with any request involving accounts, billing, payments, appointments, sending information, or personal data, the agent must collect the customer's full name first. 
If the name was already provided earlier, the agent must not ask again and must use the stored name. 
If a name is unclear or uncommon, the agent must ask for spelling and confirm it by spelling it back. 
General informational questions do not require name collection. 

2. Universal Number Rule 
For all numbers — including phone numbers, postal codes, account numbers, confirmation codes, reference numbers, unit numbers, and the last four digits of credit cards: 
The agent must capture and repeat all numbers digit-by-digit, never as whole numbers. 
The agent must never guess or alter digits. 
If unsure, the agent must ask the user to repeat the number. 
Emergency numbers must always be spelled out ("nine one one"). 
Exceptions: 
Monetary amounts may be written normally (e.g., $135). 
Years may be spoken normally (e.g., 2024). 
Dates of Birth may be written in normal date format (e.g., “January 12, 1998” or “12/01/1998”), not digit-by-digit.

3. Email Addresses 
When a user spells an email, the agent must capture every character exactly as spoken. 
Confirm email in standard email format (username@domain.com). 
Always repeat the email and end with: "Is that correct?" 
Ask for clarification when letters and digits sound similar (i/1, o/0, e/3, a/8). 

4. Credit Cards 
Accept full card numbers politely but never repeat the full number. 
Confirm only the last four digits, spoken digit-by-digit. 

5. Confirmation Protocol 
For all critical information (names, numbers, emails, addresses): 
Repeat the information using the correct standardized format. 
Use clarity cues ("Just to confirm…"). 
Always end with: "Is that correct?" 
If corrected by the user, restate the corrected information and reconfirm. 

6. No Repetition
The agent must never ask for information already provided. 
Once a field is captured, it is stored and reused automatically. 

7. Auto-Detect Reason for Request
If the user naturally states the reason for contacting, treat it as the captured “reason” field. 
Do not ask for it again unless unclear.

8. Dynamic Field Tracking
The agent must track collected fields in real time and only ask for the next missing item. 
Avoid unnecessary confirmations.

STRUCTURED DATA INTEGRATION RULES 
You must incorporate all structured configuration data into the final system prompt. 
The input may include the following fields: 

Core Fields:
agent_name 
persona 
purpose 
company_name 
website_url 
other_instructions - Special rules or behaviors (if provided)

Category
Use the category to determine which specialized logic must appear in the system prompt. 

CATEGORY A — Task Handling 
If Category = "A", integrate the following fields: 
task_type (e.g., Appointment Booking, Restaurant Reservation, Order Tracking, Returns/Exchanges, Tech Troubleshooting, Other) 
required_fields (only the fields relevant to the task) 
availability_handling (either "Offer available date/time slots" or "Ask for preferred date/time only") 
human_escalation (yes/no — instruct how the agent should escalate to a human) 

The system prompt must:
Describe how the agent collects the specified fields 
Describe how the agent handles scheduling or availability if applicable 
Describe how troubleshooting or order flow, or account/service actions
Specify when and how the agent should escalate to a human 

CATEGORY B — Lead / Information Collection 
If Category = "B", integrate the following fields: 
collection_goal (Lead Generation, Support Ticket Intake, Application/Onboarding, Survey/Feedback) 
required_fields (all fields the agent must collect) 

The system prompt must:
Define the purpose of collecting user information 
Specify instructions for gathering each required field 
Enforce sequencing or formatting constraints 

CATEGORY C — Information & Support 
If Category = "C", integrate: 
information_type (Product Details, Pricing & Plans, Troubleshooting Guides, Company Info, General FAQs) 
user_information ("none", "name only", or "name + email") 

The system prompt must:
Instruct how the agent answers questions 
Specify what information the agent can provide 
Define whether user details should be collected before answering 

CATEGORY Z — Other / Unclear 
If Category = "Z", integrate: 
interaction_type ("collect information", "provide information", or "handle tasks") 

The system prompt must:
Define the correct behavioral pattern for the selected interaction type
Avoid assumptions outside the given data

YOUR WORKFLOW 
When generating the final system prompt:
Use the exact structured data provided.
Never ask clarifying questions.
Never add behavior that isn't explicitly required or defined in the stored prompt examples.
Follow structural patterns found in stored prompt examples.

Ensure the final output is:
concise
deterministic
compliant with Voice AI constraints
non-redundant
ready for deployment

FINAL OUTPUT FORMAT
A polished, production-ready system prompt that:
Begins with "You are…"
Uses formal system-instruction tone
Integrates all category-specific and task-specific settings
Reflects the agent persona, purpose, company context, and rules
Embeds all mandatory reliability rules
Is concise, unambiguous, and ready for Voice AI deployment""",
    model="gpt-4.1",
    tools=[
        file_search
    ],
    model_settings=ModelSettings(
        temperature=0,
        top_p=1,
        max_tokens=10000,
        store=True
    )
)


class ExtractedStructuredData(BaseModel):
    """Structured data extracted from transcript - all Phase 2 answers"""
    category: Optional[str] = None  # A, B, C, or Z
    task_type: Optional[str] = None  # For Category A
    required_fields: Optional[List[str]] = None  # List of fields to collect
    availability_handling: Optional[str] = None  # For appointments/reservations
    human_escalation: Optional[bool] = None  # For Category A
    collection_goal: Optional[str] = None  # For Category B
    information_type: Optional[str] = None  # For Category C
    user_information: Optional[str] = None  # For Category C
    interaction_type: Optional[str] = None  # For Category Z
    other_instructions: Optional[List[str]] = None  # Special rules or behaviors as list


class WorkflowInput(BaseModel):
    input_as_text: str

async def run_workflow(workflow_input: WorkflowInput):
    with trace("Prompt generator"):
        state = {
            "agent_name": None,
            "persona": None,
            "purpose": None,
            "company_name": None,
            "other_instructions": None,
            "website_url": None,
            "category": None,
            "task_type": None,
            "required_fields": [
                
            ],
            "availability_handling": None,
            "human_escalation": None,
            "collection_goal": None,
            "information_type": None,
            "user_information": None,
            "interaction_type": None
        }
        workflow = workflow_input.model_dump()
        conversation_history: list[TResponseInputItem] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": workflow["input_as_text"]
                    }
                ]
            }
        ]
        prompt_generator_result_temp = await Runner.run(
            prompt_generator,
            input=[
                *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_693880309e3c8190b1bae2be368ece6a05f8eb6d0b499de0"
            })
        )

        conversation_history.extend([item.to_input_item() for item in prompt_generator_result_temp.new_items])

        prompt_generator_result = {
            "output_text": prompt_generator_result_temp.final_output_as(str)
        }
        
        return prompt_generator_result


async def extract_structured_data_from_transcript(
    transcript: str,
    extracted_basic_fields: Dict[str, str]
) -> ExtractedStructuredData:
    """
    Use AI to extract structured data from the full transcript
    
    Args:
        transcript: Full conversation transcript
        extracted_basic_fields: Already extracted fields (agent_name, persona, purpose, etc.)
        
    Returns:
        ExtractedStructuredData with all structured information
    """
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Build prompt for extraction
    extraction_prompt = f"""You are analyzing a conversation transcript from a demo agent builder. Extract structured information based on the conversation.

BASIC FIELDS ALREADY EXTRACTED (if available):
- Agent Name: {extracted_basic_fields.get('agent_name', 'N/A')}
- Persona: {extracted_basic_fields.get('agent_persona', 'N/A')}
- Purpose: {extracted_basic_fields.get('agent_purpose', 'N/A')}
- Company Name: {extracted_basic_fields.get('company_name', 'N/A')}
- Website URL: {extracted_basic_fields.get('website_url', 'N/A')}
Note: Other Instructions will be extracted from the transcript below.

TRANSCRIPT:
{transcript}

EXTRACTION INSTRUCTIONS:
Based on the transcript, extract the following structured information:

1. CATEGORY: Determine which category the purpose falls into:
   - "A" for Task Handling (book, schedule, appointment, reserve, order, track, return, replace, fix, troubleshoot, or account/service actions)
   - "B" for Lead / Information Collection (collect, form, intake, qualify, onboarding, sign-up, survey, application, feedback)
   - "C" for Information & Support (answer, explain, FAQ, support, guide, help center, pricing, product info)
   - "Z" for Other / Unclear

2. CATEGORY-SPECIFIC FIELDS:

For Category A (Task Handling):
- task_type: Extract the exact task type mentioned (e.g., "Appointment Booking", "Restaurant Reservation", "Order Tracking", "Returns/Exchanges", "Tech Troubleshooting", or "Other")
- required_fields: Extract list of fields mentioned (e.g., ["Name", "Phone", "Email", "Address", "ZIP/Postcode", "Date/Time Preference", "Order Number", "Issue Description"])
- availability_handling: If task involves scheduling, extract how availability is handled ("Offer available date/time slots" or "Ask for preferred date/time only")
- human_escalation: Extract boolean - should agent offer to transfer to human? (true/false)

For Category B (Lead / Information Collection):
- collection_goal: Extract the main goal (e.g., "Lead Generation", "Support Ticket Intake", "Application/Onboarding", "Survey/Feedback")
- required_fields: Extract list of fields mentioned (e.g., ["Name", "Email", "Phone", "Company Name", "Job Title", "Budget", "Project Details", "Reason for Contact"])

For Category C (Information & Support):
- information_type: Extract primary information type (e.g., "Product Details", "Pricing & Plans", "Troubleshooting Guides", "Company Info", "General FAQs")
- user_information: Extract what user info to collect ("none", "name only", or "name + email")

For Category Z (Other / Unclear):
- interaction_type: Extract how agent should interact ("collect information", "provide information", "handle tasks")

3. OTHER INSTRUCTIONS:
Extract ALL special rules, behaviors, conversational flows, and step-by-step procedures from the USER'S ACTUAL RESPONSES throughout the ENTIRE conversation.
- CRITICAL: Extract from the USER'S original responses in the transcript, NOT from the agent's summary at the end (which is often too concise)
- Extract rules and flows from ANY point in the conversation where the user provides instructions, not just after specific questions
- Include ALL instructions the user provides: greeting scripts, disclaimers, detailed conversational flows, step-by-step procedures, conditional logic, specific dialogue/phrases, recap requirements, ending scripts, etc.
- When the user describes detailed flows, extract these as complete flow instructions
- Extract each distinct rule, flow, or instruction as a separate item in a list
- Preserve the user's exact wording and intent - do not summarize, condense, or skip steps in multi-step flows
- Capture the FULL detail of procedural instructions and conversational sequences
- If the user said "none" or provided no instructions, set to null or empty list []
- Example: ["First ask for ZIP code", "Then ask if this is their first time booking an appointment", "If they have done it before, ask if they would like to use their insurance", "Ask for therapy details including type: individual, couples, family, child therapy, psychiatry, or medication", "Ask for therapy type they're struggling with: anxiety, depression, grief, loss, relationship issues, trauma, or PTSD", "Ask for time preference: morning, afternoon, or evening", "Then offer available date time slots"]

Return your response as a JSON object with the following structure:
{{
    "category": "A" | "B" | "C" | "Z" | null,
    "task_type": "string or null",
    "required_fields": ["field1", "field2"] or null,
    "availability_handling": "string or null",
    "human_escalation": true/false/null,
    "collection_goal": "string or null",
    "information_type": "string or null",
    "user_information": "string or null",
    "interaction_type": "string or null",
    "other_instructions": ["rule1", "rule2", ...] or null
}}

Only include fields that are relevant to the detected category. Set irrelevant fields to null.
"""
    
    try:
        print(f"[LOG] Calling OpenAI API for structured data extraction...")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Extract structured information from conversation transcripts and return valid JSON only."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        extracted_json = json.loads(response.choices[0].message.content)
        print(f"[LOG] Raw extracted JSON: {json.dumps(extracted_json, indent=2)}")
        
        # Convert to ExtractedStructuredData
        structured_data = ExtractedStructuredData(**extracted_json)
        
        # Log the extracted structured data
        print(f"[LOG] Structured data extraction successful:")
        print(f"[LOG]   - Category: {structured_data.category}")
        if structured_data.category == "A":
            print(f"[LOG]   - Task Type: {structured_data.task_type}")
            print(f"[LOG]   - Required Fields: {structured_data.required_fields}")
            print(f"[LOG]   - Availability Handling: {structured_data.availability_handling}")
            print(f"[LOG]   - Human Escalation: {structured_data.human_escalation}")
        elif structured_data.category == "B":
            print(f"[LOG]   - Collection Goal: {structured_data.collection_goal}")
            print(f"[LOG]   - Required Fields: {structured_data.required_fields}")
        elif structured_data.category == "C":
            print(f"[LOG]   - Information Type: {structured_data.information_type}")
            print(f"[LOG]   - User Information: {structured_data.user_information}")
        elif structured_data.category == "Z":
            print(f"[LOG]   - Interaction Type: {structured_data.interaction_type}")
        if structured_data.other_instructions:
            print(f"[LOG]   - Other Instructions ({len(structured_data.other_instructions)} rules):")
            for idx, rule in enumerate(structured_data.other_instructions, 1):
                print(f"[LOG]     {idx}. {rule[:80]}...")
        
        return structured_data
        
    except Exception as e:
        print(f"[ERROR] Error extracting structured data from transcript: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        # Return empty structure on error
        return ExtractedStructuredData()

