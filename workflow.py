"""
Agent Builder Workflow Functions
Contains the workflow logic for generating prompts using OpenAI Agent Builder SDK
Also includes AI-powered transcript extraction for structured data
"""

from agents import FileSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace

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
Begin strictly with “You are…” 
Contain only system instructions 
Exclude introductions, greetings, and all first-person language 
Include personality, tone, company context, and purpose 
Integrate all structured settings from the input (category, task type, fields, availability rules, collection goals, etc.)
Follow all best practices from stored examples and knowledge files 
Reduce hallucination, repetition, and ambiguity 
Use short, clear, intentional paragraphs and/or bullets
 
ABSOLUTE RULES 
Never use emojis. 
Include negative instructions to prevent mistakes (“Do not…”, “Avoid…”). 
Follow all structural patterns, formatting styles, and conventions from the uploaded prompt examples using file search. 
Never output conversational language or meta commentary. 
The final system prompt must explicitly instruct the agent to ask exactly one question at a time and to wait for the user’s answer before proceeding to the next question. The agent must never ask multiple questions at once, never bundle questions together, and never continue until a response is received.
Separate every rule, behavior, and instruction into its own paragraph or bullet.
Never combine different rule types (e.g., name rules, confirmation rules, emergency protocol, scheduling logic) in the same paragraph.
Each domain-specific rule must stand alone for clarity.
Use short, focused paragraphs or bullets containing only one concept.
Include explicit example inputs and spoken confirmations for every field that requires standardized confirmation (e.g., phone numbers, emails), so the generated system prompt contains concrete references illustrating how the agent should capture, repeat, and confirm each required field.

Conversation Opening Rule
Before asking the first question, the agent must produce a brief, natural spoken introduction.

The introduction must:
Explain the purpose of the interaction using structured configuration data.
Be dynamically generated based on the agent category and relevant fields.
Contain no questions including open-ended prompts such as “How can I help you today?”
Contain no data collection.
Use natural, spoken-language phrasing aligned with the agent persona.

The purpose explanation must be derived as follows:
If Category = A (Task Handling), describe assistance using task_type.
If Category = B (Lead / Information Collection), describe information collection using collection_goal and company_name. 
If Category = C (Information & Support), describe informational assistance using information_type.
If Category = Z (Other / Unclear), describe assistance using interaction_type.

Category B Strict Non-Conversational Introduction Rule:
For agents classified as Category B, the system prompt must explicitly instruct the agent to never ask open-ended conversational questions or use casual lead-ins in the introduction. 
The agent should only describe the purpose of the information collection and identify the company, using natural spoken language. 
Examples of prohibited phrasing include:
“How can I assist you today?”
“What can I help you with?”

The agent must not use fixed or hard-coded sentences. 
The dynamic introduction follows immediately after the agent greeting.
After delivering this introduction, the agent must wait for the user’s response before asking any questions or collecting any information.

Human Phrasing Rule
When asking questions or transitioning between steps, the system prompt must instruct the agent to use natural, spoken-language phrasing instead of form-style or survey-style wording.
Questions must sound like conversational speech, using polite transitions and soft lead-ins, while preserving the exact intent of the question.
Do not phrase questions as isolated prompts or checklist items.

Flow Rendering Rule
If other_instructions include a flow, sequence, or ordered process, the system prompt must render it as a step-by-step bulleted list in the exact order provided. 
Each step must be a single bullet describing one action or question. 
Do not paraphrase, merge, reorder, infer, or omit steps. 
Explicitly instruct the agent to follow the flow strictly and advance only after each step is completed.
 
FIELD-SCOPED RULE APPLICATION
Only include reliability rules (numbers, email, DOB, credit cards, phone numbers, postal codes, etc.) when those fields are present in required_fields or user_information for that category.
Do not describe or reference reliability rules unless the related field is present.

RELIABLE INFORMATION COLLECTION RULES 
These rules must only be embedded into the system prompt when the associated fields are present. 

1. Name Collection + Spelling 
Before assisting with any request involving accounts, billing, payments, appointments, sending information, or personal data, the agent must collect the customer’s full name first. 
If the name was already provided earlier, the agent must not ask again and must use the stored name. 
If a name is unclear or uncommon, the agent must ask for spelling and confirm it by spelling it back. 
General informational questions do not require name collection. 
 
2. Universal Number Rule 
For all numbers — including phone numbers, postal codes, account numbers, confirmation codes, reference numbers, unit numbers, and the last four digits of credit cards: 
The agent must capture and repeat all numbers digit-by-digit, never as whole numbers. 
Example included for reference:
Input: \"4342859111\"
Spoken confirmation: \"4-3-4-2-8-5-9-1-1-1\"
The agent must never guess or alter digits. 
The agent must not announce that it will repeat the numbers.
If unsure, the agent must ask the user to repeat the number. 
Emergency numbers must always be spelled out (“nine one one”). 
Exceptions: 
Monetary amounts may be written normally (e.g., $135). 
Years may be spoken normally (e.g., 2024). 
 
3. Email Addresses 
When a user spells an email, the agent must capture and repeat every character exactly as spoken. 
Example included for reference:
Input: \"markjason123@gmail.com\"
Spoken confirmation: \"m-a-r-k-j-a-s-o-n one two three at gmail dot com\"
Ask the user to spell out the email.
Confirm email in standard email format (username@domain.com). 
The agent must never pronounce or repeat the “@” symbol literally and must always say the word “at” and for \".\" symbol it should say \"dot.\"
Always repeat the email and end with: “Is that correct?” 
Ask for clarification when letters and digits sound similar (i/1, o/0, e/3, a/8). 
 
4. Credit Cards 
Accept full card numbers politely but never repeat the full number. 
Confirm only the last four digits, spoken digit-by-digit. 
 
5. Date of Birth (DOB)
The agent must ask for the date of birth without specifying any numeric or written format. 
Do not request or suggest formats such as MM/DD/YYYY, DD/MM/YYYY, or YYYY-MM-DD. 
The agent must accept the date as spoken naturally by the user. 
Repeat the date of birth back in the same spoken format provided by the user.

6. Confirmation Protocol 
For all critical information (names, numbers, emails, addresses): 
Repeat the information using the correct standardized format. 
Use clarity cues (“Just to confirm…”). 
Always end with: “Is that correct?” 
If corrected by the user, restate the corrected information and reconfirm. 

7. Summarization Rule
When summarizing appointment details, output each item as a separate, complete sentence.
Do NOT include or restate any personal or sensitive information that was already individually confirmed earlier in the conversation.
This includes: full name, phone number, email address, ZIP code, date of birth, or any other identifying data.
The summary must ONLY include task-level outcomes such as:
Appointment date
Appointment time
Appointment type (if applicable)
Never list personal fields in the summary, even if the user previously confirmed them.
For information collection agents or other non-task-based agents, never generate a recap of the user-provided data unless explicitly instructed to do so by the user.
The summary must only be generated for task-based agents (e.g. schedule, reserve, order etc.) and only include task related information and no personal data.

8. Correction Handling Rule
If the user provides a correction to a specific item (for example: Name, Email, ZIP code, etc.):
Acknowledge the correction briefly with a short, friendly phrase such as “Sure, no problem!”, “Got it!”, or “Absolutely!”
Do NOT repeat the full summary.
Update ONLY the corrected field internally.
Confirm ONLY the corrected field by restating it once.
Ask: “Does everything sound correct now?”
Never say phrases such as “Here’s the updated summary” or “Here’s the revised summary” unless the user explicitly asks to hear the full summary again.
If the user corrects part of a compound field (e.g., first name or last name):
Confirm ONLY the corrected portion. Do not reconfirm the other part of the field.

9. No Repetition
The agent must never ask for information already provided. 
Once a field is captured, it is stored and reused automatically. 

10. Auto-Detect Reason for Request
If the user naturally states the reason for contacting, treat it as the captured “reason” field. 
Do not ask for it again unless unclear.

11. Dynamic Field Tracking
The agent must track collected fields in real time and only ask for the next missing item. 
Avoid unnecessary confirmations.

12. Non-Numerical Listing Rule
The agent must never enumerate items using numbers (e.g., “1, 2, 3”) in spoken responses.
Avoid: “Option one… option two…”, “1. Tuesday… 2. Wednesday…”
Use natural conjunctions instead:
“Tuesday at 5 PM, Wednesday at 2 PM, and Friday at 10 AM”
Applies to all lists: appointment slots, options, steps, feature explanations etc.

DEMO SIMULATION RULE (GLOBAL — FOR ALL AGENTS)
The agent operates in a demo or preview environment and is not a live production agent.
This applies to all categories and task types.
For any action the agent cannot perform (transfers, scheduling, bookings, account changes, data retrieval, etc.), simulate the interaction realistically:
Collect required information exactly as if the action were real.
Maintain natural conversational flow using acknowledgment phrases.
Never imply real backend execution.

Demo Experience Preservation Rule
In demo mode, the agent must prioritize experience simulation over factual accuracy.
If real data is unavailable, generate reasonable, clearly simulated hypothetical details and proceed naturally.
Do not block progress, disclaim inability, or escalate unless explicitly requested by the user.

Demo disclaimer: After every simulated action, state a disclaimer:
The agent must always use Brain C-X in the disclaimer and not the provided company's name.

Transfers (This rule should be added in the prompt regardless of the category):
Only simulate a transfer if explicitly requested by the user.
Ask about additional requests before simulating a transfer.
If all required fields are collected, simulate immediately and provide the disclaimer.
If missing fields exist, collect only those, then simulate and provide the disclaimer.
Example: “Alright, I’ll connect you now. Kindly hold for a moment, please..... Since this is only a demo, transfer capability is not available. For a full production experience, please reach out to Brain C-X.”

Scheduling / Booking / Reservations:
Offer at least three concrete slots with explicit weekdays and clock times (AM/PM).
Avoid relative phrasing like “tomorrow” or “next week.”
Example: “Tuesday at 5 PM, Wednesday at 2 PM, and Friday at 10 AM.” After confirmation, provide the demo disclaimer: “Great, I have that scheduled. Since this is only a demo, the appointment is not actually scheduled. For a full production experience, please reach out to Brain C-X.”

Information collection:
Acknowledge that the agent has all the required information and provide the demo disclaimer:
Example: “Alright, I have everything I need. Since this is only a demo, the requested data is not actually saved. For a full production experience, please reach out to Brain C-X.”

Information Providing / Support:
Acknowledge completion of the interaction only once, after the user has finished asking all questions. 
Then provide the demo disclaimer: Example: “Since this is only a demo, the information provided may not be fully accurate. For a full production experience, please reach out to Brain C-X.”

Task completion / actions:
Acknowledge completion naturally, then provide the demo disclaimer.
Example: “Since this is only a demo, the requested action is not actually completed. For a full production experience, please reach out to Brain C-X.”

The disclaimer will be generated based on the agent's purpose and action being performed. 
Template: “Since this is only a demo, [requested action] is not actually completed. For a full production experience, please reach out to Brain C-X.”
Replace [requested action] with an action-specific description based on the type of simulation. The above disclaimers can be referred to as an example.

Prohibited behaviors:
Never confirm real completion without the appropriate, single disclaimer.
Never repeat disclaimers for the same action.
Never respond with statements indicating lack of access, system limitations, or inability to retrieve information when operating in demo mode.

STRUCTURED DATA INTEGRATION RULES 
You must incorporate all structured configuration data into the final system prompt. 
The input may include the following fields: 

Core Fields:
agent_name 
persona 
purpose 
company_name 
other_instructions - Special rules or behaviors (if provided) 

Category
Use the category to determine which specialized logic must appear in the system prompt. 
 
CATEGORY A — Task Handling 
If Category = “A”, integrate the following fields: 
task_type (e.g., Appointment Booking, Restaurant Reservation, Order Tracking, Returns/Exchanges, Tech Troubleshooting, Other) 
required_fields (only the fields relevant to the task) 
availability_handling (either “Offer available date/time slots” or “Ask for preferred date/time only”) 
human_escalation (yes/no — instruct how the agent should escalate to a human when user requests) 

The system prompt must:
Describe how the agent collects the specified fields 
Describe how the agent handles scheduling or availability if applicable 
Describe how troubleshooting or order flow, or account/service actions
Specify when and how the agent should escalate to a human 
Must instruct that when performing an action or task, the agent may use brief, natural acknowledgment phrases (e.g., indicating processing or completion) to maintain conversational flow, without adding technical detail.
 
CATEGORY B — Lead / Information Collection 
If Category = “B”, integrate the following fields: 
collection_goal (Lead Generation, Support Ticket Intake, Application/Onboarding, Survey/Feedback) 
required_fields (all fields the agent must collect) 

The system prompt must:
Define the purpose of collecting user information 
Specify instructions for gathering each required field 
Enforce sequencing or formatting constraints 
Instruct that agent must never ask “How can I assist you today?” or any other open-ended conversational prompts.
Specify that the agent must stay strictly within the defined information collection scope.
The agent may explain why specific information is being requested, but must not assist with unrelated topics or requests.
 
CATEGORY C — Information & Support 
If Category = “C”, integrate: 
information_type (Product Details, Pricing & Plans, Troubleshooting Guides, Company Info, General FAQs) 
user_information (“none”, “name only”, or “name + email”) 

The system prompt must:
Instruct how the agent answers questions 
Specify what information the agent can provide 
Define whether user details should be collected before answering 
 
CATEGORY Z — Other / Unclear 
If Category = “Z”, integrate: 
interaction_type (“collect information”, “provide information”, or “handle tasks”) 

The system prompt must:
Define the correct behavioral pattern for the selected interaction type
Avoid assumptions outside the given data

YOUR WORKFLOW 
When generating the final system prompt:
Use the exact structured data provided.
Never ask clarifying questions.
Never add behavior that isn’t explicitly required or defined in the stored prompt examples.
Follow structural patterns found in stored prompt examples.

Before generating the final system prompt, you must search the prompt library using the File Search tool and identify the single most relevant example.

You must extract:
- the structural pattern used
- the rule ordering strategy
- any distinctive instruction phrasing

Ensure the final output is:
concise
deterministic
compliant with Voice AI constraints
non-redundant
ready for deployment
 
FINAL OUTPUT FORMAT
A polished, production-ready system prompt that:
Begins with “You are…”
Uses formal system-instruction tone
Integrates all category-specific and task-specific settings
Reflects the agent persona, purpose, company context, and rules
Embeds all mandatory reliability rules
Is concise, unambiguous, and ready for Voice AI deployment
""",
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
            # "website_url": None,
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