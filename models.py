"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional, List


class AgentInfo(BaseModel):
    """Model for agent creation and prompt generation requests"""
    name: str
    persona: str
    purpose: str
    other_instructions: Optional[List[str]] = []  # Extracted from transcript as list of rules, defaults to empty list
    company_name: str
    website_url: Optional[str] = None
    prompt_template: Optional[str] = None
    voice_id: Optional[str] = "11labs-Cimo"
    language: Optional[str] = "en-US"
    llm_id: Optional[str] = None  # Use existing LLM if provided

class WebhookData(BaseModel):
    """Model for webhook data (flexible schema)"""
    model_config = ConfigDict(extra="allow")  # Allow extra fields
    
    agent_name: Optional[str] = None
    persona: Optional[str] = None
    purpose: Optional[str] = None
    company_name: Optional[str] = None
    website_url: Optional[str] = None

class PostCallAnalysisField(BaseModel):
    """Model for post-call analysis field structure"""
    type: str
    name: str
    description: str
    examples: Optional[List[str]] = None

