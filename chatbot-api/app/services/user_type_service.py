from app.schemas.chat_schema import UserType
from app.core.config import WEBSITE_NAME

def get_system_prompt_by_user_type(role: UserType) -> str:
    
    if WEBSITE_NAME == 'diplomacy.edu':
        from app.ai.prompts.diplomacy_edu_prompts import SYSTEM_PROMPTS
    elif WEBSITE_NAME == 'humainism.ai':
        from app.ai.prompts.humainisam_ai_prompts import SYSTEM_PROMPTS
    else:
        from app.ai.prompts.diplomacy_edu_prompts import SYSTEM_PROMPTS
    
    return SYSTEM_PROMPTS.get(role, "")