def route_decision(classification: Dict[str, Any]) -> Dict[str, str]:
    """
    Determine the routing action based on classification.
    
    Args:
        classification: Dict from classify_message()
    
    Returns:
        Dict with:
        {
            "action": "chat" | "escalate_human",
            "reason": str (explanation)
        }
    """
    needs_human_prob = classification["needs_human"]
    department = classification["department"]
    
    # If user needs human assistance, escalate
    if needs_human_prob > NEEDS_HUMAN_THRESHOLD:
        return {
            "action": "escalate_human",
            "reason": f"User needs human assistance (confidence: {needs_human_prob:.2f}). Escalating to {department} team."
        }
    
    # Otherwise, use conversational assistant
    return {
        "action": "chat",
        "reason": "Routing to conversational assistant."
    }
