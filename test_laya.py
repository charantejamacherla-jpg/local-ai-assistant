"""Phase 1: Minimal Laya classification test using verified current API.

This script tests the Laya classifier with 3 sample messages:
1. Billing/refund (high urgency, needs human)
2. Technical/bug (medium urgency, maybe needs help)
3. General inquiry (low urgency, no human needed)

It prints both the complete raw result and a simplified parsed view.
No integration with Ollama chat yet - Phase 1 verification only.
"""
import os
import sys
import time
import json

# Set environment before importing torch/laya
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Now safe to import laya
import laya

print("=" * 70)
print("PHASE 1: Laya Classification Test")
print("=" * 70)
print()

# Show version
print(f"Laya version: {laya.__version__}")
print()

# Define test messages
test_messages = [
    "I was charged twice for March. Please refund the duplicate today or we will cancel our plan.",
    "The app crashes every time I open the settings menu. I've tried reinstalling twice.",
    "What are your business hours and holiday schedule?",
]

# Define classification questions (verified from official docs)
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this message?",
        "criteria": {
            "billing": "invoices, payments, refunds, subscription issues",
            "technical": "bugs, crashes, outages, system errors",
            "account": "password resets, account access, profile changes",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["low", "medium", "high"]
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does the user need human assistance right now?"
    }
}

print("Initializing Router...")
print("(First run will download model checkpoint ~500MB)")
print()

# Initialize router (lazy load on first predict)
start_init = time.time()
router = laya.Router()
init_time = time.time() - start_init
print(f"Router initialized in {init_time:.2f}s (no model loaded yet)")
print()

# Process each test message
for idx, message in enumerate(test_messages, 1):
    print("-" * 70)
    print(f"TEST {idx}: {message[:60]}...")
    print("-" * 70)
    
    # Run prediction and measure time
    start_predict = time.time()
    result = router.predict(message, questions)
    predict_time = time.time() - start_predict
    
    print()
    print(f"Inference time: {predict_time*1000:.1f}ms")
    print()
    
    # ===== 1. COMPLETE RAW RESULT =====
    print("COMPLETE RAW RESULT:")
    print(json.dumps(result, indent=2))
    print()
    
    # ===== 2. PARSE AND EXTRACT FIELDS =====
    print("PARSED RESULTS:")
    
    answers = result["answers"]
    
    # Department (choice)
    dept_answer = answers["department"]
    print(f"\n  Department:")
    print(f"    type: {dept_answer['type']}")
    print(f"    choice: {dept_answer['choice']}")
    print(f"    probabilities: {dept_answer['probabilities']}")
    print(f"    confidence: {dept_answer['confidence']:.4f}")
    print(f"    answer_confidence: {dept_answer['answer_confidence']:.4f}")
    
    # Urgency (score)
    urgency_answer = answers["urgency"]
    urgency_score = urgency_answer["score"]
    # FIXED: Use probabilities to find the winning category, not int(score)
    urgency_legend = urgency_answer["legend"]
    urgency_probabilities = urgency_answer["probabilities"]
    # Find the key with the highest probability
    winning_key = max(urgency_probabilities, key=urgency_probabilities.get)
    urgency_label = urgency_legend[winning_key]
    
    print(f"\n  Urgency:")
    print(f"    type: {urgency_answer['type']}")
    print(f"    score: {urgency_score}")
    print(f"    legend: {urgency_legend}")
    print(f"    probabilities: {urgency_probabilities}")
    print(f"    label (from highest probability '{winning_key}'): {urgency_label}")
    print(f"    confidence: {urgency_answer['confidence']:.4f}")
    print(f"    answer_confidence: {urgency_answer['answer_confidence']:.4f}")
    
    # Needs human (noul - yes/no)
    needs_human_answer = answers["needs_human"]
    needs_human_prob = needs_human_answer["noul"]
    needs_human_bool = needs_human_prob > 0.5
    
    print(f"\n  Needs Human:")
    print(f"    type: {needs_human_answer['type']}")
    print(f"    probability (noul): {needs_human_prob:.4f}")
    print(f"    boolean (>0.5): {needs_human_bool}")
    print(f"    confidence: {needs_human_answer['confidence']:.4f}")
    print(f"    answer_confidence: {needs_human_answer['answer_confidence']:.4f}")
    
    # Routing info
    print(f"\n  Routing:")
    print(f"    model: {result['routing']['model']}")
    print(f"    reason: {result['routing']['reason']}")
    
    # Usage
    usage = result["usage"]
    print(f"\n  Usage:")
    print(f"    input_tokens: {usage['input_tokens']}")
    print(f"    output_tokens: {usage['output_tokens']}")
    print(f"    state_tokens: {usage['state_tokens']}")
    print(f"    state_tokens_dropped: {usage['state_tokens_dropped']}")
    print(f"    truncated: {usage['truncated']}")
    
    # ===== 3. SIMPLIFIED SUMMARY =====
    print()
    print("SIMPLIFIED SUMMARY:")
    print(f"  ✓ Department: {dept_answer['choice']} (confidence: {dept_answer['answer_confidence']:.2f})")
    print(f"  ✓ Urgency: {urgency_label} (score: {urgency_score}, confidence: {urgency_answer['answer_confidence']:.2f})")
    print(f"  ✓ Needs Human: {'YES' if needs_human_bool else 'NO'} (probability: {needs_human_prob:.2f}, confidence: {needs_human_answer['answer_confidence']:.2f})")
    
    print()

print("=" * 70)
print("PHASE 1 TEST COMPLETE")
print("=" * 70)
print()
print("Next steps:")
print("1. Review the output above for correctness")
print("2. Check inference times and model download")
print("3. Verify the parsing logic (especially urgency_label indexing)")
print("4. When satisfied, approve Phase 2 integration")
