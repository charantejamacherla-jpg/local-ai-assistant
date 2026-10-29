from memory import ConversationMemory
from memory_manager import MemoryManager
from memory_formatter import format_memory


def test_memory_basic():
    memory = ConversationMemory(max_turns=3)

    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi")
    memory.add_user_message("My name is Alex")
    memory.add_assistant_message("Nice to meet you")

    messages = memory.get_messages()

    assert len(messages) == 4
    assert messages[0]["content"] == "Hello"
    assert messages[-1]["content"] == "Nice to meet you"

    print("PASS: basic memory")


def test_memory_trimming():
    memory = ConversationMemory(max_turns=2)

    for i in range(1, 5):
        memory.add_user_message(f"User {i}")
        memory.add_assistant_message(f"Assistant {i}")

    messages = memory.get_messages()

    assert len(messages) == 4
    assert messages[0]["content"] == "User 3"
    assert messages[-1]["content"] == "Assistant 4"

    print("PASS: memory trimming")


def test_empty_message():
    memory = ConversationMemory()

    memory.add_user_message("   ")

    assert len(memory.get_messages()) == 0

    print("PASS: empty message protection")


def test_invalid_message():
    memory = ConversationMemory()

    try:
        memory.add_user_message(123)
    except TypeError:
        print("PASS: invalid message protection")
        return

    raise AssertionError("Invalid message was accepted")


def test_invalid_max_turns():
    try:
        ConversationMemory(max_turns=0)
    except ValueError:
        print("PASS: invalid max_turns protection")
        return

    raise AssertionError("Invalid max_turns was accepted")


def test_memory_manager():
    manager = MemoryManager(max_turns=2)

    manager.remember_user("My name is Alex")
    manager.remember_assistant("Nice to meet you, Alex")

    context = manager.get_context()

    assert len(context) == 2
    assert context[0]["role"] == "user"
    assert context[1]["role"] == "assistant"

    print("PASS: memory manager")


def test_memory_formatter():
    messages = [
        {
            "role": "user",
            "content": "My name is Alex.",
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, Alex.",
        },
    ]

    formatted = format_memory(messages)

    assert "User: My name is Alex." in formatted
    assert "Assistant: Nice to meet you, Alex." in formatted

    print("PASS: memory formatter")


def test_rag_business_hours():
    from vector_search import build_index, search

    index = build_index()

    results = search(
        "When are you open?",
        index,
        top_k=3,
        min_similarity=0.55,
    )

    assert len(results) > 0
    assert any(
        "Business Hours" in result["content"]
        for result in results
    )

    print("PASS: RAG business hours retrieval")


def test_rag_holidays():
    from vector_search import build_index, search

    index = build_index()

    results = search(
        "Are you closed on public holidays?",
        index,
        top_k=3,
        min_similarity=0.55,
    )

    assert len(results) > 0
    assert any(
        "Holiday Schedule" in result["content"]
        for result in results
    )

    print("PASS: RAG holiday retrieval")


def test_unknown_refund_policy():
    from vector_search import build_index, search

    index = build_index()

    results = search(
        "What is your refund policy?",
        index,
        top_k=3,
        min_similarity=0.55,
    )

    for result in results:
        answerable_for = result.get("answerable_for", [])

        assert "refund policy" not in answerable_for

    print("PASS: refund policy protection")


def run_tests():
    print("=" * 50)
    print("PHASE 6F TEST SUITE")
    print("=" * 50)

    test_memory_basic()
    test_memory_trimming()
    test_empty_message()
    test_invalid_message()
    test_invalid_max_turns()
    test_memory_manager()
    test_memory_formatter()

    test_rag_business_hours()
    test_rag_holidays()
    test_unknown_refund_policy()

    print("=" * 50)
    print("ALL PHASE 6F TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()