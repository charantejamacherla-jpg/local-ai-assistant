from memory import ConversationMemory


class MemoryManager:
    def __init__(self, max_turns: int = 10):
        self.memory = ConversationMemory(
            max_turns=max_turns
        )

    def remember_user(self, message: str):
        self.memory.add_user_message(message)

    def remember_assistant(self, message: str):
        self.memory.add_assistant_message(message)

    def get_context(self) -> list[dict]:
        return self.memory.get_messages()

    def clear(self):
        self.memory.clear()