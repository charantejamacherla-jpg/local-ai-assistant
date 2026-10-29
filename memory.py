class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        if not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")

        self.max_turns = max_turns
        self.messages = []

    def add_user_message(self, message: str):
        self._add_message("user", message)

    def add_assistant_message(self, message: str):
        self._add_message("assistant", message)

    def _add_message(self, role: str, message: str):
        if not isinstance(message, str):
            raise TypeError("message must be a string")

        message = message.strip()

        if not message:
            return

        self.messages.append(
            {
                "role": role,
                "content": message,
            }
        )

        self._trim()

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages.clear()

    def _trim(self):
        max_messages = self.max_turns * 2

        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]