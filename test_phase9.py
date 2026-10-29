
import unittest
from unittest.mock import patch

from agent import run_agent
from app import route_decision, apply_department_override


class TestLayaPhase9(unittest.TestCase):

    def test_calculator(self):
        result = run_agent("127 * 84")
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        self.assertEqual(str(result["observation"]), "10668")

    def test_empty_input(self):
        self.assertIsNone(run_agent(""))

    def test_invalid_input(self):
        self.assertIsNone(run_agent(None))

    def test_no_tool_selected(self):
        with patch("agent.select_tool", return_value=None):
            self.assertIsNone(run_agent("Hello Laya"))

    def test_human_escalation(self):
        classification = {
            "department": "billing",
            "needs_human": 0.85,
        }
        decision = route_decision(classification)
        self.assertEqual(decision["action"], "escalate_human")

    def test_normal_chat_routing(self):
        classification = {
            "department": "technical",
            "needs_human": 0.10,
        }
        decision = route_decision(classification)
        self.assertEqual(decision["action"], "chat")

    def test_department_override(self):
        classification = {"department": "technical"}
        result = apply_department_override(
            "What are your business hours?",
            classification,
        )
        self.assertEqual(result["department"], "other")


if __name__ == "__main__":
    unittest.main()
