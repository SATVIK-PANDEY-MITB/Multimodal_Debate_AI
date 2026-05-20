import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main
import utils


class RegressionTests(unittest.TestCase):
    def test_main_module_imports(self):
        self.assertTrue(callable(main.run_system))

    @patch("main.retrieve_context", return_value="retrieved context")
    @patch("main.answer_agent", side_effect=["logical answer", "creative answer"])
    @patch(
        "main.critic_agent",
        side_effect=[
            {"action_items": []},
            {"action_items": []},
            {"action_items": []},
            {"action_items": []},
            {"action_items": []},
            {"action_items": []},
        ],
    )
    @patch(
        "main.refiner_agent",
        side_effect=[
            "logical refined 1",
            "creative refined 1",
            "logical refined 2",
            "creative refined 2",
            "logical refined 3",
            "creative refined 3",
        ],
    )
    @patch("main.score_answers", return_value={"Answer1": {"C": 10, "CL": 9, "CF": 8}, "Answer2": {"C": 9, "CL": 8, "CF": 7}})
    @patch("main.judge_agent", side_effect=["Answer1", "Answer1", "Answer1", "Answer1"])
    def test_full_debate_run_produces_final_winner(
        self,
        _judge,
        _score,
        _refiner,
        _critic,
        _answer,
        _retrieve,
    ):
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            main.run_system("What is recursion?")

        output = output_buffer.getvalue()
        self.assertIn("=== ROUND 1 ===", output)
        self.assertIn("=== ROUND 2 ===", output)
        self.assertIn("=== ROUND 3 ===", output)
        self.assertIn("=== FINAL WINNER ===", output)
        self.assertIn("Answer1", output)

    def test_call_llm_handles_unavailable_backend(self):
        with patch(
            "utils.requests.post",
            side_effect=utils.requests.RequestException("backend unavailable"),
        ):
            result = utils.call_llm("test prompt", timeout=1)

        self.assertTrue(result.startswith("LLM_ERROR: request_failed:"))


if __name__ == "__main__":
    unittest.main()
