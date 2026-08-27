import importlib.util
import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


EXAMPLE_PATH = Path(__file__).with_name("05_초간단_랭그래프_분기.py")


def load_example_module():
    """파일 이름이 숫자로 시작하므로 importlib을 사용해 불러옵니다."""
    spec = importlib.util.spec_from_file_location("simple_langgraph_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChooseRouteTests(unittest.TestCase):
    def test_coding_question_goes_to_openai(self):
        """코딩 질문을 일상 질문으로 잘못 보내는 회귀를 잡습니다."""
        self.assertTrue(EXAMPLE_PATH.is_file(), "예제 파일이 아직 없습니다.")
        example = load_example_module()

        self.assertEqual(example.choose_route("파이썬 반복문 코드를 알려줘"), "coding")

    def test_daily_question_goes_to_ollama(self):
        """일상 질문을 OpenAI 경로로 잘못 보내는 회귀를 잡습니다."""
        self.assertTrue(EXAMPLE_PATH.is_file(), "예제 파일이 아직 없습니다.")
        example = load_example_module()

        self.assertEqual(example.choose_route("오늘 저녁 메뉴 추천해 줘"), "daily")


class HardcodedApiKeyTests(unittest.TestCase):
    def test_chat_can_start_without_preconfigured_environment_variable(self):
        """외부 환경변수가 없다는 이유로 채팅 시작 전에 종료되는 회귀를 잡습니다."""
        with patch.dict(os.environ, {}, clear=True):
            example = load_example_module()

            class ExitOnlyGraph:
                def invoke(self, state):
                    raise AssertionError("exit 입력 시 그래프를 실행하면 안 됩니다.")

            with (
                patch.object(example, "build_graph", return_value=ExitOnlyGraph()),
                patch("builtins.input", return_value="exit"),
                io.StringIO() as output,
                redirect_stdout(output),
            ):
                example.main()

                self.assertIn("초간단 LangGraph 분기 챗봇을 시작합니다.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
