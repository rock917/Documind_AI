import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from documind.general.qa import groq_chat

MSG = [{"role": "user", "content": "hi"}]


class Fake(BaseHTTPRequestHandler):
    status = 200
    reply = {"choices": [{"message": {"content": "ok [1]"}, "finish_reason": "stop"}]}
    seen = {}

    def do_POST(self):
        Fake.seen = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps(Fake.reply).encode()
        self.send_response(Fake.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def url():
    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Fake.status = 200
    Fake.reply = {"choices": [{"message": {"content": "ok [1]"}, "finish_reason": "stop"}]}
    yield f"http://127.0.0.1:{server.server_port}/openai/v1"
    server.shutdown()


def test_reasoning_effort_and_budget_are_sent_only_when_set(url):
    groq_chat(MSG, "m", url, "k", max_tokens=1500, reasoning_effort="low")
    assert Fake.seen["reasoning_effort"] == "low" and Fake.seen["max_tokens"] == 1500
    groq_chat(MSG, "m", url, "k")
    assert "reasoning_effort" not in Fake.seen


def test_empty_answer_has_a_plain_message(url):
    Fake.reply = {"choices": [{"message": {"content": None, "reasoning": "thinking"}, "finish_reason": "length"}]}
    with pytest.raises(ValueError, match="answer budget"):
        groq_chat(MSG, "m", url, "k")
    Fake.reply = {"choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]}
    with pytest.raises(ValueError, match="empty answer"):
        groq_chat(MSG, "m", url, "k")


def test_a_400_shows_groqs_own_message_but_never_the_key(url):
    Fake.status = 400
    Fake.reply = {"error": {"message": "reasoning_effort is not supported with this model; key=secret-key-9"}}
    with pytest.raises(ValueError) as info:
        groq_chat(MSG, "m", url, "secret-key-9")
    text = str(info.value)
    assert "status 400" in text and "reasoning_effort is not supported" in text and "secret-key-9" not in text
