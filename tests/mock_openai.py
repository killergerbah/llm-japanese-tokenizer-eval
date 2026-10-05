"""A tiny OpenAI-compatible mock server for testing the LLM path for free.

Runs MeCab and answers both the single-sentence and batch prompts. For the batch
prompt it returns one labeled ``Sk:`` line per input. Uses UniDic's lemma when
the prompt asks for lemmas.

Usage:

    .venv/bin/python tests/mock_openai.py &
    .venv/bin/tok-eval --config config.mock.yaml all --batch-size 2
    kill %1
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

import fugashi

DELIM = "\uff0f"
SEP = "\u2192"
_BATCH_LINE = re.compile(r"^\s*S(\d+)\s*[:：]\s*(.*)$")


class Handler(BaseHTTPRequestHandler):
    tagger = fugashi.Tagger()

    def log_message(self, *args):  # silence
        pass

    def _segment(self, text: str, wants_lemma: bool) -> str:
        pieces = []
        for w in self.tagger(text):
            if wants_lemma:
                lemma = getattr(w.feature, "lemma", None) or w.surface
                pieces.append(f"{w.surface}{SEP}{lemma}")
            else:
                pieces.append(w.surface)
        return DELIM.join(pieces)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        user = body["messages"][-1]["content"]
        wants_lemma = "separator" in user

        if "Texts:" in user:
            block = user.split("Texts:", 1)[-1]
            lines = []
            for line in block.splitlines():
                m = _BATCH_LINE.match(line)
                if m:
                    lines.append(f"S{m.group(1)}: {self._segment(m.group(2), wants_lemma)}")
            out = "\n".join(lines)
        else:
            text = user.split("Text:\n", 1)[-1].strip()
            out = self._segment(text, wants_lemma)

        payload = {
            "model": "mock-mecab-v1",
            "choices": [{"message": {"role": "assistant", "content": out}}],
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8731), Handler).serve_forever()
