# /// script
# requires-python = ">=3.10"
# dependencies = ["tokenizers>=0.21", "jinja2>=3.1"]
# ///
"""Golden outputs from the Hugging Face reference implementations, for the Julia tests.

Usage (uv creates an isolated environment with the dependencies above):
    uv run scripts/gen_fixtures.py tokenizer <model_dir> <name>
    uv run scripts/gen_fixtures.py chat <model_dir> <name>

`tokenizer` writes test/fixtures/tokenizer/<name>.json: token ids for TEXTS, encoded with
and without special tokens. `chat` writes test/fixtures/chat/<name>.json: CONVERSATIONS
rendered with the model's own Jinja chat template, as transformers renders it, or the
template's error for a conversation it rejects.
"""

import json
import sys
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

FIXTURES = Path(__file__).resolve().parent.parent / "test" / "fixtures"

# Chosen to hit every stage of a byte-level BPE pipeline: contractions and casing in the
# split regex, digit grouping, whitespace runs, NFC normalization, multi-byte UTF-8, and
# added tokens written inline (Qwen's and Apertus', including the strings Apertus 1.5's
# normalizer rewrites and its normalized tokens).
TEXTS = [
    "",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say it's FINE, but WE'VE heard THAT'S not true.",
    "Numbers: 1 12 123 1234 12345 3.14159 -42 1e10 2026-10-26",
    "  leading spaces, trailing spaces   ",
    "tabs\tand\t\tnewlines\n\nand\r\nwindows line ends\n",
    "multiple    spaces     between  words",
    "def f(x):\n    return x ** 2  # square\n\nprint(f(3))\n",
    "julia> A = [1 2; 3 4]; A' * A\n",
    "Café, naïve, Æsir, façade — “quotes” and ‘single’ …",
    "é vs é (decomposed vs precomposed)",
    "日本語のテキストと中文文本",
    "Emoji: 👍🏽 🇫🇷 👨‍👩‍👧 🤗",
    "Ελληνικά και русский текст",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nreasoning\n</think>\n\nanswer<|endoftext|>",
    "a<|im_end|>b<|im_start|><|im_end|>",
    "!!!??? ... ,,, ;;; ::: ---",
    "https://huggingface.co/Qwen/Qwen3-0.6B?x=1&y=2#frag",
    "x" * 300,
    "<s><|system_start|>Be brief.<|system_end|><|user_start|>Hi<|user_end|><|assistant_start|>",
    "<|inner_prefix|>let me think<|inner_suffix|>   The answer is 4.<|assistant_end|>",
    "<think>plan</think> done <thought>more</thought><answer>42</answer>",
    "<|channel>thought\n  idea<channel|>after <image> and <audio> and <|image|>",
    "[INST] old style [/INST] <unk> <pad> </s>",
]


CONVERSATIONS = [
    [{"role": "user", "content": "Hi!"}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "2+2?"}],
    [
        {"role": "user", "content": "Pick a number."},
        {"role": "assistant", "content": "<think>\nAny will do.\n</think>\n\n7"},
        {"role": "user", "content": "Another one."},
    ],
    [
        {"role": "system", "content": "Be nice."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "<think>short</think>Goodbye!"},
    ],
    [{"role": "user", "content": "a"}, {"role": "system", "content": "late system"},
     {"role": "user", "content": "b"}],
    [
        {"role": "user", "content": "Why?"},
        {"role": "assistant", "content": "<|inner_prefix|>Because.<|inner_suffix|>It is so."},
        {"role": "assistant", "content": " More."},
        {"role": "user", "content": "Thanks"},
    ],
]


def chat_fixture(model_dir: str, name: str) -> None:
    config = json.loads((Path(model_dir) / "tokenizer_config.json").read_text())
    # Newer repos ship the template as its own file.
    jinja = Path(model_dir) / "chat_template.jinja"
    source = jinja.read_text() if jinja.exists() else config["chat_template"]
    bos = config.get("bos_token")
    bos = bos.get("content") if isinstance(bos, dict) else bos

    def raise_exception(message):
        raise ValueError(message)

    # Same environment settings as transformers' chat template rendering.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)

    cases = []
    for messages in CONVERSATIONS:
        for add_generation_prompt in (True, False):
            for enable_thinking in (True, False):
                case = {
                    "messages": messages,
                    "add_generation_prompt": add_generation_prompt,
                    "enable_thinking": enable_thinking,
                }
                try:
                    case["prompt"] = template.render(messages=messages, bos_token=bos,
                                                     add_generation_prompt=add_generation_prompt,
                                                     enable_thinking=enable_thinking)
                except ValueError as e:      # the template rejects the conversation
                    case["error"] = str(e)
                cases.append(case)
    write_fixture("chat", name, cases)


def write_fixture(kind: str, name: str, cases: list) -> None:
    out = FIXTURES / kind / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {out}")


def tokenizer_fixture(model_dir: str, name: str) -> None:
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    cases = []
    for text in TEXTS:
        cases.append({
            "text": text,
            "ids": tok.encode(text, add_special_tokens=True).ids,
            "ids_no_special": tok.encode(text, add_special_tokens=False).ids,
        })
    write_fixture("tokenizer", name, cases)


if __name__ == "__main__":
    commands = {"tokenizer": tokenizer_fixture, "chat": chat_fixture}
    if len(sys.argv) != 4 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](sys.argv[2], sys.argv[3])
