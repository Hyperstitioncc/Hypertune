from shared.decoder import SpelledTokenizer


class Tok:
    eos_token_id = 0

    def get_added_vocab(self):
        return {"<tool_call>": 900, "</tool_call>": 901, "<|endoftext|>": 0}

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for name, k in self.get_added_vocab().items():
                if text.startswith(name, i):
                    ids.append(k)
                    i += len(name)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids):
        back = {v: k for k, v in self.get_added_vocab().items()}
        return "".join(back.get(i) or chr(i) for i in ids)


def test_reserved_strings_are_spelled_in_ordinary_tokens():
    raw, tok = Tok(), SpelledTokenizer(Tok())
    text = 'call:\n<tool_call>{"name": "f"}</tool_call> done'
    assert 900 in raw.encode(text) and 901 in raw.encode(text)
    ids = tok.encode(text)
    assert all(i < 900 for i in ids) and tok.decode(ids) == text
    assert tok.encode("no reserved text") == raw.encode("no reserved text")
    assert tok.eos_token_id == 0
