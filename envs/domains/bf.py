from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from envs.domains import BuildCfg, DomainEnv, FILES, RawCfg, Rows, main

CMDS = "><+-.,[]"
TAPE, MAX_STEPS, MAX_OUT = 30_000, 1_000_000, 100_000
PROMPT = ("Problem: {problem}\nWrite the solution as a Brainfuck program in a single ```brainfuck code block.\n"
          "Solution:\n```brainfuck\n")


class BFError(Exception):
    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


def compile_bf(code):
    src = "".join(c for c in code if c in CMDS)
    ops, stack, i = [], [], 0
    while i < len(src):
        c = src[i]
        if c in "+-":
            n = 0
            while i < len(src) and src[i] in "+-":
                n += 1 if src[i] == "+" else -1
                i += 1
            if n % 256:
                ops.append(["add", n % 256])
            continue
        if c in "<>":
            n = 0
            while i < len(src) and src[i] in "<>":
                n += 1 if src[i] == ">" else -1
                i += 1
            if n:
                ops.append(["mov", n])
            continue
        if src[i:i + 3] in ("[-]", "[+]"):
            ops.append(["clr", 0])
            i += 3
            continue
        if c == "[":
            stack.append(len(ops))
            ops.append(["jz", -1])
        elif c == "]":
            if not stack:
                raise BFError("compile_error")
            j = stack.pop()
            ops.append(["jnz", j])
            ops[j][1] = len(ops) - 1
        else:
            ops.append(["out" if c == "." else "in", 0])
        i += 1
    if stack:
        raise BFError("compile_error")
    return ops


def run(code, inp=""):
    ops = compile_bf(code)
    data = inp.encode("latin-1", "replace")
    tape, out = bytearray(TAPE), bytearray()
    p = pc = ip = steps = 0
    n = len(ops)
    while pc < n:
        op, a = ops[pc]
        if op == "add":
            tape[p] = (tape[p] + a) & 255
        elif op == "mov":
            p += a
            if not 0 <= p < TAPE:
                raise BFError("runtime_error")
        elif op == "jz":
            if not tape[p]:
                pc = a
        elif op == "jnz":
            if tape[p]:
                pc = a
        elif op == "clr":
            tape[p] = 0
        elif op == "out":
            out.append(tape[p])
            if len(out) > MAX_OUT:
                raise BFError("runtime_error")
        else:
            tape[p] = data[ip] if ip < len(data) else 0
            ip += ip < len(data)
        pc += 1
        steps += 1
        if steps > MAX_STEPS:
            raise BFError("timeout")
    return bytes(out)


def extract(text):
    t = text
    if "```" in t:
        head, _, rest = t.partition("```")
        if any(c in CMDS for c in head):
            t = head
        else:
            t = rest.split("\n", 1)[-1].partition("```")[0] if "\n" in rest else rest.partition("```")[0]
    return "".join(c for c in t if c in CMDS)


def grade(text, tests):
    prog = extract(text)
    if not prog:
        return False
    try:
        return all(run(prog, t["input"]) == t["output"].encode("latin-1", "replace") for t in tests)
    except BFError:
        return False


class BfAssembler:
    def __init__(self):
        self.c, self.p = [], 0

    def to(self, i):
        d = i - self.p
        self.c.append((">" if d > 0 else "<") * abs(d))
        self.p = i

    def raw(self, s):
        self.c.append(s)

    def add(self, i, n, tmp):
        n %= 256
        if n > 128:
            n -= 256
        a, sign = abs(n), "+" if n > 0 else "-"
        if a < 14:
            self.to(i)
            self.raw(sign * a)
            return
        x = min(range(2, 17), key=lambda x: x + a // x + a % x)
        self.to(tmp); self.raw("+" * x + "[")
        self.to(i); self.raw(sign * (a // x))
        self.to(tmp); self.raw("-]")
        self.to(i); self.raw(sign * (a % x))

    def out(self, i):
        self.to(i); self.raw(".")

    def inp(self, i):
        self.to(i); self.raw(",")

    def text(self, s, cell, tmp):
        cur = 0
        for ch in s:
            self.add(cell, ord(ch) - cur, tmp)
            self.out(cell)
            cur = ord(ch)
        self.to(cell); self.raw("[-]")

    def code(self):
        s = "".join(self.c)
        while "<>" in s or "><" in s:
            s = s.replace("<>", "").replace("><", "")
        return s


WORDS = ("amber basil cedar delta ember fjord grove harbor iris jade kelp lumen maple nectar onyx "
         "pearl quartz river slate tundra umber violet willow xenon yarrow zephyr").split()
ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _phrase(r):
    s = " ".join(r.choice(WORDS) for _ in range(r.randint(1, 3)))
    return r.choice([s, s.capitalize(), s.title(), s + r.choice("!?."), s.upper()])


def _line(r, lo=1, hi=16, chars=ALNUM + "  "):
    return "".join(r.choice(chars) for _ in range(r.randint(lo, hi))).strip() or r.choice(ALNUM)


def f_print(r):
    s = _phrase(r)
    a = BfAssembler(); a.text(s, 0, 1)
    return (f"No input is provided. Write exactly the string {s!r} to stdout, with nothing before or after it.",
            a.code(), lambda x: s, [""] * 6)


def f_shift(r):
    k = r.choice([1, 2, 3, 4, 5]) * r.choice([1, -1])
    how = f"increased by {k}" if k > 0 else f"decreased by {-k}"
    return (f"Read a line of text and output it with the character code of every character {how}.",
            ",[" + ("+" if k > 0 else "-") * abs(k) + ".,]", lambda x: "".join(chr(ord(c) + k) for c in x),
            [_line(r, chars=ALNUM) for _ in range(6)])


def f_double(r):
    n = r.choice([2, 3])
    return (f"Read a line of text and output it with every character written {n} times in a row.",
            ",[" + "." * n + ",]", lambda x: "".join(c * n for c in x), [_line(r) for _ in range(6)])


def f_last(r):
    return ("Read a non-empty line of text and output only its last character.",
            ",[>,]<.", lambda x: x[-1], [_line(r) for _ in range(6)])


def f_wrap(r):
    pre, post = _phrase(r) + " ", r.choice(["!", ".", " :)", " <<", ""])
    a = BfAssembler(); a.text(pre, 0, 1); a.to(2); a.raw(",[.,]"); a.text(post, 3, 4)
    return (f"Read a line of text and output it with {pre!r} written before it and {post!r} written after it.",
            a.code(), lambda x: pre + x + post, [_line(r) for _ in range(6)])


def f_repeat(r):
    a = BfAssembler(); a.inp(0); a.add(0, -48, 1); a.inp(1); a.to(0); a.raw("["); a.to(1); a.raw("."); a.to(0); a.raw("-]")
    return ("The input is a digit d from 0 to 9 followed immediately by one character c. Output c exactly d times.",
            a.code(), lambda x: x[1] * int(x[0]), [str(r.randint(0, 9)) + r.choice(ALNUM) for _ in range(6)])


def f_adddigits(r):
    a = BfAssembler(); a.inp(0); a.inp(1); a.add(1, -48, 2); a.to(1); a.raw("["); a.to(0); a.raw("+"); a.to(1); a.raw("-]"); a.out(0)
    def t():
        x = r.randint(0, 9)
        return f"{x}{r.randint(0, 9 - x)}"
    return ("The input is two digits with no separator, whose sum is at most 9. Output their sum as one digit.",
            a.code(), lambda x: str(int(x[0]) + int(x[1])), [t() for _ in range(6)])


def f_tally(r):
    ch = r.choice("1*#x|")
    a = BfAssembler(); a.add(1, ord(ch), 2); a.to(0); a.raw(",["); a.to(1); a.raw("."); a.to(0); a.raw(",]")
    return (f"Read a line of text and output the character {ch!r} once for every character of the input.",
            a.code(), lambda x: ch * len(x), [_line(r) for _ in range(6)])


def f_echo(r):
    return ("Read a line of text and echo it back exactly as received.", ",[.,]", lambda x: x,
            [_line(r) for _ in range(6)])


def f_reverse(r):
    return ("Read a line of text and output its characters in reverse order.", ">,[>,]<[.<]", lambda x: x[::-1],
            [_line(r) for _ in range(6)])


def f_even(r):
    return ("Read a line of text and output only the characters at even indices (0, 2, 4, ...).", ",[.,,]",
            lambda x: x[::2], [_line(r, chars=ALNUM) for _ in range(6)])


LOWER, UPPER, DIG = "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789"


def _word(r, chars, lo=2, hi=12):
    return "".join(r.choice(chars) for _ in range(r.randint(lo, hi)))


def _words(r):
    return " ".join(r.choice(WORDS) for _ in range(r.randint(2, 3)))


def f_first(r):
    return ("Read a non-empty line of text and output only its first character.", ",.", lambda x: x[0],
            [_word(r, ALNUM, 1) for _ in range(6)])


def f_second(r):
    return ("Read a line of at least two characters and output only its second character.", ",,.", lambda x: x[1],
            [_word(r, ALNUM, 2) for _ in range(6)])


def f_dropfirst(r):
    return ("Read a non-empty line of text and output it without its first character.", ",,[.,]", lambda x: x[1:],
            [_word(r, ALNUM, 1) for _ in range(6)])


def f_firstn(r):
    k = r.choice([2, 3, 4])
    return (f"Read a line of at least {k} characters and output only its first {k} characters.", ",." * k,
            lambda x: x[:k], [_word(r, ALNUM, k) for _ in range(6)])


def f_skipn(r):
    k = r.choice([2, 3])
    return (f"Read a line of at least {k} characters and output it without its first {k} characters.",
            "," * k + ",[.,]", lambda x: x[k:], [_word(r, ALNUM, k) for _ in range(6)])


def f_lasttwo(r):
    return ("Read a line of at least two characters and output only its last two characters.", ",[>,]<<.>.",
            lambda x: x[-2:], [_word(r, ALNUM, 2) for _ in range(6)])


def f_swap2(r):
    return ("The input is exactly two characters. Output them in the opposite order.", ",>,.<.",
            lambda x: x[1] + x[0], [_word(r, ALNUM, 2, 2) for _ in range(6)])


def f_upper(r):
    a = BfAssembler(); a.raw(",["); a.add(0, -32, 1); a.raw(".,]")
    return ("Read a line of lowercase letters and output it in uppercase.", a.code(), lambda x: x.upper(),
            [_word(r, LOWER) for _ in range(6)])


def f_lower(r):
    a = BfAssembler(); a.raw(",["); a.add(0, 32, 1); a.raw(".,]")
    return ("Read a line of uppercase letters and output it in lowercase.", a.code(), lambda x: x.lower(),
            [_word(r, UPPER) for _ in range(6)])


def f_countdigit(r):
    a = BfAssembler(); a.raw(",[>+<,]"); a.add(1, 48, 2); a.out(1)
    return ("Read a line of at most 9 characters and output how many characters it has, as one digit.", a.code(),
            lambda x: str(len(x)), [_word(r, ALNUM, 0, 9) for _ in range(6)])


def f_subdigits(r):
    a = BfAssembler(); a.inp(0); a.inp(1); a.add(1, -48, 2); a.to(1); a.raw("[<->-]"); a.out(0)
    def t():
        x = r.randint(0, 9)
        return f"{x}{r.randint(0, x)}"
    return ("The input is two digits with no separator, the first at least as large as the second. Output the first "
            "minus the second as one digit.", a.code(), lambda x: str(int(x[0]) - int(x[1])), [t() for _ in range(6)])


def f_doubledigit(r):
    a = BfAssembler(); a.inp(0); a.raw("[>+>+<<-]"); a.add(2, -48, 3); a.to(2); a.raw("[<+>-]"); a.out(1)
    return ("The input is one digit from 0 to 4. Output twice its value as one digit.", a.code(),
            lambda x: str(2 * int(x)), [str(r.randint(0, 4)) for _ in range(6)])


def f_muldigit(r):
    k = r.choice([2, 3, 4])
    a = BfAssembler(); a.inp(0); a.add(0, -48, 1); a.to(0); a.raw("[>" + "+" * k + "<-]"); a.add(1, 48, 2); a.out(1)
    return (f"The input is one digit from 0 to {9 // k}. Output {k} times its value as one digit.", a.code(),
            lambda x: str(k * int(x)), [str(r.randint(0, 9 // k)) for _ in range(6)])


def f_interleave(r):
    c = r.choice("-_|*:~")
    a = BfAssembler(); a.add(1, ord(c), 2); a.to(0); a.raw(",[.>.<,]")
    return (f"Read a line of text and output it with the character {c!r} written after every character.", a.code(),
            lambda x: "".join(ch + c for ch in x), [_word(r, ALNUM, 1) for _ in range(6)])


def f_countdown(r):
    a = BfAssembler(); a.inp(0); a.add(0, -48, 1); a.to(0); a.raw("["); a.add(0, 48, 1); a.out(0); a.add(0, -48, 1)
    a.to(0); a.raw("-]")
    return ("The input is one digit d. Output the digits d, d-1, ... down to 1, with nothing between them "
            "(nothing at all for 0).", a.code(), lambda x: "".join(str(i) for i in range(int(x), 0, -1)),
            [str(r.randint(0, 9)) for _ in range(6)])


def f_printn(r):
    w = r.choice(WORDS)[:5]
    a = BfAssembler(); a.inp(0); a.add(0, -48, 1); a.to(0); a.raw("["); a.text(w, 1, 2); a.to(0); a.raw("-]")
    return (f"The input is one digit d from 0 to 5. Output the word {w!r} exactly d times, with nothing between "
            "the copies.", a.code(), lambda x: w * int(x), [str(r.randint(0, 5)) for _ in range(6)])


def f_parity(r):
    a = BfAssembler(); a.raw(",[>[>+<-]+>[<->-]<<,]"); a.add(1, 48, 2); a.out(1)
    return ("Read a line of text and output 1 if it has an odd number of characters and 0 if the number is even.",
            a.code(), lambda x: str(len(x) % 2), [_word(r, ALNUM, 0, 14) for _ in range(6)])


def f_firstword(r):
    a = BfAssembler(); a.inp(0); a.add(0, -32, 1); a.to(0); a.raw("["); a.add(0, 32, 1); a.out(0); a.inp(0)
    a.add(0, -32, 1); a.to(0); a.raw("]")
    return ("The input is two or more words separated by single spaces. Output only the first word.", a.code(),
            lambda x: x.split(" ")[0], [_words(r) for _ in range(6)])


def f_afterspace(r):
    a = BfAssembler(); a.inp(0); a.add(0, -32, 1); a.to(0); a.raw("["); a.inp(0); a.add(0, -32, 1); a.to(0); a.raw("]")
    a.raw(",[.,]")
    return ("The input is two or more words separated by single spaces. Output everything after the first space.",
            a.code(), lambda x: x.split(" ", 1)[1], [_words(r) for _ in range(6)])


def f_nextchars(r):
    k = r.choice([2, 3, 4])
    return (f"The input is one character. Output it followed by the next {k - 1} characters in code order "
            f"({k} characters in all).", ",." + "+." * (k - 1), lambda x: "".join(chr(ord(x) + i) for i in range(k)),
            [r.choice("abcdefghijklmnopqrstuABCDEFGHIJKLMNOPQRSTU012345") for _ in range(6)])


def f_alphabet(r):
    a = BfAssembler(); a.inp(0); a.add(0, -48, 1); a.add(1, 96, 2); a.to(0); a.raw("[>+.<-]")
    return ("The input is one digit d. Output the first d lowercase letters of the alphabet (nothing for 0).",
            a.code(), lambda x: LOWER[:int(x)], [str(r.randint(0, 9)) for _ in range(6)])


def f_mirror(r):
    return ("Read a line of text and output it followed by the same text reversed.", ">,[.>,]<[.<]",
            lambda x: x + x[::-1], [_word(r, ALNUM, 1) for _ in range(6)])


def f_prefixrev(r):
    pre = _phrase(r) + " "
    a = BfAssembler(); a.text(pre, 0, 1); a.to(3); a.raw(",[>,]<[.<]")
    return (f"Read a line of text and output {pre!r} followed by the input reversed.", a.code(),
            lambda x: pre + x[::-1], [_word(r, ALNUM, 1) for _ in range(6)])


FAMILIES = {"print": (f_print, False), "shift": (f_shift, False), "double": (f_double, False),
            "last": (f_last, False), "wrap": (f_wrap, False), "repeat": (f_repeat, False),
            "adddigits": (f_adddigits, False), "tally": (f_tally, False),
            "echo": (f_echo, True), "reverse": (f_reverse, True), "even": (f_even, True),
            "first": (f_first, False), "second": (f_second, False), "dropfirst": (f_dropfirst, False),
            "firstn": (f_firstn, False), "skipn": (f_skipn, False), "lasttwo": (f_lasttwo, False),
            "swap2": (f_swap2, False), "upper": (f_upper, True), "lower": (f_lower, False),
            "countdigit": (f_countdigit, False), "subdigits": (f_subdigits, False),
            "doubledigit": (f_doubledigit, False), "muldigit": (f_muldigit, False),
            "interleave": (f_interleave, False), "countdown": (f_countdown, False), "printn": (f_printn, False),
            "parity": (f_parity, False), "firstword": (f_firstword, False), "afterspace": (f_afterspace, False),
            "nextchars": (f_nextchars, False), "alphabet": (f_alphabet, False), "mirror": (f_mirror, False),
            "prefixrev": (f_prefixrev, False)}


def make_problem(family, r):
    problem, gold, ref, inputs = FAMILIES[family][0](r)
    tests = [{"input": x, "output": ref(x)} for x in inputs]
    if not grade(gold, tests):
        return None
    return {"family": family, "problem": problem, "prompt": PROMPT.format(problem=problem), "tests": tests,
            "gold": gold, "target": gold + "\n```"}


def gen(n, seed=0, bench=False):
    r = random.Random(seed)
    fams = [f for f, (_, b) in FAMILIES.items() if bench or not b]
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 50 * n:
        it = make_problem(fams[tries % len(fams)], r)
        tries += 1
        key = it and (it["problem"], it["gold"], json.dumps(it["tests"]))
        if it is None or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


@dataclass(frozen=True)
class BfCfg:
    bf_data: Path


def make(cfg: BfCfg) -> DomainEnv:
    return DomainEnv("bf", cfg.bf_data, grade, (*FILES, "all", "esolang"))


def build(cfg: BfCfg, b: BuildCfg, r: random.Random, raw: RawCfg) -> tuple[dict[str, Rows], str]:
    items = gen(b.n_train + b.n_dev + b.n_test, seed=r.randrange(10**6))
    rows = [{"id": f"bf-{i}", "prompt": it["prompt"], "target": " " + it["target"], "gold": it["tests"],
             "family": it["family"]} for i, it in enumerate(items)]
    r.shuffle(rows)
    splits = {"train": rows[:b.n_train], "dev": rows[b.n_train:b.n_train + b.n_dev], "test": rows[b.n_train + b.n_dev:]}
    with open(raw.raw / "esolang" / "test.jsonl") as f:
        splits["esolang"] = [{"id": f"esolang-{x['id']}", "prompt": PROMPT.format(problem=x["description"]),
                              "target": "", "gold": x["test_cases"], "family": x["difficulty"]}
                             for x in map(json.loads, f)]
    return (splits, "bf_tasks.gen (verified gold programs; families that restate benchmark problems left out); "
            "extra split esolang = Lossfunk/Esolang-Bench")


if __name__ == "__main__":
    main(BfCfg, make, build, RawCfg)
