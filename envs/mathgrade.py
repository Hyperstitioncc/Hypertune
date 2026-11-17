import re
import sympy

BOXED = "\\boxed{"
SPACING = ("\\left", "\\right", "\\!", "\\,", "\\;")


def _close(s, j):
    depth = 1
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return j, depth


def last_boxed(s):
    i = s.rfind(BOXED)
    if i < 0:
        return None
    j, depth = _close(s, i + len(BOXED))
    return s[i + len(BOXED):j - 1] if depth == 0 else None


def cut_at_first_boxed(text: str) -> str | None:
    b = text.find(BOXED)
    if b < 0:
        return None
    i, depth = _close(text, b + len(BOXED))
    if depth:
        return None
    while i < len(text) and text[i] in "$.":
        i += 1
    return text[:i]


# The graded answer of a completion: its last non-empty \boxed{...}. An empty \boxed{} carries no answer (it is
# usually the instruction echoed back after the real answer), and a box left unclosed at the end (cut off mid-answer)
# means no answer. Boxes chained to the last one by connective text only ("\boxed{A} and \boxed{C}",
# "\boxed{3}, \boxed{4}", boxes on consecutive lines) form the final answer together: two different answers there are
# a hedge.
_BOX = re.compile(r"\\boxed\s*\{")
TEXT_CMD = re.compile(r"\\(?:text|textbf|textrm|textit|mathrm|mathbf|mbox)\s*\{([^{}]*)\}")
CONNECTIVES = frozenset({"", "and", "or", "andor", "nor"})


def _boxes(s):
    out, i = [], 0
    while True:
        m = _BOX.search(s, i)
        if not m:
            return out, False
        j, depth = _close(s, m.end())
        if depth:
            return out, True
        out.append((m.start(), j, s[m.end():j - 1]))
        i = j


def _gap(gap):
    g = TEXT_CMD.sub(r" \1 ", gap)
    g = re.sub(r"\\[a-zA-Z]+|\\.", " ", g)
    return re.sub(r"[\s$.,;:&/()\[\]*]", "", g).lower()


def _connective(gap):
    return _gap(gap) in CONNECTIVES


def _innermost(c):
    while True:
        bs, _ = _boxes(c)
        bs = [b for b in bs if b[2].strip()]
        if not bs:
            return c
        c = bs[-1][2]


# A base model that has answered often runs on into a new turn: a hallucinated system prompt or the boxed-answer
# instruction echoed for a fresh problem, whose own boxes are not its answer. Text from the first such marker that
# follows a non-empty box is not graded.
NEW_TURN = re.compile(r"You are an AI assistant|(?:put|write) (?:your|the) final answer (?:with)?in \\boxed\{\}", re.I)


def answer_region(s):
    for m in NEW_TURN.finditer(s):
        bs, _ = _boxes(s[:m.start()])
        if any(b[2].strip() for b in bs):
            return s[:m.start()]
    return s


def boxed_run(s):
    """The final answer's boxes: the last non-empty box and the boxes chained to it by connective text, or None."""
    s = answer_region(s)
    bs, unclosed = _boxes(s)
    if unclosed:
        return None
    bs = [b for b in bs if b[2].strip()]
    if not bs:
        return None
    while len(bs) > 1 and _gap(s[bs[-2][1]:bs[-1][0]]) in ("not", "butnot", "andnot"):
        bs.pop()  # "\boxed{3}, not \boxed{5}": the last box is the rejected one
    run = [bs[-1]]
    for b in reversed(bs[:-1]):
        if not _connective(s[b[1]:run[0][0]]):
            break
        run.insert(0, b)
    return [_innermost(c) for _, _, c in run]


def answer_key(a):
    return norm_answer(a).lower()


def final_boxed(s):
    """The last non-empty boxed answer, or None when there is none or the final answer hedges between boxes."""
    run = boxed_run(s)
    if run is None or len({answer_key(c) for c in run}) > 1:
        return None
    return run[-1]


def norm_answer(a):
    a = str(a).strip().strip("$").strip()
    a = a.replace("{,}", ",").replace("{:}", ":")
    for junk in (*SPACING, " ", "\\$"):
        a = a.replace(junk, "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return a.rstrip(".").replace("^{\\circ}", "").replace("^\\circ", "")


UNIT_TAIL = re.compile(r"\s*\\(?:text|mathrm|mbox)\s*\{\s*([A-Za-z][A-Za-z. ]*)\}\s*\.?\s*$")
QUALIFIERS = {"or", "and", "not", "more", "less", "than", "least", "most", "about", "approximately", "over", "under"}


def drop_unit(a):
    """'4:30 \\text{ PM}' -> '4:30', '400\\text{ g}' -> '400'; a qualifier ('\\text{ or more}') is kept."""
    a = str(a)
    m = UNIT_TAIL.search(a)
    if m and a[:m.start()].strip() and not set(m.group(1).lower().replace(".", " ").split()) & (QUALIFIERS | MULT_WORDS):
        return a[:m.start()]
    return a


def grade_str(pred, truth):
    if pred is None:
        return False
    return norm_answer(pred) == norm_answer(truth) or norm_answer(drop_unit(pred)) == norm_answer(drop_unit(truth))


def _take_group(s, i):
    j, _ = _close(s, i + 1)
    return s[i + 1:j - 1], j


def _cmd_to_py(s, cmd, fmt, nargs):
    out, i = [], 0
    tok = "\\" + cmd
    while i < len(s):
        if s.startswith(tok, i) and not (i + len(tok) < len(s) and s[i + len(tok)].isalpha()):
            j = i + len(tok)
            args = []
            ok = True
            root = None
            if cmd == "sqrt" and j < len(s) and s[j] == "[":
                k = s.find("]", j)
                if k < 0:
                    ok = False
                else:
                    root, j = s[j + 1:k], k + 1
            for _ in range(nargs):
                while j < len(s) and s[j] == " ":
                    j += 1
                if j < len(s) and s[j] == "{":
                    a, j = _take_group(s, j)
                    args.append(a)
                elif j < len(s):
                    args.append(s[j]); j += 1
                else:
                    ok = False; break
            if ok:
                out.append(f"(({args[0]})**(1/({root})))" if root else fmt.format(*args)); i = j; continue
        out.append(s[i]); i += 1
    return "".join(out)


FUNCS = {"sqrt", "log", "sin", "cos", "tan", "exp", "factorial", "binomial"}
NAMES = {"pi", "E", "I", "oo"}
MULT_WORDS = {"thousand", "million", "billion", "trillion", "hundred", "percent", "squared", "cubed", "and", "or", "nor",
              "not", "but", "plus", "minus", "times", "over", "per"}
_TOKEN = re.compile(r"\d+\.?\d*|\.\d+|[A-Za-z_][A-Za-z_0-9]*|\*\*|\S")


def _texts(s):
    """\\text-like groups: a number inside is kept; other text (units, labels, "(C)") is dropped
    at either end of the answer and makes the answer unparseable in the middle ("3\\text{ or }4" is not 34)."""
    parts = TEXT_CMD.split(s)
    if len(parts) == 1:
        return s
    out = []
    for k, p in enumerate(parts):
        if k % 2 == 0:
            out.append(p)
            continue
        c = p.strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", c):
            out.append(c)
        elif "".join(parts[:k]).strip() and "".join(parts[k + 1:]).strip() or \
                set(re.findall(r"[a-z]+", c.lower())) & (QUALIFIERS | MULT_WORDS):
            return None
        else:
            out.append(" ")
    return "".join(out)


def _implicit(s):
    """Tokenize and insert the multiplications LaTeX leaves implicit. A multi-letter word that is not a known function or
    constant makes the answer unparseable: no Python keyword ("3 or 4", "1 if 1 else 2") reaches sympify, and words
    are not turned into products of letters (anagrams would compare equal)."""
    toks = _TOKEN.findall(s)
    if any(t[0].isalpha() and len(t) > 1 and t not in FUNCS and t not in NAMES for t in toks):
        return None
    out, k = [], 0
    atom_end = lambda t: t[0].isalnum() or t[0] == "." or t == ")"
    while k < len(toks):
        t = toks[k]
        if out and atom_end(out[-1]) and (t[0].isalnum() or t[0] == "." or t == "("):
            prev = out[-1]
            if (prev[0].isdigit() or prev[0] == ".") and (t[0].isdigit() or t[0] == "."):
                return None  # "3 4": two numbers side by side
            if prev not in FUNCS:
                out.append("*")
        if t in FUNCS and k + 1 < len(toks) and toks[k + 1] != "(":
            nxt = toks[k + 1]
            if not (nxt[0].isalnum() or nxt[0] == "."):
                return None
            out += [t, "(", nxt, ")"]
            k += 2
            continue
        out.append(t)
        k += 1
    return " ".join(out)


def latex_to_sympy(a):
    if a is None:
        return None
    s = str(a).strip().strip("$").strip()
    if s == "" or "\\begin" in s or "\\text{" in s and len(s) > 40:
        return None
    s = s.replace("{,}", ",").replace("{:}", ":")
    s = _texts(s)
    if s is None:
        return None
    for junk in (*SPACING, "\\$", "$", "^\\circ", "^{\\circ}", "\\%", "%", "\\ ", "~"):
        s = s.replace(junk, "")
    s = s.replace("\\qquad", " ").replace("\\quad", " ")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\div", "/").replace("\\pi", " pi ").replace("\\infty", " oo ")
    s = s.replace("^{\\prime}", "").replace("\\prime", "")
    s = re.sub(r"(?<=\d),(?=\d\d\d(\D|$))", "", s)
    m = re.fullmatch(r"\s*(-?[\d.]+)\s+((?:[A-Za-z]{3,}\.?\s*)+)", s)  # "8 seconds", "12 inches"
    if m and not set(m.group(2).lower().replace(".", " ").split()) & (MULT_WORDS | FUNCS | NAMES):
        s = m.group(1)
    # mixed numbers: 1\frac{1}{2} is 3/2, not 1/2
    s = re.sub(r"(?<![\w.^_}])(\d+)\s*\\frac\s*\{(\d+)\}\s*\{(\d+)\}",
               lambda m: f"({m[1]}+{m[2]}/{m[3]})" if int(m[2]) < int(m[3]) else m[0], s)
    s = re.sub(r"\\log_\{([^{}]+)\}\s*(\{[^{}]*\}|\([^()]*\)|\w+)", r"(\\ln{\2}/\\ln{\1})", s)
    s = re.sub(r"\\log_(\w)\s*(\{[^{}]*\}|\([^()]*\)|\w+)", r"(\\ln{\2}/\\ln{\1})", s)
    for _ in range(6):
        ns = _cmd_to_py(s, "frac", "(({0})/({1}))", 2)
        ns = _cmd_to_py(ns, "sqrt", "sqrt({0})", 1)
        ns = _cmd_to_py(ns, "binom", "binomial(({0}),({1}))", 2)
        if ns == s:
            break
        s = ns
    for cmd, py in (("ln", "log"), ("log", "log"), ("sin", "sin"), ("cos", "cos"), ("tan", "tan"), ("exp", "exp")):
        s = re.sub(r"\\" + cmd + r"(?![A-Za-z])", f" {py} ", s)
    s = re.sub(r"(\d+)!", r"factorial(\1)", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^(\w)", r"**\1", s)
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    if "\\" in s:
        return None
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,^ =<>\[\]|]*", s):
        return None
    s = _implicit(s)
    if s is None:
        return None
    s = re.sub(r"(?<![A-Za-z_])e(?![A-Za-z_])", "E", s)  # Euler's number on both sides
    try:
        e = sympy.sympify(s, rational=True)
    except Exception:
        return None
    return e


def strip_lhs(a):
    """"x = 5", "m+n=37", "\\log_{50} x = 75" -> the right-hand side; None unless it is one plain equation."""
    a = str(a)
    if a.count("=") != 1 or re.search(r"[<>]|\\[lg]eq?\b|\\neq|\\approx|\\text|\\mbox|,|;|\b(or|and)\b", a):
        return None
    lhs, rhs = a.split("=")
    if not re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", "", lhs)) or not rhs.strip():
        return None
    return rhs.strip()


def grade_sympy(pred, truth):
    a, b = latex_to_sympy(pred), latex_to_sympy(truth)
    if a is None or b is None:
        return None
    try:
        if a == b:
            return True
        d = sympy.simplify(a - b)
        if d == 0:
            return True
        v = complex(sympy.N(d))
        return abs(v) < 1e-9
    except Exception:
        try:
            return abs(complex(sympy.N(a)) - complex(sympy.N(b))) < 1e-9
        except Exception:
            return None
