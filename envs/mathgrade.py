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


def norm_answer(a):
    a = str(a).strip().strip("$").strip()
    for junk in (*SPACING, " ", "\\$"):
        a = a.replace(junk, "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return a.rstrip(".").replace("^{\\circ}", "").replace("^\\circ", "")


def grade_str(pred, truth):
    return pred is not None and norm_answer(pred) == norm_answer(truth)


def _take_group(s, i):
    j, _ = _close(s, i + 1)
    return s[i + 1:j - 1], j


def _cmd_to_py(s, cmd, fmt, nargs):
    out, i = [], 0
    tok = "\\" + cmd
    while i < len(s):
        if s.startswith(tok, i):
            j = i + len(tok)
            args = []
            ok = True
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
                out.append(fmt.format(*args)); i = j; continue
        out.append(s[i]); i += 1
    return "".join(out)


def latex_to_sympy(a):
    if a is None:
        return None
    s = str(a).strip().strip("$").strip()
    if s == "" or "\\begin" in s or "\\text{" in s and len(s) > 40:
        return None
    for junk in (*SPACING, "\\$", "$", "^\\circ", "^{\\circ}", "\\%", "%", "\\ ", "~"):
        s = s.replace(junk, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\!", "").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\div", "/").replace("\\pi", "pi")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mbox\{[^}]*\}", "", s)
    s = s.replace("^{\\prime}", "").replace("\\prime", "")
    s = re.sub(r"(?<=\d),(?=\d\d\d(\D|$))", "", s)
    for _ in range(6):
        ns = _cmd_to_py(s, "frac", "(({0})/({1}))", 2)
        ns = _cmd_to_py(ns, "sqrt", "sqrt({0})", 1)
        if ns == s:
            break
        s = ns
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^(\w)", r"**\1", s)
    s = s.replace("{", "(").replace("}", ")")
    if "\\" in s:
        return None
    s = re.sub(r"\)\s*\(", r")*(", s)
    s = re.sub(r"(?<=\d)(?=[a-zA-Z(])", "*", s)
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,^ =<>\[\]|]*", s):
        return None
    try:
        e = sympy.sympify(s, rational=True)
    except Exception:
        return None
    return e


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
