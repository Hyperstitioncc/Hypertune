"""Grader audit of the six SFT domains (bf, finqa, funcall, intent, parse, formula): regression tests for each fix, with
the completion that showed the problem where there was one."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.base import Problem  # noqa: E402
from envs.domains import bf, finqa, formula, funcall, intent, parse  # noqa: E402

# finqa

TABLE = [["year", "2019", "2020"], ["rev", "$1,000", "1200"]]


def _fq_prompt(q):
    return f"ctx\n\nQuestion: {q}\nWrite the calculation as a program. Operations: add, ...\nProgram:"


def _fq_env():
    return finqa.make(finqa.FinqaCfg(Path(".")))


def test_finqa_fraction_and_percent_are_one_answer_to_a_percentage_question():
    env = _fq_env()
    q = "what is the percent change in general and administrative expense from 2000 to 2001?"
    gold_pct = {"exe": -26.06635, "table": TABLE}  # gold multiplies by const_100
    gold_frac = {"exe": 0.03091, "table": TABLE}  # gold stops at the fraction
    p1, p2 = Problem("a", _fq_prompt(q), gold_pct), Problem("b", _fq_prompt(q), gold_frac)
    assert env.grade([(p1, " subtract(15.6, 21.1), divide(#0, 21.1)"),
                      (p2, " divide(98.05, 95.11), subtract(#0, 1), multiply(#1, 100)"),
                      (p2, " subtract(98.05, 95.11), divide(#0, 95.11)")]) == [1.0, 1.0, 1.0]
    # still a different number: off by 1% of the answer, or by a factor other than 100
    assert env.grade([(p1, " subtract(15.6, 21.2), divide(#0, 21.1)"),
                      (p2, " divide(98.05, 95.11), subtract(#0, 1), multiply(#1, 1000)")]) == [0.0, 0.0]


def test_finqa_percent_scaling_only_where_the_question_asks_for_a_percentage():
    env = _fq_env()
    gold = {"exe": 28597891.58, "table": TABLE}
    prog = " multiply(3227753, 8.86), divide(#0, const_100)"
    q = "if all outstanding options warrants and rights were exercised what would be the total cash inflow?"
    assert env.grade([(Problem("a", _fq_prompt(q), gold), prog)]) == [0.0]
    assert not finqa.grade(" subtract(15.6, 21.1), divide(#0, 21.1)", {"exe": -26.06635, "table": TABLE})  # no question


def test_finqa_percent_scaling_respects_the_five_decimals_of_a_small_fraction():
    q = "what is the percentage change in the price from 2016 to 2017?"
    gold = {"exe": 0.00327, "table": TABLE}  # divide(705.4, 703.1), subtract(#0, const_1)
    assert finqa.grade(" divide(705.4, 703.1), subtract(#0, const_1), multiply(#1, const_100)", gold, q)
    # wrong base (705.4 instead of 703.1): 0.326% against 0.327%, which a 1e-5 absolute slack would have passed
    assert not finqa.grade(" subtract(705.4, 703.1), divide(#0, 705.4), multiply(#1, 100)", gold, q)


def test_finqa_small_fractions_need_the_same_value_not_within_1e4_absolute():
    # seen: gold add(242, 177947), divide(242, #0) = 0.00136; the model's 260/177947 = 0.00146 used to pass
    assert not finqa.grade(" divide(260, 177947)", {"exe": 0.00136, "table": TABLE})
    assert not finqa.grade(" divide(27, 5014)", {"exe": 0.00536, "table": TABLE})  # transposed digits of 5041
    assert finqa.grade(" add(242, 177947), divide(242, #0)", {"exe": 0.00136, "table": TABLE})
    assert finqa.grade(" divide(242, 178189)", {"exe": 0.00136, "table": TABLE})
    assert finqa.grade(" divide(1, 3)", {"exe": 0.33333, "table": TABLE})
    assert finqa.grade(" add(177947, 242)", {"exe": 178189.0, "table": TABLE})
    assert not finqa.grade(" add(177947, 260)", {"exe": 178189.0, "table": TABLE})
    assert not finqa.grade(" divide(1383346, 1472684)", {"exe": 0.93939, "table": TABLE})  # 0.93934: another answer
    assert finqa.grade(" divide(1, 3)", {"exe": 0.33334, "table": TABLE})  # one unit of the fifth decimal
    assert finqa.grade(" multiply(6474102, 9.10)", {"exe": 58914328.2, "table": TABLE})


def test_finqa_yes_no_and_signs_unchanged():
    q = "what is the percent change in revenue?"
    assert finqa.grade(" greater(5, 3)", {"exe": "yes", "table": TABLE}, q)
    assert not finqa.grade(" greater(3, 5)", {"exe": "yes", "table": TABLE}, q)
    assert not finqa.grade(" subtract(31, 32)", {"exe": 1.0, "table": TABLE}, q)  # sign is part of the answer
    assert not finqa.grade(" subtract(1200, 1000), divide(#0, 1000)", {"exe": "yes", "table": TABLE}, q)


def test_finqa_question_is_read_from_the_prompt():
    assert finqa.question(_fq_prompt("what percentage of debt matured?")) == "what percentage of debt matured?"
    assert finqa.question("no question here") == ""


# funcall

def _fc(*calls):
    return "".join(f"<tool_call>\n{json.dumps({'name': n, 'arguments': a})}\n</tool_call>\n" for n, a in calls)


def _gold(*calls):
    return [list(c) for c in funcall.calls(_fc(*calls))]


def test_funcall_numbers_by_value():
    gold = _gold(("f", {"x": 100.0, "y": 3}))
    assert funcall.grade(_fc(("f", {"x": 100, "y": 3.0})), gold)
    assert not funcall.grade(_fc(("f", {"x": 100.5, "y": 3})), gold)
    assert not funcall.grade(_fc(("f", {"x": "100", "y": 3})), gold)  # a string is not a number
    assert not funcall.grade(_fc(("f", {"x": True, "y": 3})), _gold(("f", {"x": 1, "y": 3})))


def test_funcall_json_document_in_a_string_by_content():
    # seen: gold '{"interactions":[{"customer_id":"58293",...' compact, the model wrote it with ", " and ": "
    doc = {"interactions": [{"customer_id": "58293", "date": "2023-04-05"}]}
    gold = _gold(("f", {"logs": json.dumps(doc, separators=(",", ":"))}))
    assert funcall.grade(_fc(("f", {"logs": json.dumps(doc, indent=2)})), gold)
    assert not funcall.grade(_fc(("f", {"logs": doc})), gold)  # an object where a string is asked
    other = {"interactions": [{"customer_id": "58294", "date": "2023-04-05"}]}
    assert not funcall.grade(_fc(("f", {"logs": json.dumps(other)})), gold)
    assert not funcall.grade(_fc(("f", {"logs": "{not json"})), _gold(("f", {"logs": "{not  json"})))


def test_funcall_null_argument_same_as_left_out():
    # seen: {"filters": {"brand": null, ...}} against a gold without "brand"
    gold = _gold(("f", {"a": 1, "filters": {"price": 5}}))
    assert funcall.grade(_fc(("f", {"a": 1, "filters": {"price": 5, "brand": None}})), gold)
    assert funcall.grade(_fc(("f", {"a": 1, "filters": {"price": 5}, "extra": None})), gold)
    assert not funcall.grade(_fc(("f", {"a": 1, "filters": {"price": 5, "brand": ""}})), gold)
    assert not funcall.grade(_fc(("f", {"a": 1, "filters": {"price": 5, "brand": "x"}})), gold)


def test_funcall_strings_lists_and_calls_still_exact():
    gold = _gold(("f", {"s": "Entertainment", "l": ["a", "b"]}), ("g", {}))
    assert funcall.grade(_fc(("g", {}), ("f", {"l": ["a", "b"], "s": "Entertainment"})), gold)  # call order free
    assert not funcall.grade(_fc(("f", {"s": "entertainment", "l": ["a", "b"]}), ("g", {})), gold)
    assert not funcall.grade(_fc(("f", {"s": "Entertainment", "l": ["b", "a"]}), ("g", {})), gold)
    assert not funcall.grade(_fc(("f", {"s": "Entertainment", "l": ["a", "b"]})), gold)
    assert not funcall.grade(_fc(("f", {"s": "Entertainment", "l": ["a", "b"]}), ("g", {}), ("g", {})), gold)
    assert not funcall.grade("", gold) and not funcall.grade("<tool_call>{bad</tool_call>", gold)


def test_funcall_python_literal_json_cannot_hold_is_wrong_not_a_crash():
    gold = _gold(("f", {"x": [1, 2]}))
    assert funcall.calls("<tool_call>{'name': 'f', 'arguments': {'x': {1, 2}}}</tool_call>") is None
    assert not funcall.grade("<tool_call>{'name': 'f', 'arguments': {'x': {1, 2}}}</tool_call>", gold)
    assert funcall.grade("<tool_call>{'name': 'f', 'arguments': {'x': [1, 2.0]}}</tool_call>", gold)


# intent

def test_intent_label_with_question_mark():
    assert intent.grade(" reverted_card_payment?", "reverted_card_payment?")
    assert intent.grade(" reverted_card_payment", "reverted_card_payment?")
    assert intent.grade(" Refund_not_showing_up\nCustomer message: x", "Refund_not_showing_up")
    assert intent.grade(" card_arrival.", "card_arrival")
    assert not intent.grade(" card_payment_not_working", "card_payment_not_recognised")
    assert not intent.grade(" get_virtual_card", "getting_virtual_card")  # a near name is not the label
    assert not intent.grade(" card_arrival is the intent", "card_arrival")


# parse

def test_parse_tokenisation_of_the_text_does_not_matter():
    # seen: TOPv2 writes "I'd" in one gold and "I 'd" in another; "p.m" vs "p . m"; "family - oriented"
    assert parse.grade(" [IN:PLAY_MUSIC Find new [SL:MUSIC_TYPE music ] that I 'd like ]",
                       "[IN:PLAY_MUSIC Find new [SL:MUSIC_TYPE music ] that I'd like ]")
    assert parse.grade(" [IN:GET_EVENT What [SL:ATTRIBUTE_EVENT family-oriented ] events ]",
                       "[IN:GET_EVENT What [SL:ATTRIBUTE_EVENT family - oriented ] events ]")


def test_parse_structure_labels_and_text_still_exact():
    gold = "[IN:CREATE_ALARM Wake me up [SL:DATE_TIME at 7 am ] ]"
    assert parse.grade(" " + gold + "\nUtterance: next", gold)
    assert not parse.grade(" [IN:CREATE_ALARM Wake me up at [SL:DATE_TIME 7 am ] ]", gold)  # slot boundary moved
    assert not parse.grade(" [IN:CREATE_ALARM Wake me [SL:DATE_TIME up at 7 am ] ]", gold)
    assert not parse.grade(" [IN:CREATE_ALARM Wake me up [SL:TIME at 7 am ] ]", gold)
    assert not parse.grade(" [IN:CREATE_ALARM Wake me up [SL:DATE_TIME at 8 am ] ]", gold)
    assert not parse.grade(" [IN:CREATE_ALARM Wake me up [SL:DATE_TIME at 7 am ]", gold)
    assert not parse.grade(" [IN:CREATE_ALARM Wake me up [SL:DATE_TIME at 7 am] ]", gold)  # "am]" is not "am ]"
    assert not parse.grade(" [IN:CREATE_ALARM wake me up [SL:DATE_TIME at 7 am ] ]", gold)  # case kept
    assert parse.tree_key("[IN:A x y [SL:B z ] ]") == ["[IN:A", "xy", "[SL:B", "z", "]", "]"]


# formula

def test_formula_same_atoms_in_any_order():
    assert formula.grade(" C18H20N2O3", "C18H20N2O3")
    assert formula.grade(" H20C18O3N2", "C18H20N2O3")
    assert formula.grade(" C2H5OH", "C2H6O")
    assert formula.grade(" H2F2P+", "F2H2P+")
    assert formula.grade(" C18H20N2O3.", "C18H20N2O3")
    assert not formula.grade(" C18H20N2O3+", "C18H20N2O3")
    assert not formula.grade(" C25H50N2O2+", "C25H50N2O2+2")
    assert formula.grade(" C25H50N2O2+2", "C25H50N2O2+2")
    assert not formula.grade(" C18H21N2O3", "C18H20N2O3")
    assert not formula.grade(" C18H20N2O3S0", "C18H20N2O3")
    assert not formula.grade(" C18 H20 N2 O3", "C18H20N2O3")
    assert not formula.grade(" c18h20n2o3", "C18H20N2O3")
    assert not formula.grade("", "C18H20N2O3")
    assert formula.composition("C2H5OH") == ({"C": 2, "H": 6, "O": 1}, 0)
    assert formula.composition("C12H16N2O4-2") == ({"C": 12, "H": 16, "N": 2, "O": 4}, -2)
    assert formula.composition("C10H16N2O.UVWY") is None


# bf (no change; the semantics the audit checked)

def test_bf_semantics_the_audit_relied_on():
    tests = [{"input": "abc", "output": "abc"}]
    assert bf.grade(",[.,]\n```\nExplanation: `,` reads, `.` prints, [loop] <- -> +", tests)  # prose after the fence
    assert not bf.grade(",[.,]++++++++++.", tests)  # a trailing newline is output
    assert not bf.grade(",----------[++++++++++.,----------]", tests)  # waits for a newline the input never has


# every gold target of the real data grades correct (where SFT_DOMAIN_DATA holds it)

DATA = Path(os.environ.get("SFT_DOMAIN_DATA", "sft_data"))  # the SFT domain data (README: reproducing the plots)
MODS = {"bf": (bf, bf.BfCfg), "finqa": (finqa, finqa.FinqaCfg), "funcall": (funcall, funcall.FuncallCfg),
        "intent": (intent, intent.IntentCfg), "parse": (parse, parse.ParseCfg), "formula": (formula, formula.FormulaCfg)}


@pytest.mark.parametrize("d", sorted(MODS))
def test_every_gold_target_passes_its_grader(d):
    if not (DATA / d / "test.jsonl").exists():
        pytest.skip(f"no data at {DATA / d}")
    m, cfg = MODS[d]
    env = m.make(cfg(DATA / d))
    for split in ("train", "dev", "test"):
        rows = env.rows(split)
        got = env.grade([(Problem(r["id"], r["prompt"], r["gold"]), r["target"]) for r in rows])
        assert [r["id"] for r, g in zip(rows, got) if g != 1.0] == [], split
