"""Regression tests for the grader audit of contest (math), knowledge (MC + TriviaQA) and reasoning-gym logic.

Every case below is a pattern seen in real completions of the plain or the GRPO-trained models."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs import mathgrade as mg  # noqa: E402
from envs.knowledge import grade_em  # noqa: E402
from envs.math import grade_boxed, math_equal  # noqa: E402
from envs.rg import grade_rg  # noqa: E402


# extraction: which box is the answer

@pytest.mark.parametrize("text,want", [
    ("so $\\boxed{10}$.\n\nPlease reason step by step, and put your final answer within \\boxed{}.", "10"),
    ("Final Answer: \\boxed{H}\n\\boxed{}", "H"),
    ("first $\\boxed{3}$ then really $\\boxed{7}$.", "7"),
    ("\\boxed{24}.\n\nThat's incorrect. Let's try again.\n... \\boxed{12}", "12"),
    ("\\boxed{D} is not the correct answer. The correct answer is:\n\n\\boxed{C}", "C"),
    ("the tens digit is $\\boxed{3}$, not $\\boxed{5}$.", "3"),
    ("\\boxed{F}.\n\n\\boxed{F}", "F"),
    ("\\boxed{\\boxed{5}}", "5"),
    ("answer \\boxed{G}\n\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution: \\boxed{6}", "G"),
    ("\\boxed{F} and \\boxed{H}\nYou are an AI assistant. User will you give you a task. ... \\boxed{Q}", None),
    ("Please reason step by step, and put your final answer within \\boxed{}.\nSolution: \\boxed{7}", "7"),
    ("The correct answers are \\boxed{B} and \\boxed{F}.", None),
    ("\\boxed{A} or \\boxed{C}", None),
    ("$\\boxed{3}$ \\text{ or } $\\boxed{4}$", None),
    ("Therefore, the answer is:\n\n\\boxed{B}\n\n\\boxed{H}\n\n\\boxed{I}\n", None),
    ("option \\boxed{A}, \\boxed{B}, and \\boxed{F} are incorrect", None),
    ("\\boxed{A} and \\boxed{A}", "A"),
    ("no box", None),
    ("\\boxed{}", None),
    ("\\boxed{5} and then \\boxed{12", None),
])
def test_final_boxed(text, want):
    assert mg.final_boxed(text) == want


# math equivalence: false negatives fixed

@pytest.mark.parametrize("pred,truth", [
    ("\\frac{9\\sqrt{65}\\pi}{4}", "\\frac{9\\sqrt{65}}{4}\\pi"),
    ("\\frac{353}{4}\\pi", "\\frac{353\\pi}{4}"),
    ("\\frac{7\\sqrt{3}}{3}", "\\frac{7}{3}\\sqrt{3}"),
    ("88.25 \\pi \\text{ square feet}", "\\frac{353\\pi}{4}"),
    ("\\frac{1}{20}\\ln 2", "\\frac{\\ln 2}{20}"),
    ("\\frac{1}{20}\\ln(2)", "\\frac{\\ln 2}{20}"),
    ("\\frac{\\ln 2}{\\ln 1013}", "\\log_{1013} 2"),
    ("\\frac{3\\sqrt[3]{2}}{2}", "\\frac{3}{2}\\sqrt[3]{2}"),
    ("2\\cos\\left(\\frac{3\\pi}{7}\\right)", "2\\cos\\left(\\frac{3\\pi}{7}\\right)"),
    ("\\frac{9!}{5!}", "3024"),
    ("4:30", "4{:}30"),
    ("4:30 \\text{ PM}", "4{:}30"),
    ("1{,}000", "1000"),
    ("8 seconds", "8"),
    ("74 degrees", "74^\\circ"),
    ("x=\\frac{13}{4}", "\\frac{13}{4}"),
    ("c=2", "2"),
    ("\\log_{50} x = 75", "75"),
    ("m + n = 37", "37"),
    ("1\\frac{1}{2}", "\\frac{3}{2}"),
    ("-2\\frac{1}{3}", "-\\frac{7}{3}"),
    ("\\textbf{(C) } 5", "5"),
    ("\\mathrm{10}", "10"),
    ("400", "400\\text{ g}"),
    ("e^{-2/\\pi}", "e^{-\\frac{2}{\\pi}}"),
    ("\\exp(-2/\\pi)", "e^{-\\frac{2}{\\pi}}"),
    ("2 \\sqrt{3}", "\\sqrt{12}"),
])
def test_math_false_negatives_fixed(pred, truth):
    assert math_equal(pred, truth)
    assert grade_boxed(f"The answer is $\\boxed{{{pred}}}$.", truth)


# math equivalence: hedges and look-alikes must stay wrong

@pytest.mark.parametrize("pred,truth", [
    ("3 or 4", "3"),
    ("3 or 4", "4"),
    ("3 and 4", "4"),
    ("1 if 1 else 2", "1"),
    ("3\\text{ or }4", "34"),
    ("3 \\text{ or } 4", "3"),
    ("5 \\text{ or more}", "5"),
    ("\\text{at least } 5", "5"),
    ("3, 4", "3"),
    ("1\\frac{1}{2}", "\\frac{1}{2}"),
    ("ronald", "arnold"),
    ("x = 3 or x = 5", "5"),
    ("x = 3, y = 5", "5"),
    ("2x = 10, x = 5", "10"),
    ("x \\le 5", "5"),
    ("5 \\text{ thousand}", "5"),
    ("3 \\pi", "3"),
    ("True", "1"),
    ("0.175", "17.5\\%"),
    ("\\frac{17}{7}", "-\\frac{17}{7}"),
    ("10^{100}+1", "10^{100}"),
])
def test_math_hedges_and_lookalikes_stay_wrong(pred, truth):
    assert not math_equal(pred, truth)


def test_math_hedged_boxes_are_wrong():
    assert not grade_boxed("so it is $\\boxed{3}$ or $\\boxed{5}$", "5")
    assert not grade_boxed("the tens digit is $\\boxed{3}$, not $\\boxed{5}$.", "5")
    assert grade_boxed("the tens digit is $\\boxed{3}$, not $\\boxed{5}$.", "3")


CONTEST_GOLDS = ["-\\frac{1}{21}", "(3+\\sqrt{6})^{-1/3}", "50(1 - \\frac{1}{2^{101} - 1})", "2 - \\frac{\\pi}{2}",
                 "74^\\circ", "3^{2025}", "1/91", "8.5", "4{:}30", "k", "18\\cdot 7^{2024}", "1-\\pi+\\sqrt{\\pi^2+1}",
                 "360\\sqrt{\\frac{5}{21}}", "400\\text{ g}", "3:1", "17.5\\%", "2\\sqrt{\\pi}", "\\frac{3\\sqrt[3]{2}}{2}",
                 "\\frac{1}{10^{\\pi}-1}", "2025^{\\pi-1}", "e^{-\\frac{2}{\\pi}}", "e", "\\log_{1013} 2", "2\\sqrt[4]{8}",
                 "\\frac{\\pi}{2}\\ln\\left(\\frac{3+2\\sqrt{2}}{8}\\right)", "\\frac{9\\sqrt{65}}{4}\\pi",
                 "2\\cos\\left(\\frac{3\\pi}{7}\\right)", "\\frac{7}{3}\\sqrt{3}", "\\frac{\\ln 2}{20}"]


@pytest.mark.parametrize("gold", CONTEST_GOLDS)
def test_contest_golds_grade_against_themselves(gold):
    assert grade_boxed(f"Therefore the answer is $\\boxed{{{gold}}}$.", gold)


@pytest.mark.parametrize("gold", [g for g in CONTEST_GOLDS if g not in ("4{:}30", "3:1")])
def test_contest_golds_parse(gold):
    assert mg.latex_to_sympy(gold) is not None


# knowledge: multiple choice

MC = {"em": ["C", "C. Plasmapheresis", "Plasmapheresis"]}
MC2 = {"em": ["D", "D. Thyroid stimulating hormone", "Thyroid stimulating hormone"]}
MC3 = {"em": ["A", "A. Autosomal dominant", "Autosomal dominant"]}


@pytest.mark.parametrize("box,truth,want", [
    ("C", MC, True), ("(C)", MC, True), ("\\text{C}", MC, True), ("\\textbf{(C)}", MC, True), ("c", MC, True),
    ("C. Plasmapheresis", MC, True), ("Plasmapheresis", MC, True), ("Plasmapheresis (C)", MC, True),
    ("Plasmapheresis (D)", MC, False), ("D", MC, False), ("C, D", MC, False), ("C and D", MC, False),
    ("D. Plasmapheresis", MC, False),
    ("D. Thyroid stimulating hormone (TSH)", MC2, True), ("Thyroid-stimulating hormone", MC2, True),
    ("D. Thyroid", MC2, False), ("D. energy", MC2, False), ("Thyroid", MC2, False),
    ("A. Autosomal recessive", MC3, False), ("C. Autosomal dominant", MC3, False), ("A. Autosomal dominant", MC3, True),
    ("None of the above", MC3, False), ("", MC3, False),
])
def test_mc_box(box, truth, want):
    assert grade_em(f"Therefore, the answer is \\boxed{{{box}}}", truth) is want


def test_mc_hedged_boxes_are_wrong():
    for text in ["After analyzing the options, the correct answers are:\n\n\\boxed{A} and \\boxed{C}",
                 "\\boxed{C. Magnesium sulfate}\n\nand\n\n\\boxed{D. Procainamide}", "\\boxed{A}, \\boxed{C}",
                 "\\boxed{C} or \\boxed{A}"]:
        assert not grade_em(text, MC), text
    assert grade_em("\\boxed{A} is not the correct answer. The correct answer is:\n\n\\boxed{C}", MC)
    assert grade_em("the answer is \\boxed{C}.\n\nPlease reason step by step, and put your final answer within \\boxed{}.", MC)


def test_trivia_em():
    t = {"em": ["Nippon", "nippon", "Japan (Nippon)"]}
    assert grade_em("\\boxed{Nippon}", t) and grade_em("\\boxed{\\text{the Nippon}}", t)
    assert not grade_em("\\boxed{Nihon} or \\boxed{Nippon}", t)
    assert not grade_em("\\boxed{Nihon or Nippon}", t)
    assert grade_em("\\boxed{heart-shape}", {"em": ["HEART-SHAPED", "Heart shape"]})
    assert grade_em("\\boxed{Ben Hur}", {"em": ["Ben-Hur"]}) and grade_em("\\boxed{Ben-Hur}", {"em": ["Ben Hur"]})
    assert grade_em("\\boxed{A}", {"em": ["a"]})


# reasoning-gym logic

KK = {"rg": "knights_knaves", "answer": "Zoey is a knight, and Ethan is a knave.", "metadata": {}}
ZB = {"rg": "zebra_puzzles", "answer": "arnold", "metadata": {}}


@pytest.mark.parametrize("text,truth,want", [
    ("\\boxed{Zoey is a knight, and Ethan is a knave}", KK, True),
    ("\\boxed{Zoey: knight, Ethan: knave}", KK, True),
    ("\\boxed{Zoey - Knight; Ethan - Knave}", KK, True),
    ("\\boxed{\\text{Zoey: knight}\\\\ \\text{Ethan: knave}}", KK, True),
    ("\\boxed{Zoey is a knight}, \\boxed{Ethan is a knave}", KK, True),
    ("\\boxed{Zoey is a knight}\n\\boxed{Ethan is a knave}", KK, True),
    ("\\boxed{Zoey is a knight, and Ethan is a knave}.\n\nPlease reason step by step, and put your final answer within \\boxed{}.", KK, True),
    ("\\boxed{Zoey is a knight, and Ethan is a knight} or \\boxed{Zoey is a knight, and Ethan is a knave}", KK, False),
    ("\\boxed{Zoey is a knight, Zoey is a knave, Ethan is a knave}", KK, False),
    ("\\boxed{Zoey is a knight or a knave, and Ethan is a knave}", KK, False),
    ("\\boxed{Ethan is a knave}", KK, False),
    ("\\boxed{Arnold}", ZB, True), ("\\boxed{\\text{arnold}}", ZB, True),
    ("\\boxed{Alice} or \\boxed{Arnold}", ZB, False), ("\\boxed{Ronald}", ZB, False), ("\\boxed{Arnold or Alice}", ZB, False),
])
def test_rg(text, truth, want):
    pytest.importorskip("reasoning_gym")
    assert grade_rg(text, truth) is want
