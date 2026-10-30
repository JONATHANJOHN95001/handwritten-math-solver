"""Score the project on real questions from the three maths courses, in the wording of the papers.

BASELINE is the 49 questions tried before the project was taught these courses, when it solved 0 of them as written.
UNSEEN holds questions from previous-year papers, kept unread until the methods and the English reader were finished, so
it measures questions the project was not taught on. Each answer is checked against one worked out separately.
Writes reports/course_check.json.

Usage: python check_courses.py
"""
import json
import pathlib
import sys

from course_questions import BASELINE, UNSEEN
from questions import answer_question

HERE = pathlib.Path(__file__).parent


def score(questions, name):
    rows = []
    for number, (course, source, question, expected) in enumerate(questions, 1):
        try:
            result = answer_question(question)
            answer = "\n".join(result.answer)
            kind = "right" if all(part in answer for part in expected) else "wrong"
            detail = answer[:300]
        except Exception as error:
            kind, detail = "refused", f"{type(error).__name__}: {error}"[:300]
        rows.append({"course": course, "source": source, "question": question[:140], "expected": expected, "outcome": kind, "detail": detail})
        print(f"{name} {number}/{len(questions)} {kind} {source}", flush=True)
    return rows


def summary(rows):
    table = {}
    for row in rows:
        counts = table.setdefault(row["course"], {"tested": 0, "right": 0, "wrong": 0, "refused": 0})
        counts["tested"] += 1
        counts[row["outcome"]] += 1
    return table


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    report = {}
    for name, questions in (("baseline", BASELINE), ("unseen", UNSEEN)):
        rows = score(questions, name)
        report[name] = {"summary": summary(rows), "rows": rows}
        print(f"\n{name}: " + json.dumps(report[name]["summary"]))
        for row in rows:
            if row["outcome"] != "right":
                print(f"  {row['outcome'].upper()} [{row['course']}] {row['source']}: {row['question'][:90]}\n     wanted {row['expected']}, got: {row['detail'][:200]}")
        (HERE / "reports" / "course_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")  # kept if a later set is stopped


if __name__ == "__main__":
    main()
