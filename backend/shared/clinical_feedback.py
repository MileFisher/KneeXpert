"""Predefined KL grade clinical feedback (0–4) for X-ray ensemble output."""

FINDINGS_BY_GRADE: dict[int, list[str]] = {
    0: ["No radiographic features of osteoarthritis.", "Joint space preserved."],
    1: ["Doubtful joint space narrowing.", "Possible early osteophytic change."],
    2: [
        "Definite osteophytes identified.",
        "Possible joint space narrowing in medial compartment.",
        "Mild subchondral changes.",
    ],
    3: [
        "Joint space narrowing (medial compartment).",
        "Osteophyte formation (tibial plateau).",
        "Subchondral sclerosis detected.",
    ],
    4: [
        "Marked joint space narrowing.",
        "Large osteophytes and subchondral sclerosis.",
        "Definite bone contour deformity.",
    ],
}


def findings_for_grade(grade: int) -> list[str]:
    g = max(0, min(4, int(grade)))
    return list(FINDINGS_BY_GRADE.get(g, FINDINGS_BY_GRADE[0]))
