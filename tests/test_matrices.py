"""Real questions from Matrix Algebra and Calculus (24BSMA11), checked by a separate route such as NumPy."""
import numpy as np

from questions import answer_question

A = "[[8,-6,2],[-6,7,-4],[2,-4,3]]"
RANK = "[[-2,-1,-3,-1],[1,2,3,1],[1,0,1,1],[0,1,1,1]]"


def text_of(question):
    return "\n".join(answer_question(question).answer)


def test_rank_ia1_q6():
    assert text_of(f"rank {RANK}") == f"Rank = {np.linalg.matrix_rank(np.array([[-2,-1,-3,-1],[1,2,3,1],[1,0,1,1],[0,1,1,1]]))}"


def test_eigenvalues_ia1_q2():
    assert sorted(np.linalg.eigvals(np.array([[-1, 3], [-2, 4]])).round(6)) == [1, 2]
    assert text_of("eigenvalues [[-1,3],[-2,4]]") == "λ = 1, 2"


def test_eigenvectors_and_diagonalise_3_by_3():
    assert sorted(np.linalg.eigvalsh(np.array([[8, -6, 2], [-6, 7, -4], [2, -4, 3]])).round(6)) == [0, 3, 15]
    answer = text_of(f"eigenvectors {A}")
    assert "λ = 0" in answer and "λ = 3" in answer and "λ = 15" in answer
    assert "P =" in text_of(f"diagonalize {A}")


def test_power_by_diagonalisation_assignment_q10():
    matrix = np.array([[-1, 3], [-2, 4]])
    power = np.linalg.matrix_power(matrix, 4)
    answer = text_of("diagonalize [[-1,3],[-2,4]] power 4")
    assert f"{power[0, 0]}" in answer and f"{power[1, 1]}" in answer


def test_consistency_ia1_q7():
    solution = np.linalg.solve(np.array([[1, 1, 1], [1, -2, 3], [2, 1, -1]]), np.array([9, 8, 3]))
    assert solution.round(6).tolist() == [2, 3, 4]
    answer = text_of("solve x+y+z=9; x-2y+3z=8; 2x+y-z=3")
    assert "unique" in answer and "x = 2" in answer and "y = 3" in answer and "z = 4" in answer


def test_inconsistent_system_assignment_q2():
    assert "inconsistent" in text_of("solve x-4y+7z=14; 3x+8y-2z=13; 7x-8y+26z=5")


def test_homogeneous_system_with_free_unknown():
    assert "infinitely many" in text_of("solve x+2y+3z=0; 3x+4y+4z=0; 7x+10y+12z=0") or "x = 0" in text_of("solve x+2y+3z=0; 3x+4y+4z=0; 7x+10y+12z=0")


def test_four_unknowns_subscripts_assignment_q6():
    answer = text_of("solve x1-2x2+x3-x4=0; x1+x2-2x3+3x4=0; 4x1+x2-5x3+8x4=0; 5x1-7x2+2x3-x4=0")
    assert "infinitely many" in answer


def test_investigate_assignment_q7():
    answer = text_of("investigate x+y+z=6; x+2y+3z=10; x+2y+λz=μ")
    assert "Unique solution: λ ≠ 3" in answer
    assert "Infinitely many solutions: λ = 3 and μ = 10" in answer
    assert "No solution: λ = 3 and μ ≠ 10" in answer


def test_gauss_seidel_ia1_q8():
    answer = text_of("gauss seidel 10x+y+z=12; x+10y+z=12; x+y+10z=12; iterations=3")
    assert "x = 1." in answer or "x = 0.99" in answer


def test_gauss_seidel_reorders_rows_assignment_q8():
    answer = answer_question("gauss seidel x+y+54z=110; 27x+6y-z=85; 6x+15y+2z=72; iterations=4; places=3")
    exact = np.linalg.solve(np.array([[27, 6, -1], [6, 15, 2], [1, 1, 54]]), np.array([85, 72, 110]))
    values = [float(line.split("=")[1]) for line in answer.answer]
    assert np.allclose(values, exact, atol=0.01)


def test_gauss_seidel_start_end_sem_8b():
    answer = answer_question("gauss-seidel 5x+2y+z=12; x+4y+2z=15; x+2y+5z=20; iterations=4; start=(1,0,3)")
    exact = np.linalg.solve(np.array([[5, 2, 1], [1, 4, 2], [1, 2, 5]]), np.array([12, 15, 20]))
    assert np.allclose([float(line.split("=")[1]) for line in answer.answer], exact, atol=0.01)


def test_matlab_question_summary_end_sem_13b():
    answer = text_of("summary [[1,5,8],[0,4,3],[2,3,6]]")
    assert f"Determinant = {round(np.linalg.det(np.array([[1, 5, 8], [0, 4, 3], [2, 3, 6]])))}" in answer
    assert "Rank = 3" in answer and "Size = 3 × 3" in answer
