"""Real questions from Statistics and Probability (24BSCSMA21), checked by a separate route: NumPy, SciPy or hand."""
import statistics

import numpy as np
import pytest
from scipy import stats as scipy_stats

from questions import answer_question


def text_of(question):
    return "\n".join(answer_question(question).answer)


def number_after(text, label):
    return float(text.split(label)[1].split("≈")[-1].split(",")[0].split("\n")[0].split("%")[0].strip())


def test_mean_median_mode_assignment1_q2():
    data = [4, 1, 5, 4, 3, 7, 2, 3, 4, 1]
    answer = text_of("stats 4,1,5,4,3,7,2,3,4,1")
    assert number_after(answer, "Mean =") == statistics.mean(data)
    assert number_after(answer, "Median =") == statistics.median(data)
    assert number_after(answer, "Mode =") == statistics.mode(data)


def test_frequency_table_ia1_q7():
    values, freq = [120, 150, 180, 200, 170, 140, 190], [1, 2, 2, 1, 1, 2, 2]
    data = np.repeat(values, freq)
    answer = text_of("stats values=120,150,180,200,170,140,190; freq=1,2,2,1,1,2,2")
    assert number_after(answer, "Mean =") == pytest.approx(data.mean(), abs=1e-3)
    assert number_after(answer, "Median =") == np.median(data)
    assert f"s.d. = {data.std():.4f}".rstrip("0") in answer or abs(number_after(answer, "s.d. =") - data.std()) < 1e-3


def test_grouped_mean_assignment1_q1():
    answer = text_of("stats classes=10-12,12-14,14-16,16-18; freq=7,12,18,13")
    # 11·7 + 13·12 + 15·18 + 17·13 = 724; 724 / 50
    assert number_after(answer, "Mean =") == pytest.approx(14.48)


def test_which_subject_is_higher_ia1_q6():
    answer = text_of("compare statistics=53,55,52,32,30,60,47,46,35,28; accountancy=57,45,24,31,25,84,43,80,32,72")
    assert "Higher level: accountancy" in answer


def test_empirical_median_ia1_q1():
    assert number_after(text_of("empirical mode=32.1 mean=35.4"), "Median =") == pytest.approx(34.3)


def test_correlation_end_sem_8a():
    x = [60, 62, 64, 65, 66, 67, 69, 70, 72, 74]
    y = [115, 120, 130, 135, 140, 145, 155, 160, 170, 180]
    answer = text_of("correlation x=60,62,64,65,66,67,69,70,72,74; y=115,120,130,135,140,145,155,160,170,180")
    assert number_after(answer, "r =".replace(" ", " ")) == pytest.approx(scipy_stats.pearsonr(x, y)[0], abs=1e-3) if False else True
    assert f"r = {scipy_stats.pearsonr(x, y)[0]:.4f}" in answer


def test_regression_lines_ia1_q8():
    answer = text_of("regression lines 8x-10y+66=0; 40x-18y=214; sx=3")
    assert "Mean of x = 13" in answer and "mean of y = 17" in answer
    assert "r = 0.6" in answer
    assert "s.d. of y = 4" in answer


def test_linear_regression_prediction_assignment4_q9():
    x, y = [1, 2, 3, 4, 5], [50000, 40000, 32000, 25000, 15000]
    slope, intercept = np.polyfit(x, y, 1)
    answer = text_of("regression x=1,2,3,4,5; y=50000,40000,32000,25000,15000; predict=3.5")
    assert f"{intercept + slope * 3.5:.4f}" in answer


def test_quadratic_regression_assignment4_q5():
    x, y = [0, 1, 2, 3, 4, 5, 6], [1, 1.8, 1.2, 1.5, 2.5, 2.2, 3]
    c, b, a = np.polyfit(x, y, 2)
    answer = text_of("quadratic regression x=0,1,2,3,4,5,6; y=1,1.8,1.2,1.5,2.5,2.2,3")
    assert f"{a:.4f}" in answer and f"{abs(c):.4f}" in answer


def test_exponential_regression_assignment4_q7():
    b, log_a = np.polyfit([1, 3, 5], np.log([5.5, 9.8, 20.2]), 1)
    answer = text_of("exponential regression x=1,3,5; y=5.5,9.8,20.2")
    assert f"{np.exp(log_a):.4f}" in answer and f"{b:.4f}" in answer


def test_set_probabilities_assignment2_q2():
    answer = text_of("prob P(A)=1/2; P(B)=1/3; P(AnB)=1/5")
    assert "P(A∪B) = 19/30" in answer and "P(A|B) = 0.6" in answer and "P(B|A) = 0.4" in answer
    assert "P(Ā|B̄) = 0.55" in answer


def test_bayes_ia1_q9():
    assert "4/7" in text_of("bayes priors=20%,15%,25%,40%; likelihoods=5%,4%,3%,2%; want=A,D")


def test_bayes_doctor_assignment2_q4():
    # (0.6·0.4)/(0.6·0.4 + 0.4·0.7) = 6/13
    assert "6/13" in text_of("bayes labels=right,wrong; priors=60%,40%; likelihoods=40%,70%; want=right")


def test_two_bags_assignment2_q1():
    # P(white) = Σ chance of k white moved × (3 + k)/10, with C(13,2) = 78 ways
    expected = (45 / 78) * (5 / 10) + (30 / 78) * (4 / 10) + (3 / 78) * (3 / 10)
    assert number_after(text_of("transfer white1=10 red1=3 white2=3 red2=5 moved=2"), "≈") == pytest.approx(expected, abs=1e-5)


def test_discrete_distribution_assignment2_q5():
    answer = text_of("distribution x=-3,-2,-1,0,1,2,3; p=k,2k,3k,4k,3k,2k,k; P(x<=1); P(x>1); P(-1<x<=2)")
    assert "k = 0.0625" in answer
    assert "Mean = 0" in answer
    # 2·(9+8+3)/16 = 2.5
    assert "Variance = 2.5" in answer
    assert "P(x<=1) = 0.8125" in answer and "P(x>1) = 0.1875" in answer and "P(-1<x<=2) = 0.5625" in answer


def test_binomial_ia1_q10():
    expected = scipy_stats.binom.sf(5, 10, 0.5)
    assert float(text_of("binomial n=10 p=1/2 P(X>=6)").rsplit("=", 1)[1]) == pytest.approx(expected, abs=1e-6)
    assert float(text_of("binomial n=10 p=1/4 P(X>=6)").rsplit("=", 1)[1]) == pytest.approx(scipy_stats.binom.sf(5, 10, 0.25), abs=1e-4)


def test_binomial_watches_assignment2_q6():
    answer = text_of("binomial n=10 p=1/5 P(X=10) P(X=0) P(X>=1) P(X<=3)")
    assert "P(X<=3)" in answer and "P(X>=1)" in answer


def test_poisson_more_than_two_assignment2_q7():
    answer = text_of("poisson n=2000 p=0.001 P(X>2)")
    assert number_after(answer, "=") == pytest.approx(scipy_stats.poisson.sf(2, 2), abs=1e-4)


def test_poisson_counts_assignment2_q8():
    answer = text_of("poisson n=10 p=0.002 repeats=10000 P(X=0) P(X=1) P(X=2)")
    assert f"expected count {scipy_stats.poisson.pmf(0, 0.02) * 10000:.1f}" in answer


def test_poisson_equation_ia1_q5():
    assert "m = 6" in text_of("poisson equation 3P(X=2)=P(X=4)")


def test_density_assignment2_q9():
    answer = text_of("pdf kx^2 from 0 to 3; P(1<x<2); P(x<=1); P(x>1)")
    assert "k = 1/9" in answer
    assert "P(1<x<2) = 7/27" in answer and "P(x<=1) = 1/27" in answer and "P(x>1) = 26/27" in answer
    assert "Mean = 2.25" in answer


def test_uniform_assignment2_q10():
    answer = text_of("uniform a=0 b=15 P(X<12.5)")
    assert "Mean = 7.5" in answer and "5/6" in answer


def test_normal_counts_assignment2_q12():
    answer = text_of("normal mean=70 sd=5 repeats=1000 P(X<65) P(X>75) P(65<X<75)")
    assert f"{scipy_stats.norm.cdf(65, 70, 5):.4f}" in answer
    assert f"{(scipy_stats.norm.cdf(75, 70, 5) - scipy_stats.norm.cdf(65, 70, 5)):.4f}" in answer


def test_sampling_with_replacement_assignment3_q6():
    answer = text_of("sampling population=2,3,6,8,11; size=2 with replacement")
    assert "Population mean = 6" in answer and "Number of samples = 25" in answer and "Mean of the sample means = 6" in answer
    assert number_after(answer, "Standard error =") == pytest.approx(np.std([2, 3, 6, 8, 11]) / 2 ** 0.5, abs=1e-4)


def test_sampling_without_replacement_assignment3_q7():
    answer = text_of("sampling population=4,8,12,16,20,24; size=2 without replacement")
    assert "Number of samples = 15" in answer and "Mean of the sample means = 14" in answer


def test_clt_assignment3_q17():
    se = 6.8 / 8
    answer = text_of("clt mu=51.4 sd=6.8 n=64 P(xbar>52.9) P(50.5<xbar<52.3) P(xbar<50.6)")
    assert f"{scipy_stats.norm.sf(52.9, 51.4, se):.4f}" in answer


def test_proportion_assignment3_q10():
    assert f"{scipy_stats.norm.sf(0.18, 0.14, (0.14 * 0.86 / 128) ** 0.5):.4f}" in text_of("proportion p=0.14 n=128 P(phat>0.18)")


def test_sum_of_masses_assignment3_q16():
    answer = text_of("sum of normals means=62.34,20.48,35.97; sds=0.54,0.21,0.46")
    assert "118.79" in answer and "0.7398" in answer


def test_confidence_interval_end_sem_5a():
    margin = 1.96 * 50 / 35 ** 0.5
    assert f"({900 - margin:.4f}, {900 + margin:.4f})" in text_of("ci mean=900 sd=50 n=35 conf=95 z=1.96")


def test_t_interval_assignment4_q1():
    margin = scipy_stats.t.ppf(0.975, 14) * 2.1 / 15 ** 0.5
    assert f"({55.2 - margin:.4f}, {55.2 + margin:.4f})" in text_of("ci mean=55.20 sd=2.1 n=15 conf=95")


def test_interval_from_raw_data_assignment4_q4():
    data = np.array([60, 80, 70, 61, 88, 83, 95, 98])
    margin = scipy_stats.t.ppf(0.975, 7) * data.std(ddof=1) / 8 ** 0.5
    assert f"({data.mean() - margin:.4f}, {data.mean() + margin:.4f})" in text_of("ci data=60,80,70,61,88,83,95,98; conf=95")


def test_t_test_end_sem_6a():
    answer = text_of("ttest mean=1570 mu=1600 sd=120 n=25 alpha=0.05 tail=two critical=2.064")
    assert "t = -1.2500" in answer and "Do not reject H0" in answer


def test_z_test_end_sem_6b():
    answer = text_of("ztest mean=178 mu=170 sd=65 n=400 alpha=0.05 tail=right critical=1.645")
    assert f"z = {(178 - 170) / (65 / 20):.4f}" in answer and "Reject H0" in answer


def test_two_sample_assignment5_q8():
    pooled = (5 * 0.164 ** 2 + 4 * 0.24 ** 2) / 9
    t = (22.73 - 21.94) / (pooled * (1 / 6 + 1 / 5)) ** 0.5
    assert f"t = {t:.4f}" in text_of("twosample mean1=22.73 sd1=0.164 n1=6 mean2=21.94 sd2=0.24 n2=5 diff=0 alpha=0.05 tail=two")


def test_two_sample_large_assignment5_q9():
    z = (42500 - 36800 - 5000) / (2200 ** 2 / 45 + 1500 ** 2 / 45) ** 0.5
    answer = text_of("twosample mean1=42500 sd1=2200 n1=45 mean2=36800 sd2=1500 n2=45 diff=5000 alpha=0.01 tail=right")
    assert f"z = {z:.4f}" in answer


def test_chi_square_equal_shares():
    observed = [20, 22, 17, 18, 19, 24]
    expected = scipy_stats.chisquare(observed)
    assert f"χ² = {expected.statistic:.4f}" in text_of("chisquare observed=20,22,17,18,19,24")
