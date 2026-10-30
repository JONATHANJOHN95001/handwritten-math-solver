"""Questions in the wording of this year's assignments and exam papers, with the answers the method tests confirm."""
import pytest

from questions import answer_question

QUESTIONS = [
    # Matrix Algebra and Calculus
    ("Determine the rank of matrix A = [[-2,-1,-3,-1],[1,2,3,1],[1,0,1,1],[0,1,1,1]] by reducing it to row Echelon form.", "Rank = 3"),
    ("Find the rank of the matrix A= [ −2 −1 −3 −1 1 2 3 1 1 0 1 1 0 1 1 1 ] by reducing it to row echelon form.", "Rank = 3"),
    ("Calculate Eigen values of the matrix A=[[-1,3],[-2,4]]", "λ = 1, 2"),
    ("Determine the Eigenvalues and the corresponding Eigenvectors of the matrix A=[[8,-6,2],[-6,7,-4],[2,-4,3]].", "λ = 15"),
    ("Diagonalize the matrix A=[[8,-6,2],[-6,7,-4],[2,-4,3]].", "P ="),
    ("Reduce the matrix A = [[-1,3],[-2,4]] to the diagonal form and hence find A^4.", "A^4 ="),
    ("Test the consistency and solve x+y+z=9; x-2y+3z=8; 2x+y-z=3.", "x = 2"),
    ("Test for consistency and solve 𝑥−4𝑦+ 7𝑧= 14; 3𝑥+ 8𝑦−2𝑧= 13; 7𝑥−8𝑦+ 26𝑧= 5.", "inconsistent"),
    ("Solve the system of equations 𝑥+ 2𝑦+ 3𝑧= 0 ; 3𝑥+ 4𝑦+ 4𝑧= 0; 7𝑥+ 10𝑦+ 12𝑧= 0.", "x = 0"),
    ("Investigate the values of 𝜆 and 𝜇 such that the system of equations 𝑥+ 𝑦+ 𝑧= 6 ; 𝑥+ 2𝑦+ 3𝑧= 10 ; 𝑥+ 2𝑦+ 𝜆𝑧= 𝜇, may have a] Unique solution, b] Infinite Solution c] No solution.", "μ = 10"),
    ("Solve the system of equations 10x+y+z=12; x+10y+z=12; x+y+10z=12, by Gauss-Seidel method to obtain the final solution correct to three decimal places by performing 3 iterations.", "x = 1"),
    ("Employ Gauss-Seidel method to solve the following system of linear equations. 5x+2y+z=12, x+4y+2z=15, and x+2y+5z=20 Carry out four iterations taking the initial approximation to the solution as (1,0,3).", "x = 0.9958"),
    ("Solve y'''+4y''+y'-6y=0.", "y = "),
    ("Solve d3y/dx3+4d2y/dx2+dy/dx-6y=0.", "y = "),
    ("Find the particular integral of (D^2+5D+6)y=e^(3x).", "e^(3·x)/30"),
    ("Solve (D^2-2D+5)y=e^(2x)sin x.", "y = "),
    ("Solve the initial value problem: (D^3-D^2-4D+4)y=0; y(0)=1, y'(0)=2, y''(0)=1.", "y = "),
    ("Using the method of variation of parameters, solve y''-2y'+y=e^x/x.", "y = "),
    ("Solve (2x+1)^2y''-6(2x+1)y'+16y=8(2x+1)^2.", "y = "),
    ("Solve X' = [[-5,1],[4,-2]] X by diagonalization and hence discuss the stability of the system", "Stable"),
    ("Find the first and the second-order partial derivatives of the given function. z=x^3+y^3-3axy.", "∂/∂x"),
    ("Find the first order partial derivatives of the function z=x^3+y^3-3axy.", "∂/∂y"),
    ("If u=x+3y^2-z^3, v=4x^2yz, w=2z^2-xy. Find Jacobian of ∂(u,v,w) with respect to ∂(x,y,z) at (1,-1,0).", "J = "),
    ("Show that ∂(u,v,w)/∂(x,y,z) = 4, where u=yz/x, v=zx/y, w=xy/z.", "= 4"),
    ("Investigate the extreme value/s of the given function x^3+3xy^2-3x^2-3y^2+4.", "maximum, value 4"),
    ("Show that minimum value of u=xy+a^3/x+a^3/y is 3a^2.", "3·a^2"),
    ("The temperature T at any point (x,y,z) in space is given by T=400xyz^2. Find the highest temperature at the surface of a unit sphere x^2+y^2+z^2=1.", "Largest value 50"),
    ("A rectangular box open at the top is to have a volume of 32 cubic feet. Find its dimension if the total surface area is minimum.", "z = 2"),
    ("Prove that ∂u/∂x+∂u/∂y+∂u/∂z = 3/(x+y+z), where u = log(x^3+y^3+z^3-3xyz)", "True"),
    ("Compute div v for the vector v=3xz i+2xy j-yz^2 k.", "div v"),
    ("Given f(x,y,z)=2x+yz-3y^2 then calculate grad f", "(-6·y + z)j"),
    ("Find Curl v, where v=∇(x^3+y^3+z^3-3xyz).", "= 0"),
    ("Find curl (curl v) given that v=xy i+y^2z j+z^2y k.", "curl(curl v)"),
    ("Show that the vector point function v=(x i+y j)/(x^2+y^2) is both solenoidal and irrotational.", "irrotational"),
    ("Show that F=(2xy^2+yz)i+(2x^2y+xz+2yz^2)j+(2y^2z+xy)k is a conservative force field. Find its Scalar potential.", "conservative"),
    ("Find the directional derivative of the function f(x,y,z)=2x^2+3y^2+z^2 at the point (2,1,3) in the direction of the vector a=i-2k.", "-1.78885"),
    ("Calculate directional derivatives of φ=4xz^3-3x^2y^2z. At (2,-1,2) along 2i-3j+6k.", "Directional derivative"),
    ("Find the components of velocity and acceleration at t=2 on the curve r=(t^2+1)i+(4t-3)j+(2t^2-6t)k in the direction i+2j+2k.", "16/3"),
    ("If x=t^2+1, y=4t-3, z=2t^2-6t represents the parametric equation of a curve, then find the unit tangent vector at any point on the curve and the angle between the tangents at t=1 and t=2", "cos θ"),
    ("Find the angle between the spheres x^2+y^2+z^2=29 and x^2+y^2+z^2+4x-6y-8z-47=0 at the point (4,-3,2).", "√(551)/29"),
    ("Solve the integral ∫_0^π x sin^7 x dx using Reduction formula.", "π"),
    ("Solve ∫_0^(π/2) sin^4 x dx", "3·π/16"),
    ("Solve the integral ∫_0^∞ dx/(1+x^4) using Beta-Gamma function.", "≈"),
    ("Write the general form of Cauchy's Differential Equations.", "Cauchy"),
    ("Define rank of the matrix.", "linearly independent"),
    # Statistics and Probability
    ("In a moderately asymmetrical distribution, the values of mode and mean are 32.1 and 35.4 respectively. Find the median value.", "Median = 34.3"),
    ("In a survey of 10 households, the number of children was found to be 4, 1, 5, 4, 3,7, 2, 3, 4,1. Determine the mode. Calculate the mean number of children per household and the median.", "Mean = 3.4"),
    ("The mileage (km per litre) of 50 cars of the same model was tested by a manufacturer, and details are tabulated as given below: Mileage (km/l) 10 - 12 12 - 14 14 - 16 16 - 18 Number of cars 7 12 18 13 Find the mean mileage.", "Mean = 14.48"),
    ("A cinema records the number of tickets sold in a week: Tickets Sold Per Day: 120, 150, 180, 200, 170, 140, 190 Frequency (Days): 1, 2, 2, 1, 1, 2, 2 Compute the mean, median, mode, variance, and standard deviation.", "Mean ="),
    ("A coach records the heights of 8 players in different height categories: Height Range (cm): 160-165, 165-170, 170-175, 175-180, 180-185, 185-190, 190-195, 195-200 Number of Players: 1, 2, 3, 4, 3, 2, 1, 1 Compute the mean, median, mode, variance, and standard deviation.", "Mean ="),
    ("The following table represents the marks obtained by a batch of 10 students in certain class tests in statistics and Accountancy. Identify in which subject is the level of knowledge higher? Serial No 1 2 3 4 5 6 7 8 9 10 Marks (Statistics) 53 55 52 32 30 60 47 46 35 28 Marks (Accountancy) 57 45 24 31 25 84 43 80 32 72", "Higher level: accountancy"),
    ("If 8x-10y+66=0 and 40x-18y=214 are the two regression lines, then determine means's of x's, y's and the correlation coefficient also calculate σy if σx=3?", "s.d. of y = 4"),
    ("A study was conducted to examine the relationship between height and weight of 10 individuals. Person 1 2 3 4 5 6 7 8 9 10 Height (inches) 60 62 64 65 66 67 69 70 72 74 Weight (lbs) 115 120 130 135 140 145 155 160 170 180 Calculate the correlation coefficient between height and weight. Find the regression equations.", "r = 0.99"),
    ("A mobile phone retailer collects data on the age of smartphones (in years) and their resale values: Age(x) 1 2 3 4 5 Resale Value(y) 50000 40000 32000 25000 15000 Determine the linear regression equation for the given data. Also, estimate the resale price of a smartphone that is 3.5 years old.", "At x = 3.5"),
    ("Using the data, calculate the quadratic regression equation that models the least square sense for the following data. x 0 1 2 3 4 5 6 y 1 1.8 1.2 1.5 2.5 2.2 3", "x²"),
    ("In a bolt factory there are four machines A, B, C, D manufacturing respectively 20%, 15%, 25%, 40% of the total production. Out of these 5%, 4%, 3%, 2% are defective. If a bolt drawn at random was found defective what is the probability that it was manufactured by A or D?", "4/7"),
    ("The chance that a doctor will diagnose a disease correctly is 60%. The chance that a patient will die after correct diagnose is 40% and the chance of death by wrong diagnosis is 70%. If a patient dies, what is the chance that his disease was correctly diagnosed?", "6/13"),
    ("A bag contains 10 white and 3 red balls while another bag contains 3 white and 5 red balls. 2 balls are drawn at random from the first bag and put in the second bag. Then a ball is drawn at random from the second bag. What is the probability that it is a white?", "P(white)"),
    ("Given P(A)=1/2, P(B)=1/3 and P(A∩B)=1/5, Find P(AUB), P(A|B) and P(B|A).", "19/30"),
    ("In a Poisson distribution 3P(X=2)=P(X=4). Find the parameter 'm'.", "m = 6"),
    ("In the quiz contest of answering 'Yes' or 'No' what is the probability of guessing at least 6 answers correctly out of 10 questions asked? Also find the probability of same if there are 4 options for a correct answer.", "P(X>=6)"),
    ("From 6 positive and 8 negative numbers, 4 numbers are chosen at random (without replacement) and multiplied. What is the probability that the product is a positive number?", "505/1001"),
    ("20 wristwatches in a box of 100 are defective. If 10 watches are selected at random. Find the probability that (i) 10 are defective (ii) 10 are good (iii) at least one watch is defective (iv) at most 3 are defective.", "P(X<=3)"),
    ("If the probability of a bad reaction from a certain injection is 0.001, determine the chance that out of 2000 individuals, more than two will get a bad reaction.", "P(X>2)"),
    ("In a certain factory turning out razor blades, there is a small chance of 0.002 For any blade to be defective. The blades are supplied in a packet of ten, use Poisson's distribution to calculate the approximate number of packets no defective, one defective and two defective blades respectively in a consignment of 10,000 packets.", "expected count"),
    ("Find the constant k such that f(x)= kx^2, 0<x<3 is a probability distribution function. Also find compute P(1<x<2), P(x<=1), P(x>1).", "k = 1/9"),
    ("Find the value of k such that the following distribution represents a finite probability distribution. Hence find its mean and standard deviation. Also find p(x<=1), p(x>1) and p(-1<x<=2). x -3 -2 -1 0 1 2 3 p(x) k 2k 3k 4k 3k 2k k", "k = 0.0625"),
    ("The amount of time (in minutes), that a person must wait for a bus is uniformly distributed between 0 and 15 minutes, inclusive. What is the probability that a person waits fewer than 12.5 minutes?", "5/6"),
    ("In a test on 2000 electric bulbs, it was found that life of a particular make was normally distributed with an average life of 2040 hours and S.D of 60 hours. Estimate the number of bulbs likely to burn for a) more than 2150 hours b) less than 1950 hours and c) more than 1920 hours and but less than 2160 hours.", "expected count"),
    ("The marks of 1000 students in an examination follows a normal distribution with mean 70 and standard deviation 5. Find the number of students whose marks will be (i) less than 65 (ii) more than 75 (iii) between 65 and 75.", "expected count"),
    ("A population consists of the five numbers 2,3,6,8,11. Consider all possible samples of size 2 which can be drawn with replacement from this population. Find the mean of the population, the standard deviation, the mean of the sample distribution of means, and the standard error of means.", "Number of samples = 25"),
    ("A population consists of six numbers 4, 8, 12, 16, 20, 24. Consider all samples of size two which can be drawn without replacement from this population.", "Number of samples = 15"),
    ("A random sample of size 64 is taken from a normal population with μ=51.4 and σ=6.8. What is the probability that the mean of the sample will (a) exceed 52.9 (b) fall between 50.5 and 52.3 (c) be less than 50.6.", "P(xbar>52.9)"),
    ("Three masses are measured as 62.34, 20.48, 35.97 kgs with S.D 0.54, 0.21, 0.46 kgs. Find the mean and S.D of the sum of the masses.", "118.79"),
    ("It is generally believed that electrical problems affect about 14% of new cars. An automobile mechanic conducts diagnostic tests on 128 new cars on the lot. What is the probability that in this group over 18% of the new cars will be found to have electrical problems?", "P(phat>0.18)"),
    ("A factory produces rechargeable batteries, and the lifespan of these batteries follows an approximately normal distribution with a standard deviation of 50 hours. A sample of 35 batteries is tested, and the average lifespan is found to be 900 hours. Find a 95% confidence interval for the population mean lifespan of all batteries produced by this factory. (Given z_c = 1.96)", "(883.4"),
    ("A random sample of 15 bolts is taken in a study of the tensile strength of the bolts. Measurements of the tensile strength were made for each of the 15 bolts, yielding an average value of 55.20 with a sample standard deviation of 2.1. Assuming the measurements to be normally distributed, construct a 95% confidence interval for the mean tensile strength of the bolts.", "(54.03"),
    ("A random sample of 8 girls had the following I.Qs: 60, 80, 70, 61, 88, 83, 95 and 98. Find a 95 % confidence interval.", "confidence interval"),
    ("In 64 randomly selected hours of production, the sample mean and sample s.d. of the number of acceptable pieces produced by an automatic machine are 1038 and 146, respectively. At a= 0.05, does this enable us to reject H0 : m= 1000 against H1 : m> 1000?", "Reject H0"),
    ("The mean life time of a sample of 25 fluorescent light bulbs produced by a company is computed to be 1570 hours with a standard deviation of 120 hours. The company claims that the average life of the bulbs produced by the company is 1600 hours using the level of significance of 0.05. Is the claim acceptable? (t0.025, 24 = 2.064)", "Do not reject H0"),
    ("A department store manager determines that a new billing system will be cost effective only if the mean monthly account is more that $170. A random sample of 400 monthly accounts is drawn, for which the sample mean is $178. It is known that the accounts are approximately normally distributed with standard deviation of $65. At the 5% significance level, can we conclude that the new system will be cost-effective? (Given zc = 1.645)", "Reject H0"),
    ("Assuming normality of the two density distributions, test H0: μ1-μ2 = 0 versus Ha: μ1-μ2 ≠ 0 using the data: m=6, xbar=22.73, s1=0.164, n=5, ybar=21.94, s2=0.240", "t = "),
    ("A manufacturing company wants to compare the average lifetimes of two different battery brands, A and B. A random sample of 100 batteries from Brand A has a mean lifetime of 500 hours with a standard deviation of 50 hours. For Brand B, a random sample of 120 batteries has a mean lifetime of 520 hours with a standard deviation of 60 hours. At a 5% significance level, test whether there is a significant difference.", "z = "),
    ("A market survey is conducted among 400 consumers. Test at a 5% significance level whether the consumer preferences follow an equal distribution. Observed: 120 90 100 90", "χ²"),
    ("Define the correlation coefficient and specify its range.", "-1 ≤ r ≤ 1"),
    ("Discuss Type I and Type II errors with suitable examples.", "Type I error"),
    ("State and explain the Central Limit Theorem.", "σ/√n"),
    # Discrete Mathematics and Graph Theory
    ("Cars of a particular manufacturer come in 4 models, 12 colours, 3 engine sizes, and 2 transmission types. How many distinct cars (of this company) can be manufactured?", "288"),
    ("A license plate consists of two English letters followed by four digits. If repetitions are allowed, how many of the plates have only vowels (A, E, I, O, U) and even digits (0, 2, 4, 6, 8)?", "15625"),
    ("Show that if 30 dictionaries in a library contain a total of 61,327 pages, then one of the dictionaries must have at least 2045 pages.", "2045"),
    ("There are eight letters to eight different people to be placed in eight different envelopes. Find the number of ways of doing this so that at least one gets to the right person.", "25487"),
    ("Find the number of Non-negative integer solutions of x1 + x2 + x3 + x4 + x5 = 30, Where x1 >= 2, x2 >= 3, x3 >= 4, x4 >= 2, x5 >= 0.", "8855"),
    ("Find the number of non-negative integer solutions of the equation x1+x2+x3 <= 11.", "364"),
    ("How many integers are between 1 and 300 (inclusive) i) Divisible by at least one of 5, 6, 8? ii) Divisible by none of 5, 6, 8?", "by none"),
    ("In how many ways can the 26 letters of the English alphabet be permuted so that none of the pattern's CAR, DOG, PUN or BYTE occurs?", "permutations"),
    ("Find the rook polynomial for the 3 × 3 board by using the expansion formula.", "6·x^3"),
    ("An apple, a banana, a mango, and an orange are to be distributed to four boys, B1, B2, B3, and B4. The boys B1 and B2 do not wish to have an apple, the boy B3 does not want a banana or a mango, and B4 refuses an orange. In how many ways can the distribution be made so that no boy is displeased?", "ways"),
    ("Determine the coefficient of x20 in (x2 + x3 + x4 + x5 + x6)5", "381"),
    ("Determine the coefficient of x^15 in (1+x)^4/(1-x)^4.", "Coefficient"),
    ("Find the generating functions for the following sequence (i) 0,1,-2,3,-4,... (ii) 1,2,3,4,...", "G(x)"),
    ("Find the sequence generated by the following functions (i) (3+x)^3, (ii) 3x^3+e^(2x).", "27, 27, 9"),
    ("Determine the sequences for which the following are the exponential generating functions (i) 6e^(5x)-3e^(2x), (ii) 3/(1-2x)+e^x.", "3, 24"),
    ("In how many ways can we distribute 24 pencils to 4 children so that each child gets at least 3 pencils but not more than eight.", "solutions"),
    ("Using generating function, find the number of partitions of n = 6.", "p(6) = 11"),
    ("Using exponential generating function, find the number of ways in which 4 of the letters in the word ENGINE be arranged.", "arrangements"),
    ("There are 10 marbles of the same size but of different colors (1 red, 3 blue, 2 green, 2 orange, 1 white, and 1 black) in a bag. Find in how many ways six of these marbles can be arranged in a row.", "arrangements"),
    ("Solve the recurrence relation a(n+1) = 4a(n), given that a(0) = 3", "3·4^n"),
    ("Solve the recurrence relation an+1 = 4an, for given that a0 = 3", "3·4^n"),
    ("Solve the recurrence relation Fn+2 = Fn+1 + Fn ∀n >= 0 given F0 = 0, F1 = 1", "F(n)"),
    ("Solve the recurrence relation an + 4an-1 + 4an-2 = 8 and a0 = 1, a1 = 2 for n >= 2", "a(n)"),
    ("Solve the recurrence relation an-3an-1 = 5 × 7n for n >= 1 given that a0 = 2", "a(n)"),
    ("The number of virus-affected files in a system is 1000 (to start with) and this increases 250% every two hours. Use a recurrence relation to determine the number of virus affected files in the system after one day.", "3379220508"),
    ("There are 3 pegs fixed vertically on a table, and n circular disks ... Determine the number of moves for the transfer of all the n disks", "2^n - 1"),
    ("Suppose that there are n >= 2 persons at a party and that each of these persons shakes hands (exactly once) with all of the other persons present. Using a recurrence relation find the number of hand shakes.", "n(n-1)/2"),
    ("Prove that K5 is non-planar graph.", "not planar"),
    ("Consider the graph K2,3. Find the chromatic polynomial of the graph.", "P(λ)"),
    ("A connected planar graph G has 9 vertices with degrees 2, 2, 3, 3, 3, 4, 5, 6, 6. Find the number of regions of G.", "r = 10"),
    ("If a tree T has four vertices of degree 2, one vertex of degree 3, two vertices of degree 4, one vertex of degree 5. Find the number of leaves in T.", "Leaves = 10"),
    ("A class room has 25 microcomputers that has to be connected to a wall socket that has four outlets. The connections are made by using extension cords that have four outlets each. What is the least number of cords needed to get computer to use.", "8 extension cords"),
    ("Using the merge sorting method sort the list {12, 11, 2, 3, 1, 5, 6, 7, 10, 9, 4, 15}.", "[1, 2, 3"),
    ("Construct an optimal prefix code for the symbols a, o, q, u, y, z that occur with frequencies 20, 28, 4, 17, 12, 7, respectively.", "Total weighted length = 210"),
    ("Construct an optimal prefix code for the message ROAD IS GOOD. Indicate the code.", "space"),
    ("State the product rule.", "m × n"),
    ("Define complete graph with an example.", "n(n-1)/2"),
]


@pytest.mark.parametrize("question, expected", QUESTIONS, ids=[q[:50] for q, _ in QUESTIONS])
def test_question_as_written(question, expected):
    result = answer_question(question)
    assert expected in "\n".join(result.answer)


def test_shows_what_it_understood():
    assert answer_question("Find the rank of the matrix A=[[1,2],[3,4]].").read_as == "rank [[1,2],[3,4]]"


def test_drawn_graph_asks_for_the_edges():
    with pytest.raises(ValueError, match="edges"):
        answer_question("Construct a minimal spanning tree for the given weighted graph using Prim's Algorithm.")


def test_sentence_it_cannot_understand_is_refused_not_guessed():
    with pytest.raises(ValueError, match="understand"):
        answer_question("Find the price of a pizza in the town where the river is")
