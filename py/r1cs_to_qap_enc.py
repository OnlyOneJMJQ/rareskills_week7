import numpy as np
import galois
from functools import reduce
from py_ecc.bn128 import curve_order, G1, G2, multiply, add

# # Original Equation:
# # out = x**4 - 5*y**2*x**2
# #
# # Constraints:
# # v1 = x * x
# # v2 = v1 * v1
# # v3 = -5y * y
# # -v2 + out = v3*v1
# #
# # Witness:
# # [1, out, x, y, v1, v2, v3]


# -----
# Define Matrices L, R, and O
# -----

L = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, -5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
])

R = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
])

O = np.array([
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, -1, 0],
])


# -----
# Verify R1CS
# -----

print("Verifying R1CS...")

x = 4
y = -2
v1 = x * x
v2 = v1 * v1 # x^4
v3 = -5*y * y
out = v3*v1 + v2 # -5y^2 * x^2

witness = np.array([1, out, x, y, v1, v2, v3])

assert all(np.equal(np.matmul(L, witness) * np.matmul(R, witness), np.matmul(O, witness))), "not equal"
print ("R1CS verified!")


# -----
# Redefine Matrices to Eliminate Negative Values
# -----

def remove_negatives(matrix):
    return np.where(matrix < 0, matrix + curve_order, matrix)

L = remove_negatives(L)
R = remove_negatives(R)
O = remove_negatives(O)


# -----
# Recalculate Witness and Verify Galois Field
# -----

print("Verifying Galois Field...")

GF = galois.GF(curve_order)

L_galois = GF(L)
R_galois = GF(R)
O_galois = GF(O)

x = GF(4)
y = GF(curve_order-2) # we are using 79 as the field size, so 79 - 2 is -2
v1 = x * x
v2 = v1 * v1 # x^4
v3 = GF(curve_order-5)*y * y
out = v3*v1 + v2 # -5y^2 * x^2

witness = GF(np.array([1, out, x, y, v1, v2, v3]))

assert all(np.equal(np.matmul(L_galois, witness) * np.matmul(R_galois, witness), np.matmul(O_galois, witness))), "not equal"
print("Galois Field verified!")


# -----
# Interpolate Polynomial
# -----

def interpolate_column(col):
    xs = GF(np.array([1,2,3,4]))
    return galois.lagrange_poly(xs, col)

# axis 0 is the columns. apply_along_axis is the same as doing a for loop over the columns and collecting the results in an array
U_polys = np.apply_along_axis(interpolate_column, 0, L_galois)
V_polys = np.apply_along_axis(interpolate_column, 0, R_galois)
W_polys = np.apply_along_axis(interpolate_column, 0, O_galois)


# -----
# Calculate h(x)
# -----

def inner_product_polynomials_with_witness(polys, witness):
    mul_ = lambda x, y: x * y
    sum_ = lambda x, y: x + y
    return reduce(sum_, map(mul_, polys, witness))

term_1 = inner_product_polynomials_with_witness(U_polys, witness)
term_2 = inner_product_polynomials_with_witness(V_polys, witness)
term_3 = inner_product_polynomials_with_witness(W_polys, witness)

# t = (x - 1)(x - 2)(x - 3)(x - 4)
t = galois.Poly([1, (curve_order - 1)], field = GF) * galois.Poly([1, (curve_order - 2)], field = GF) * galois.Poly([1, (curve_order - 3)], field = GF) * galois.Poly([1, (curve_order - 4)], field = GF)

h = (term_1 * term_2 - term_3) // t


# -----
# Verify QAP
# -----

print("Verifying QAP...")

assert term_1 * term_2 == term_3 + h * t, "division has a remainder"
print("QAP verified!")


# -----
# Evaluating Polynomials
# -----

print("Evaluating Polynomials...")

GF = galois.GF(curve_order)

# Not faster unfortunately :(

# term_1 = galois.Poly([0, 21888242871839275222246405745257275088548364400416034343698204186575808495616, 21888242871839275222246405745257275088548364400416034343698204186575808495614, 28, 21888242871839275222246405745257275088548364400416034343698204186575808495597], field=GF)
# term_2 = galois.Poly([0, 11, 21888242871839275222246405745257275088548364400416034343698204186575808495536, 178, 21888242871839275222246405745257275088548364400416034343698204186575808495513], field=GF)
# term_3 = galois.Poly([0, 82, 21888242871839275222246405745257275088548364400416034343698204186575808494867, 1916, 1, 21888242871839275222246405745257275088548364400416034343698204186575808494385], field=GF)
# t = galois.Poly([1, 21888242871839275222246405745257275088548364400416034343698204186575808495607, 35, 21888242871839275222246405745257275088548364400416034343698204186575808495567, 24], field=GF)
# h = galois.Poly([0, 0, 21888242871839275222246405745257275088548364400416034343698204186575808495606, 21888242871839275222246405745257275088548364400416034343698204186575808495555, 138], field=GF)

tau = 6
powers_of_tau_G1 = [multiply(G1, tau**4), multiply(G1, tau**3), multiply(G1, tau**2), multiply(G1, tau), G1]
powers_of_tau_G2 = [multiply(G2, tau**4), multiply(G2, tau**3), multiply(G2, tau**2), multiply(G2, tau), G2]

def evaluate_poly(coeffs, powers_of_tau):
    final_point = None
    for i in range(len(coeffs)):
        if (coeffs[i] == 0):
            continue
        if (final_point == None):
            final_point = multiply(powers_of_tau[i], int(coeffs[i]))
            continue
        final_point = add(final_point, multiply(powers_of_tau[i], int(coeffs[i])))
    return final_point

def evaluate_HT(coeffs, T):
    final_point = None
    for i in range(len(coeffs)):
        if (coeffs[i] == 0):
            continue
        if (final_point == None):
            final_point = multiply(T, int(coeffs[i]))
            continue
        final_point = add(final_point, multiply(T, int(coeffs[i])))
    return final_point

print("Calculating A, B, C_prime...")

A = evaluate_poly(term_1.coefficients(5), powers_of_tau_G1)
B = evaluate_poly(term_2.coefficients(5), powers_of_tau_G2)
C_prime = evaluate_poly(term_3.coefficients(5), powers_of_tau_G1)

print("A: ", A)
print("B: ", B)
print("C_prime: ", C_prime)

print("Calculating C...")

# Evaluate h(tau) to get coefficients to use a scalars to mulitply with G points for t
T = evaluate_poly(t.coefficients(5), powers_of_tau_G1)
HT = evaluate_HT(h.coefficients(5), T)

print("T: ", T)
print("HT: ", HT)

C = add(C_prime, HT)

print("C: ", C)

# A:  (13759698281541362948100552505387368848018484641321223091947306654885243690704, 19085758730982447701000020042396817950058300487342942250774036309437784519175)
# B:  ((20612212392094086306364540745946752302460861136183711417678330327971014080866, 6437904630709262427076771279303527374478637745469415580882092156608085150344), (21565022623258350584099627224199690557564040841853974577221108886135149707376, 185005392932418116559214466311134383327539521921765719365598421090223115630))
# C_prime:  (12806406773824251777842571326230006263001972579768453696688118625729906387296, 15904296211112431992352702789230228132270987756070948971623182790675116062059)
# T:  (2747517507890653313006032249699734168352039494722462666318484735518429114319, 17769594319884551394326400904703145588834032543352917093414832572556669509807)
# HT:  (8594507273757250065894197846687656576232923295568279515819483835305163312182, 3990342454086166001284419026135753058139574459976979118242633689545119334373)
# C:  (14621062448483834300513373961816752655982234698599643148809688710706661982167, 19681235089638021251169009609775845392078975762008150095307573108444991436405)
