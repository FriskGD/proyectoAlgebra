import numpy as np
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt


def encontrar_inversa(matriz):
    try:
        inversa = np.linalg.inv(matriz)
        return inversa
    except np.linalg.LinAlgError:
        return "La matriz no es invertible"


def multiplicar_matrices(matriz1, matriz2):
    try:
        resultado = np.dot(matriz1, matriz2)
        return resultado
    except ValueError:
        return "Las matrices no tienen dimensiones compatibles para la multiplicación"


def resolver_sistema_ecuaciones(coeficientes, resultados):
    try:
        if len(coeficientes) == 2:  
            det = np.linalg.det(coeficientes)
            if det != 0:
                x = np.linalg.det(np.column_stack((resultados, coeficientes[:, 1]))) / det
                y = np.linalg.det(np.column_stack((coeficientes[:, 0], resultados))) / det
                return x, y
        elif len(coeficientes) == 3:  
            det = np.linalg.det(coeficientes)
            if det != 0:
                x = np.linalg.det(np.column_stack((resultados, coeficientes[:, 1], coeficientes[:, 2]))) / det
                y = np.linalg.det(np.column_stack((coeficientes[:, 0], resultados, coeficientes[:, 2]))) / det
                z = np.linalg.det(np.column_stack((coeficientes[:, 0], coeficientes[:, 1], resultados))) / det
                return x, y, z
    except np.linalg.LinAlgError:
        pass
    return "No se puede resolver el sistema de ecuaciones con la regla de Cramer"


def graficar_ecuaciones(ecuaciones):
    for ec in ecuaciones:
        x = symbols('x')
        y = symbols('y')
        ecuacion = Eq(ec[0] * x + ec[1] * y, ec[2])
        solucion = solve(ecuacion, y)
        y_vals = [solucion[x].subs(x, i) for i in range(-10, 11)]
        plt.plot(range(-10, 11), y_vals, label=str(ec))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gráfica de ecuaciones')
    plt.grid()
    plt.show()


matriz_a = np.array([[2, 3], [1, 4]])
print("Inversa de la matriz A:")
print(encontrar_inversa(matriz_a))


matriz_b = np.array([[5, 6], [7, 8]])
print("\nResultado de la multiplicación de matrices A y B:")
print(multiplicar_matrices(matriz_a, matriz_b))

coeficientes_2x2 = np.array([[2, 3], [1, 4]])
resultados_2x2 = np.array([5, 6])
print("\nSolución del sistema de ecuaciones 2x2 con la regla de Cramer:")
print(resolver_sistema_ecuaciones(coeficientes_2x2, resultados_2x2))

coeficientes_3x3 = np.array([[1, 1, 1], [2, -1, 1], [1, 2, -2]])
resultados_3x3 = np.array([5, 3, 2])
print("\nSolución del sistema de ecuaciones 3x3 con la regla de Cramer:")
print(resolver_sistema_ecuaciones(coeficientes_3x3, resultados_3x3))


ecuaciones_a_graficar = [(2, 3, 5), (1, -1, 0)]
graficar_ecuaciones(ecuaciones_a_graficar)
