import numpy as np
import matplotlib.pyplot as plt

# Simulación paso a paso de una red con una sola neurona y dos entradas
# Inicializamos valores
x = np.array([0.5, 0.85])     # Entradas
y_true = 1                    # Valor objetivo

# Pesos y bias iniciales (simulados)
w = np.array([0.15, 0.2])
b = 0.4

# Función sigmoide y derivada
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Forward propagation
z = np.dot(w, x) + b
a = sigmoid(z)

# Cálculo del error (función de coste simple: MSE)
error = 0.5 * (y_true - a)**2

# Backpropagation
# Derivada del error respecto a la activación
dE_da = -(y_true - a)

# Derivada de la activación respecto a z
da_dz = sigmoid_derivative(z)

# Derivada de z respecto a cada w (son las entradas x)
dz_dw = x

# Gradiente final: ∂E/∂w = ∂E/∂a * ∂a/∂z * ∂z/∂w
gradient_w = dE_da * da_dz * dz_dw

# Gradiente del bias (como ∂z/∂b = 1)
gradient_b = dE_da * da_dz

# Simulación de actualización de pesos
learning_rate = 0.1
new_w = w - learning_rate * gradient_w
new_b = b - learning_rate * gradient_b

import pandas as pd
import ace_tools as tools

# Preparamos un DataFrame para mostrar paso a paso
data = {
    'Paso': ['Entrada', 'Pesos iniciales', 'Bias inicial', 'Z (ponderación)', 'A (sigmoid)', 'Error (MSE)', 
             'Gradiente W', 'Gradiente B', 'W actualizado', 'B actualizado'],
    'Valor': [x, w, b, z, a, error, gradient_w, gradient_b, new_w, new_b]
}

df = pd.DataFrame(data)
tools.display_dataframe_to_user(name="Simulación de una neurona paso a paso", dataframe=df)
