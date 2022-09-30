import numpy as np
import csv
from scipy.optimize import linprog

archivo = open('avaluos1.csv', 'r')
aux_avaluos = csv.reader(archivo)
    # Los contenidos del archivo csv quedan en aux_avaluos,
    # cuya estructura de dato es de tipo "lista". Cada fila del archivo aparece como un
    # elemento de esa lista (o sea, aux_avaluos es una lista de filas). Cada fila,
    # a su vez, es una lista de números. O sea, aux_avaluos será una lista de listas
    # (como una matriz).

N=0;
    # Llamaremos N al número de recursos estratégicos a defender
    # Empieza en cero y va aumentando a medida que leamos filas del archivo avaluos.csv
avaluos = [];
    #  Guardaremos los montos en que el agente avalúa cada recurso estratégico
    # en el vector avalúos, con N componentes.
    #  Como estructura de dato, comenzaremos usando una lista.
    #  Aquí la inicializamos como una lista vacía.

for fila in aux_avaluos:
    avaluos.append(float(fila[0]));
    N = N+1;    # La variable N aumenta con cada fila leída

archivo.close()

####
# Ahora construímos la matriz de N x N con entradas
# R_{ij} = avaluo_i   si i=j,
#           -avaluo_j si i es distinto de j,
# la cual define el problema de optimización
# Pensando en hacer cálculos, como estructura de datos usaremos
# un "array" de la librería numpy (que en este código hemos importado bajo el pseudónimo
# "np"). La inicializamos como una matriz de ceros con el tamaño correcto.

tamaño = (N,N);
R = np.zeros(tamaño);

for i in range(0,N):
    for j in range(0,N):
        if i==j:
            R[i][j]= avaluos[i];
            # R[i][j] = 0;
        if i!=j:
            R[i][j] = -avaluos[j];


# Ahora procedemos a construir las matrices A_{ub} y A_{eq}
# y los vectores c, b_{ub}, b_{eq} y l que pide el scipy.linprog como argumentos

# Como función objetivo tomaremos la siguiente:
# 0*x_0 + 0*x_1 + ... + 0*x_{N-1} + (-1)*x_N

c = np.zeros(N+1);
c[N] = -1;
# Esto produce que c*x sea igual a 0*x_0 + 0*x_1 + ... + 0*x_{N-1} + (-1)*x_N,
# esto es, la función objetivo (a minimizar por linprog) será -x_N

# Las primeras N restricciones son de la forma
#  0*x1 + 0*x2 + ... + (-1)*x_j + ... + 0*x_{N}  <= 0
tamaño = (N+N, N+1);
A_ub = np.zeros(tamaño);    # Inicializamos con ceros, pero le damos ya el tamaño correcto

# Primero llenaremos las filas correspondientes a las primeras N desigualdades,
# las que son de la forma (-x_i)<=0. En esas filas todas las entradas son cero, salvo
# las "de la diagonal", que llevan un -1.
for i in range(0,N):
    A_ub[i][i] = - 1;


# Las segundas N restricciones son de la forma:
# (-R_0j)*x0 + (-R_1j)*x1 + ... + (-R_{(N-1),j})*x_{N-1} + 1*x_N

# Ahora llenamos "la parte de abajo" de la matriz A_ub. Se trata de las N filas
# correspondientes a las restricciones de la forma
#    (-R_0j)*x_0  + (-R_1j)*x1 + .... + (-R_(N-1)j)*x_{N-1} + x_N <= 0.
# Dejaremos la última columna para el final, llenaremos primero las cols. de la 0 a la N-1
for j in range(0,N):
    fila_A=N+j;
    for col_A in range(0,N):
        i = col_A;              # poner aquí mucha atención:
                                # en la naturaleza de nuestro problema
                                # está el que al pasar a A_{ub} la matriz R
                                # se traspone.
        A_ub[fila_A][col_A] = - R[i][j];
    # Ahora llenamos la última columna, que es una columna de unos
    A_ub[fila_A][N] = 1;

# Vamos ahora por el lado derecho de las restricciones con desigualdad
b_ub = np.zeros(N+N)      # En las N+N desigualdades es un cero lo que hay al lado derecho.
                        # Se nos pide que sea un array unidimensional

# Solo hay una restricción con igualdad

# Ahora las restricciones con igualdad A_eq * x = b_eq. En nuestro caso solo tenemos
# una restricción (x_0+ ... + x_{N-1} = 1), entonces A será un arreglo bidimensional
# de una sola fila y N+1 columnas  (con un cero en la última componente porque
# x_N no aparece en la ecuación que define la restricción del espacio de búsqueda).
tamaño = (1,N+1);
A_eq = np.zeros(tamaño);   # Definimos el tamaño de la matriz. Inicializamos con ceros.
for i in range(N):
    A_eq[0][i] = 1;

# Viene a continuación el lado derecho. Se pide que b_eq sea un arreglo unidimensional.
# En este caso, será un arreglo de tamaño 1. El valor de la única componente también es 1,
# porque el lado derecho de la ecuación x_0 + ... + x_{N-1} = 1 es, justamente, 1.
tamaño = 1;
b_eq = np.zeros(tamaño);
b_eq[0] = 1;


# Estamos listos para invocar al comando linprog
# Lo de
#   bounds=(None, None)
# es para que no imponga ni una cota inferior común l
# ni una cota superior común u a todas las incógnitas
# (una restricción general del tipo l<=x_i <= u para todo i)
# Si no ponemos el "boundes=(None,None)" entonces solo
# busca entre los vectores x que tienen todas sus componentes negativas.
# En el problema de optimización que estamos considerando,
# no encontraremos una solución si a la última de las incógnitas,
# a x_N, le impedimos tomar valores negativos.
resultado = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))

print(resultado.x)
print(resultado.message)

####
## Ahora preparémonos para imprimir los resultados en un archivo .csv

aux_salida = []
for i in range(0,N):
    aux_salida.append([i, avaluos[i], round(resultado.x[i]*100, 1) ])

archivo = open('resultado.csv', 'w');
with archivo:
    writer = csv.writer(archivo);
    writer.writerows(aux_salida)
archivo.close()
