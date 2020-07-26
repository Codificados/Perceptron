import numpy

def sigmoide(x):
    return 1 / (1 + numpy.exp(-x))

def derivada(x):
    return x * (1 - x)

input_entrenamiento = numpy.array([[1,1],
                                    [1,0],
                                    [0,1]])

output_entrenamiento = numpy.array([[1,1,0]]).T

numpy.random.seed(10)

pesos = numpy.random.random((2,1))

for iteracion in range(10000):
    capa_input = input_entrenamiento
    resultados = sigmoide(numpy.dot(capa_input, pesos))
    error = output_entrenamiento - resultados
    ajustes = error * derivada(resultados)
    pesos += numpy.dot(input_entrenamiento.T, ajustes)

    print("Iteracion:", iteracion)
    print(pesos)
dato_input = numpy.array([[0,1]])
resultado = sigmoide(numpy.sum(numpy.dot(dato_input, pesos)))

print("Resultado es:", resultado)