import numpy

class redNeuronal():

    def __init__(self):
        self.pesos = numpy.random.random((3,1))

    def sigmoidal(self, x):
        return 1 / (1 + numpy.exp(-x))

    def derivada(self, x):
        return x * (1 - x)

    def predecir(self, input):
        output = self.sigmoidal(numpy.sum(numpy.dot(input, self.pesos)))
        return output

    def entrenamiento(self,input_entrenamiento,output_entrenamiento, iteraciones ):
        for iteracion in range(iteraciones):
            capa_input = input_entrenamiento
            resultados = self.sigmoidal(numpy.dot(capa_input, self.pesos))
            error = output_entrenamiento - resultados
            ajustes = error * self.derivada(resultados)
            self.pesos += numpy.dot(input_entrenamiento.T, ajustes)

        print("Red Neuronal entranada con exito")
        print("El error en la {} iteracion es de {}".format(iteracion,error))

input_entrenamiento = numpy.array([[1,1,1],
                                    [1,0,0],
                                    [0,1,1]])

output_entrenamiento = numpy.array([[1,1,0]]).T

perceptron = redNeuronal()
perceptron.entrenamiento(input_entrenamiento, output_entrenamiento, 100000)

resultado = perceptron.predecir(numpy.array([[0,1,0]]))
print(resultado)