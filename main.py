from model import trainModel
import matplotlib.pyplot as plt
from data import x, y 

def main():
    model, history = trainModel(x, y)

    # imprimo en pantalla los parámetros del model (w y b) después del entrenamiento
    print("Parámetros del model:")
    print("w:", model.layers[0].get_weights()[0])
    print("b:", model.layers[0].get_weights()[1])

    # grafico el error cuadrático medio (ECM) vs. el número de épocas
    plt.plot(history.history['loss'])
    plt.xlabel('Epocas')
    plt.ylabel('Perdida')
    plt.title('Perdida por epoca')
    plt.show()

    # superpongo la recta de regresión resultante sobre los datos originales y visualizo ambas gráficas
    plt.scatter(x, y)
    plt.plot(x, model.predict(x), color='red')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Regression line')
    plt.show()

    # debo aclarar que cambie los datos del ejercicio para que se vea tanto el ECM como la recta de regresión
    # tambien tuve que modificar la cantidad de epocas porque a mi compu no le da la potencia.

main()