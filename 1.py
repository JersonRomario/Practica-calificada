import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class DataHandler:
    def __init__(self, file):
        self.file = file
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file, delimiter=';')
        if self.data.columns[0] != 'edad':
            self.data.columns = ['edad', 'peso', 'altura']

    def preview_data(self):
        return self.data.head(10)

    def calculate_statistics(self):
        if 'edad' in self.data.columns and 'peso' in self.data.columns and 'altura' in self.data.columns:
            numeric_data = self.data[['edad', 'peso', 'altura']].dropna()
            mean = numeric_data.mean()
            median = numeric_data.median()
            std_dev = numeric_data.std()
            return mean, median, std_dev
        else:
            return None, None, None

    def plot_regression(self):
        if 'edad' in self.data.columns and 'peso' in self.data.columns and 'altura' in self.data.columns:
            numeric_data = self.data[['edad', 'peso']].dropna()
            X = numeric_data[['edad']].values.reshape(-1, 1)
            y = numeric_data['peso'].values

            model = LinearRegression()
            model.fit(X, y)

            y_pred = model.predict(X)

            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, color='blue', label='Datos reales')
            plt.plot(X, y_pred, color='red', linewidth=2, label='Regresión lineal')
            plt.xlabel('Edad')
            plt.ylabel('Peso')
            plt.title('Regresión lineal: Edad vs Peso')
            plt.legend()
            plt.grid(True)
            return plt
        else:
            return None

def main():
    st.title('Aplicación de Análisis de Datos')

    file = st.file_uploader("Cargar CSV", type=["csv"])

    if file is not None:
        data_handler = DataHandler(file)
        data_handler.load_data()

        st.subheader('Vista Previa de los Datos')
        st.write(data_handler.preview_data())

        mean, median, std_dev = data_handler.calculate_statistics()
        if mean is not None:
            st.subheader('Estadísticas')
            st.write("Media:")
            st.write(mean)
            st.write("Mediana:")
            st.write(median)
            st.write("Desviación Estándar:")
            st.write(std_dev)
        else:
            st.write("No se encontraron las columnas 'edad', 'peso' y 'altura' en el conjunto de datos.")

        st.subheader('Gráfica de Regresión Lineal')
        plt = data_handler.plot_regression()
        if plt:
            st.pyplot(plt)
        else:
            st.write("No se encontraron las columnas 'edad' y 'peso' en el conjunto de datos.")

if __name__ == "__main__":
    main()