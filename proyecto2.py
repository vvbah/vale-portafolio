import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
# Título
    st.title("Análisis Exploratorio de Datos")

#Body

    st.write("""
    En la actualidad, los datos se han convertido en un recurso valioso para las empresas, ya que ofrecen la posibilidad de descubrir 
    **insights clave** que respaldan la toma de decisiones estratégicas. Esta aplicación te permite cargar tu base de datos (*hasta 200 MB*) 
    y te entrega herramientas para realizar un análisis exploratorio, detectar datos anormales y crear tablas dinámicas para resumir la 
    información.    
         
    **Aquí podrás pasar del dato al insight**.  
         
    En el menú de la izquierda puedes cargar tu data y definir el análisis que deseas.
         """)

# Subida de archivos
    st.sidebar.header("Cargar datos")
    archivo_cargado = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

    if archivo_cargado:
    # Cargar archivo
        if archivo_cargado.name.endswith(".csv"):
            data = pd.read_csv(archivo_cargado)
        else:
            data = pd.read_excel(archivo_cargado)

    # Vista previa
        st.subheader("Vista previa de los datos")
        st.write(data.head())

    # Selección del análisis
        st.sidebar.header("Selecciona un análisis")
        analysis_type = st.sidebar.selectbox(
        "Elige el análisis que quieres realizar",
        ["Análisis Exploratorio", "Detección de Anomalías", "Tablas Pivot"]
    )

    # 1. Análisis Exploratorio
        if analysis_type == "Análisis Exploratorio":
            st.subheader("Análisis Exploratorio")
        
        # Estadísticas descriptivas
            st.write("Estadísticas generales:")
            st.write(data.describe())
        
        # Identificar columnas numéricas
            numeric_cols = data.select_dtypes(include=["float", "int"]).columns.tolist()
            if numeric_cols:
                col_to_plot = st.selectbox("Selecciona una columna para visualizar su distribución", numeric_cols)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data[col_to_plot], kde=True, ax=ax)
                ax.set_ylabel("Recuento")
                ax.set_title(f"Distribución de {col_to_plot}")
                st.pyplot(fig)
            else:
                st.write("No hay columnas numéricas para analizar.")

    # 2. Detección de Anomalías
        elif analysis_type == "Detección de Anomalías":
            st.subheader("Detección de Anomalías")
        
        # Seleccionar columna
            numeric_cols = data.select_dtypes(include=["float", "int"]).columns.tolist()
            if numeric_cols:
                col_to_analyze = st.selectbox("Selecciona una columna para buscar datos anormales", numeric_cols)
                mean, std = data[col_to_analyze].mean(), data[col_to_analyze].std()
            
            # Calcular outliers
                anomalies = data[(data[col_to_analyze] < mean - 3*std) | (data[col_to_analyze] > mean + 3*std)]
                st.write(f"Se encontraron {len(anomalies)} anomalías en la columna {col_to_analyze}:")
                st.write(anomalies)
        
        # Agregar gráfico boxplot
                st.write("Visualización con Boxplot:")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x=data[col_to_analyze], ax=ax)
                ax.set_title(f"Boxplot de {col_to_analyze}")
                st.pyplot(fig)
            else:
                st.write("No hay columnas numéricas para analizar.")

    # 3. Tablas Pivot
        elif analysis_type == "Tablas Pivot":
            st.subheader("Tablas Pivot")
        
        # Selección de columnas
            group_col = st.selectbox("Selecciona la columna para agrupar", data.columns)
            value_col = st.selectbox("Selecciona la columna para calcular", data.select_dtypes(include=["float", "int"]).columns)

        # Calcular pivot table
            pivot_table = data.groupby(group_col)[value_col].agg(["sum", "mean", "count"]).reset_index()
            pivot_table.rename(columns={"sum": "Suma", "mean": "Promedio", "count": "Recuento"}, inplace=True)
            st.write("Tabla pivot:")
            st.write(pivot_table)

if __name__ == "__main__":
    main()