import streamlit as st
import pandas as pd
import joblib
import os


####################################################################################################################
######################################## PROYECTO 1 ################################################################
####################################################################################################################

#### APP: 

### TITULO ###
def main():

    st.title("Predicción con Machine Learning")

    ### INTRODUCCION: ###

    st.write("""
    El aprendizaje automático es un campo que ha transformado diversas industrias al permitir realizar predicciones o clasificaciones a 
    partir de datos históricos, lo cual facilita la toma de decisiones estratégicas. Un sector donde la inteligencia artificial ha cobrado 
    gran relevancia es el marketing. Las empresas, especialmente en el sector retail, suelen almacenar la información de compra de los 
    clientes para diseñar estrategias personalizadas que fomenten su fidelización. Un ejemplo clásico son los descuentos impresos en las 
    boletas de compra de supermercados, las cuales se generan después de ingresar el RUT de un comprador frecuente, generalmente basados 
    en su historial de compras, entre otros ejemplos.  """)
    
    st.header('Predicción Precio Propiedades Santiago, Chile')

    st.markdown("""

    La siguiente aplicación tiene como objetivo predecir el valor en UF de las propiedades en Santiago de Chile. El modelo fue entrenado 
                con 20.000 registros del año 2022, considerando las tres variables con mayor correlación con el precio: superficie útil, 
                número de dormitorios y número de baños. Este modelo proporciona una estimación de los precios en el mercado inmobiliario 
                de la capital, permitiendo a los interesados obtener una aproximación de lo que podrían pagar por una propiedad según sus 
                características. Cabe destacar que este es un modelo simplificado y existen otras variables que también pueden influir en 
                el precio, como la proximidad al metro, la ubicación, la cercanía a comercios, entre otras.
    """)


    st.subheader("Aplicación: ")

   
    ### BOTON TIPO PROP ###

    tipo_propiedad = st.selectbox("Tipo de propiedad", ["Casa", "Departamento"])

    ### BOTON DORMITORIOS ###
    dormitorios = st.number_input("Número de Dormitorios", min_value=1, max_value=10)

    ### BOTON SUP UTIL ###
    superficie_util = st.number_input("Superficie Útil (m²)", min_value=1.0, value=60.0 ,max_value=1000.0)

    ### BAÑOS ###
    num_banos = st.number_input("Número de Baños", min_value=1, max_value=10)

    ### BOTON PREDICCION ###
    if st.button("Predecir Precio"):
        if tipo_propiedad == "Casa":
            # Cargar el modelo de casa
            model = joblib.load('reg_lineal_casa.joblib')
        else:
            # Cargar el modelo de departamento
            model = joblib.load('reg_lineal_depto.joblib')

        precio_predicho = model.predict([[superficie_util,dormitorios, num_banos]])[0]

        st.write(f"El precio predicho es: {precio_predicho:.2f} UF")

if __name__ == "__main__":
    main()