import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from proyecto1 import main as proyecto1_main
from proyecto2 import main as proyecto2_main
from proyecto3 import main as proyecto3_main
from proyecto4_st_optimized import main as proyecto4_st_optimized_main
import os

##########################################################################################################################
################################## PORTADA PORTAFOLIO ####################################################################
##########################################################################################################################


#### NOMBRE PESTAÑA:

st.set_page_config(page_title="Portafolio Virtual", page_icon=":computer:")

#### BARRA LATERAL:

with st.sidebar:
    pagina_actual = option_menu("Menu", ["Sobre Mí", 'Predicción Precio Propiedades',
                                        'Análisis de Datos', 'Similitud de Textos','Predicción Ventas Local - Categoría'], 
        icons=['house', 'book','book','book','book'], default_index=0)

if pagina_actual == "Sobre Mí":

    st.title("""¡Bienvenida/o a mi Portafolio Virtual!""")
    st.write('---')

    st.header('Sobre mi:')
    st.write("""
    Hola! Soy Valentina Vergara, Ingeniera Comercial con Máster en Business Analytics de la Universidad Adolfo Ibáñez. Actualmente, me desempeño como **Retail Account Executive** en UpTheTrade, gestionando la cuenta de **cinco retailers del canal moderno**, distribuidos en tres sectores: **supermercados**, **farmacias** y **tiendas de conveniencia**.  
             
    Mi objetivo principal es **transformar datos de mercado en insights accionables** que permitan a mis clientes tomar decisiones estratégicas para impulsar su crecimiento en el competitivo mercado minorista. Para lograrlo, utilizo herramientas digitales como **MS Excel** y **Tableau**, diseñando reportes personalizados y visualizaciones que facilitan la toma de decisiones basadas en datos. 
    Además, logré automatizar la carga de diversos reportes semanales y mensuales. 
             
    Mi experiencia profesional ha sido clave para fortalecer mi interés y habilidades en el análisis de datos:  
     + Durante mi práctica profesional y proyecto de título en Maxxa, desarrollé un modelo de *machine learning* para predecir la fuga mensual de clientes para un **Software de Gestión Empresarial**, además de automatizar envíos de KPI al gerente comercial y cargar bases de datos utilizando **Python** y **PostgreSQL**.  
     + En mi práctica intermedia en Rewchile, trabajé en el área de **marketing digital**, apoyando una campaña en Facebook Ads para mejorar el *engagement* con un *customer target* específico.  
             
     Estas experiencias me han demostrado el **impacto transformador del análisis de datos en las organizaciones**. Identificar patrones y tendencias permite no solo anticipar comportamientos del mercado, sino también optimizar recursos y generar ventajas competitivas clave.  
             
     Mi pasión por el análisis, el machine learning y la visualización de datos me impulsa a **seguir aprendiendo** y explorando cómo implementar herramientas digitales en distintos modelos de negocio. Estoy comprometida con aportar soluciones estratégicas que marquen la diferencia.      

    + **Manejo los siguientes lenguajes de programación:** Python, SQL, Tableau, R.  
    + **Principales Habilidades:**  
        + Modelos de predicción y clasificación con sklearn (python).  
        + Visualización de datos con matplotlib, seaborn, plotly express (python), Tableau y ggplot2 (R).  
        + Análisis de datos con pandas, numpy (python), data.table, tidyr y dplyr (R).  
        + Visualizaciones de aplicaciones, machine learning y paneles de control con streamlit (python).  
        + Visualización datos espaciales con geopandas (python), sp y rgdal (R).   

    A la izquierda podrás encontrar un menú con 4 proyectos.
    """)

elif pagina_actual == "Predicción Precio Propiedades":
    proyecto1_main()
elif pagina_actual == "Análisis de Datos":
    proyecto2_main()
elif pagina_actual == "Similitud de Textos":
    proyecto3_main()
elif pagina_actual == "Predicción Ventas Local - Categoría":
    proyecto4_st_optimized_main()

