import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import uuid

def load_data():
    """Carga y preprocesa el dataset de ventas."""
    try:
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'sales.csv')
        df = pd.read_csv(file_path)
        
        # Preprocesamiento
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['año'] = df['Date'].dt.year
        df['mes'] = df['Date'].dt.month
        df['semana'] = df['Date'].dt.isocalendar().week.astype(int)
        df['IsHoliday'] = df['IsHoliday'].apply(lambda x: 1 if x == 'True' else 0)
        df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
        df.rename(columns={'Store': 'local', 'Dept': 'categoria', 'Weekly_Sales': 'ventas', 'Sale Id': 'id_venta', 'Date': 'fecha'}, inplace=True)
        df = df[df['ventas'] > 0].sort_values(by='fecha').reset_index(drop=True)
        
        return df
    except FileNotFoundError:
        st.error("Error: El archivo 'sales.csv' no se encuentra en el directorio.")
        return None
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

def plot_weekly_sales_trend(df):
    """Genera gráfico de tendencias de ventas semanales."""
    ventas_sem = df.groupby('fecha')['ventas'].mean().reset_index()
    fig = px.line(ventas_sem, x='fecha', y='ventas', title='Tendencia de Ventas Semanales (Promedio)')
    fig.update_layout(xaxis_title='Fecha', yaxis_title='Ventas Promedio ($)', template='plotly_white')
    st.plotly_chart(fig)

def plot_yearly_weekly_sales(df):
    """Genera un gráfico de ventas semanales promedio por año con un eje x del 1 al 52."""
    ventas_sem = df.groupby(['año', 'semana'])['ventas'].mean().reset_index()
    ventas_sem = ventas_sem[ventas_sem['semana'].between(1, 52)].sort_values(['año', 'semana'])
    fig = px.line(ventas_sem, x='semana', y='ventas', color='año', 
                  title='Ventas Semanales Promedio por Año (Semanas 1-52)', 
                  markers=True,
                  labels={'semana': 'Semana', 'ventas': 'Ventas Promedio ($)', 'año': 'Año'})
    fig.update_xaxes(range=[1, 52], tickmode='linear', dtick=1)
    fig.update_layout(template='plotly_white', xaxis_title='Semana', yaxis_title='Ventas Promedio ($)')
    st.plotly_chart(fig)

def plot_category_sales_share(df):
    """Genera gráfico de participación de ventas por categoría."""
    ventas_cat = df.groupby('categoria')['ventas'].sum().reset_index()
    ventas_cat['categoria'] = ventas_cat['categoria'].astype(str)
    ventas_cat['peso'] = round((ventas_cat['ventas'] / ventas_cat['ventas'].sum()) * 100, 2)
    ventas_cat = ventas_cat.sort_values(by='ventas', ascending=False)
    fig = px.bar(ventas_cat, x='categoria', y='ventas', text='peso', title='Participación de Ventas por Categoría (%)')
    fig.update_xaxes(type='category')
    fig.update_traces(texttemplate='%{text}%', textposition='auto')
    st.plotly_chart(fig)

def plot_category_weekly_sales(df, period='all'):
    """Genera gráfico de ventas promedio semanales por categoría."""
    if period == '2011':
        df = df[df['año'] == 2011].copy()
        title = 'Ventas Promedio Semanales por Categoría (Año 2011)'
    else:
        title = 'Ventas Promedio Semanales por Categoría (Histórico)'
    
    ventas_cat = df.groupby('categoria')['ventas'].mean().reset_index()
    ventas_cat['categoria'] = ventas_cat['categoria'].astype(str)
    ventas_cat['ventas_app'] = round(ventas_cat['ventas'])
    ventas_cat = ventas_cat.sort_values(by='ventas', ascending=False)
    
    fig = px.bar(ventas_cat, x='categoria', y='ventas_app', title=title, labels={'ventas_app': 'Ventas Promedio ($)'})
    fig.update_xaxes(type='category')
    st.plotly_chart(fig)

def plot_top_categories_by_store(df):
    """Genera gráfico interactivo de las 10 categorías principales por local."""
    cat_local = df.groupby(['local', 'categoria'])['ventas'].sum().reset_index()
    ventas_local_total = cat_local.groupby('local')['ventas'].sum().reset_index(name='ventas_local')
    cat_local = cat_local.merge(ventas_local_total, on='local')
    cat_local['peso'] = round((cat_local['ventas'] / cat_local['ventas_local']) * 100, 2)
    
    local = st.selectbox('Selecciona un local:', cat_local['local'].unique(), key='top_categories')
    top_10 = cat_local[cat_local['local'] == local].nlargest(10, 'peso')
    top_10['peso_label'] = top_10['peso'].round(2).astype(str) + '%'
    
    fig = px.pie(top_10, values='peso', names='categoria', hole=0.2, title=f'Top 10 Categorías - Local {local}')
    fig.update_traces(text=top_10['peso_label'], textinfo='text', 
                      customdata=top_10[['ventas']].values, 
                      hovertemplate='Categoría: %{label}<br>Peso: %{value:.2f}%<br>Ventas: %{customdata[0]:,}')
    st.plotly_chart(fig)

def plot_seasonal_decomposition(df):
    """Genera gráfico de descomposición estacional usando Plotly."""
    ventas_sem = df.groupby('fecha')['ventas'].sum().reset_index()
    descomp = seasonal_decompose(ventas_sem['ventas'], model='additive', period=52)
    
    decomp_df = pd.DataFrame({
        'Fecha': ventas_sem['fecha'],
        'Observado': descomp.observed,
        'Tendencia': descomp.trend,
        'Estacionalidad': descomp.seasonal,
        'Residuos': descomp.resid
    })
    
    fig = px.line(decomp_df, x='Fecha', y=['Observado', 'Tendencia', 'Estacionalidad', 'Residuos'],
                  title='Descomposición Estacional de Ventas')
    fig.update_layout(yaxis_title='Ventas ($)', template='plotly_white')
    st.plotly_chart(fig)

def plot_correlation_matrix(df):
    """Genera matriz de correlación con Plotly."""
    correlation_matrix = df.drop('IsHoliday', axis=1).corr()
    fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        colorscale='RdBu',
        zmin=-1, zmax=1,
        annotation_text=correlation_matrix.round(2).values
    )
    fig.update_layout(title='Matriz de Correlación', template='plotly_white')
    st.plotly_chart(fig)

def train_and_evaluate_model():

    current_dir = os.path.dirname(__file__)

    file_path = os.path.join(current_dir, 'y_test.csv')
    y_test = pd.read_csv(file_path)

    file_path = os.path.join(current_dir, 'y_pred_rf.csv')
    y_pred_rf = pd.read_csv(file_path)
    
    st.write(f"""
    **Resultados del Modelo Random Forest**  
    - **Error Cuadrático Medio (MSE):** $US11402401    
    - **R² (coeficiente de determinación):** 0.9768    
    El modelo explica el 97.68% de la variabilidad de las ventas.
    """)
    
    fig_rf = px.scatter(x=y_test['ventas'], y=y_pred_rf['ventas'], trendline="ols", 
                        title='Random Forest: Valores Reales vs. Predichos',
                        labels={'x': 'Valores Reales ($)', 'y': 'Valores Predichos ($)'})
    st.plotly_chart(fig_rf)
    


def plot_sales_forecast(combined_df_rf, local, categoria):
    """Genera gráfico de ventas reales vs. predichas para un local y categoría."""
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'combined_df_rf.csv')
    combined_df_rf = pd.read_csv(file_path)
    sample_rf = combined_df_rf[(combined_df_rf['local'] == local) & (combined_df_rf['categoria'] == categoria)]
    fig_rf = px.line(sample_rf, x='fecha', y='ventas', color='tipo', 
                     title=f'Ventas Reales vs. Predichas - Local {local}, Categoría {categoria}',
                     labels={'fecha': 'Fecha', 'ventas': 'Ventas Semanales ($)', 'tipo': 'Tipo'})
    fig_rf.update_layout(template='plotly_white')
    st.plotly_chart(fig_rf)

def main():
    """Función principal de la aplicación Streamlit."""
    st.title("Predicción de Ventas por Local y Categoría para una Cadena de Supermercados")
    
    st.markdown("""
    ### Contexto del Negocio
    En el competitivo sector *retail* de supermercados, factores como la estacionalidad, los feriados y las tendencias económicas influyen significativamente en el desempeño. Las cadenas enfrentan el desafío de optimizar recursos, gestionar inventarios y mejorar la experiencia del cliente para maximizar ingresos y garantizar la rentabilidad.

    **Problema Identificado**: La asignación ineficiente de recursos entre locales limita el crecimiento y afecta el presupuesto operativo, ya que se priorizan locales con menor aporte al negocio.

    **Solución Propuesta**: Desarrollar un modelo predictivo basado en *machine learning* para estimar las ventas semanales por local y categoría durante las próximas 52 semanas. Esto permite anticipar la demanda, optimizar inventarios, diseñar estrategias de marketing efectivas y reducir costos operativos.

    **Dataset**: Se utiliza un conjunto de datos históricos de 45 locales en EE. UU., con las siguientes variables:
    - `local`: Identificador de la tienda (1 a 45).
    - `categoria`: Categoría del producto (81 valores únicos).
    - `fecha`: Fecha de la venta (para derivar semana y año).
    - `ventas`: Ventas semanales por categoría y local.
    - `IsHoliday`: Indicador de feriados (1: sí, 0: no).  
                
    Este dataset permite modelar patrones temporales, estacionalidades, comportamiento diferenciado por local y categoría, así como también
    el impacto de eventos especiales. Para comenzar, es necesario análizar con mayor profundidad los datos.
    """)
    
    st.markdown("### Análisis Exploratorio de Datos (EDA)")
    
    df = load_data()
    if df is None:
        return
    
    st.markdown("#### Tendencia de Ventas Semanales")
    plot_weekly_sales_trend(df)
    st.markdown("""Podemos notar un alza en las ventas en dos fechas importantes para EEUU: Día de Acción de Gracias (27/10) y 
    Navidad (25/12), pero esto ¿se repite todos los años?         
    """)
    
    st.markdown("#### Comparación de Ventas Semanales por Año")
    plot_yearly_weekly_sales(df)
    st.markdown("Se observa una estacionalidad consistente en las alzas de ventas durante los feriados, aunque los datos de 2012 son incompletos.")

    st.markdown("#### Participación de Ventas por Categoría")
    plot_category_sales_share(df)
    st.markdown("La categoría 92 lidera con un 7.9% de las ventas totales, seguida por la categoría 95 con un 6.6% (período 2010-2012).")
    
    st.markdown("#### Ventas Promedio Semanales por Categoría")
    plot_category_weekly_sales(df, period='all')
    plot_category_weekly_sales(df, period='2011')
    st.markdown("La categoría 92 registra un promedio de $75,192 semanales, manteniendo su liderazgo en 2011.")
    
    st.markdown("#### Top 10 Categorías por Local")
    st.markdown("¿Te gustaría saber la venta de las 10 categorías mas importantes por local? Te invito a interactuar con el siguiente gráfico:")
    plot_top_categories_by_store(df)
    
    st.markdown("#### Análisis de Estacionalidad")
    st.markdown("""
    Se aplicó un modelo aditivo para descomponer las ventas en tendencia, estacionalidad y residuos:
    - **Tendencia**: Crecimiento sostenido y lento en las ventas a lo largo del tiempo.
    - **Estacionalidad**: Alzas recurrentes en Día de Acción de Gracias y Navidad (período de 52 semanas).
    - **Residuos**: Las alzas estacionales no son completamente capturados por un modelo lineal, sugiriendo la necesidad de modelos no lineales, pues se observa como el error se aleja de 0 cercano a las festividades.
    """)
    plot_seasonal_decomposition(df)
    
    st.markdown("#### Análisis de Correlación")
    st.markdown("""
    La matriz de correlación revela:
    - Alta correlación entre `local` e `id_venta` (se elimina `id_venta` para evitar multicolinealidad).
    - Las variables `año` y `semana` se mantienen para capturar patrones temporales.
    """)
    plot_correlation_matrix(df)
    
    st.markdown("### Modelo Predictivo")
    st.markdown("""
    Se entrenó un modelo Random Forest para predecir las ventas semanales, utilizando las variables `local`, `categoria`, `IsHoliday`, `año` y `semana`. 
    La fecha de corte fue la semana 6 de 2012, con un 73.43% de los datos para entrenamiento y un 26.57% para prueba.
    """)
    
    train_and_evaluate_model()
    
    st.markdown("### Predicciones a Futuro")
    st.markdown("El modelo Random Forest captura la tendencia al alza y la estacionalidad, permitiendo proyecciones precisas para las próximas 52 semanas.")
    
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'combined_df_rf.csv')
    combined_df_rf = pd.read_csv(file_path)
    
    st.markdown("#### Visualización de Predicciones")
    st.markdown("Selecciona un local y categoría para comparar las ventas reales y predichas.")
    local = st.selectbox('Selecciona un local:', combined_df_rf['local'].unique(), key='forecast_local')
    categoria = st.selectbox('Selecciona una categoría:', combined_df_rf['categoria'].unique(), key='forecast_categoria')
    plot_sales_forecast(combined_df_rf, local, categoria)
    
    st.markdown("""
    ### Conclusión
    Este proyecto demuestra cómo un modelo predictivo basado en *machine learning* puede anticipar la demanda futura, optimizar la asignación de recursos y respaldar decisiones estratégicas en el sector *retail*. La capacidad de identificar patrones estacionales y tendencias permite a las cadenas de supermercados mejorar la gestión de inventarios, diseñar promociones efectivas y aumentar la rentabilidad.
    """)

if __name__ == "__main__":
    main()