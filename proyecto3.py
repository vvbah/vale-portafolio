import streamlit as st
import pandas as pd
import os



def main():
    st.title('Modelo para Comparar Similitud entre Textos')

    st.write("""
    Hoy en día, los clientes suelen compartir su opinión tras comprar por el canal e-commerce, comentan sobre algún producto o servicio en redes 
    sociales , además, o incluso enviar mails a la empresa respecto a un tema específico, 
    entre otros. Para las empresas, toda esta información puede ofrecer *insights* valiosos que apoyen la 
    toma de decisiones y mejoren sus productos o servicios. Por ejemplo, si una organización recibe 1000 
    correos diarios de clientes expresando su nivel de satisfacción, sería muy lento y costoso para un trabajador revisar cada uno de ellos
    y clasificarlos manualmente. Sin embargo si esta tarea se automatiza, el tiempo 
    y recursos humanos se reducen, agregando valor a la empresa, pues cumplir el objetivo en poco tiempo *(incluso un par de minutos)* 
    permite tomar decisiones estratégicas rápidamente, basandose en los resultados.  


    """)

    st.write("""
    Todo lo anterior, es posible gracias al **Procesamiento del Lenguaje Natural** (NLP), una rama de Machine Learning que 
    permite a las computadoras comprender el lenguaje humano para agilizar procesos específicos. Para esto, es necesario 
    transformar la data no estructurada, ya sean textos o mensajes de voz, en representaciones matemáticas que puedan ser 
    procesadas por algoritmos. Esto se logra extrayendo cada palabra de los documentos en forma de *“tokens”* que luego 
    se convierten en vectores que representan los textos.  


    """)

    st.write("""
    Para entender cómo se realiza la transformación vectorial de los textos, primero es importante comprender lo que ocurre con las palabras 
    en sí. Un enfoque inicial es estudiar su morfología, es decir, la estructura interna que descompone las palabras en 
    unidades más pequeñas (morfema) para comprenderlas desde su raíz. Luego estos términos se “lematizan”, ósea, se convierten 
    en su forma más simple y válida, por ejemplo, el lema de “tomaste” es “tomar”. Después, estas se transforman en expresiones 
    regulares: Un conjunto de símbolos que se pueden concatenar, unir y/o repetir para que sean entendidas 
    por las computadoras como un “lenguaje regular”. Estos caracteres permiten generalizar las palabras 
    *(lenguaje expresiones regulares - referencia rápida: https://learn.microsoft.com/es-es/dotnet/standard/base-types/regular-expression-language-quick-reference).*  


    """)

    st.write(r"""
    Entendiendo como se preprocesan los términos de los documentos, se puede comprender como se representa esta información en forma vectorial. 
    Como se mencionó, para que un modelo de Machine Learning pueda ser entrenado es necesario tener la data representada 
    en numéricamente, pero ¿cómo se pueden transformar textos en números? Primero, se debe considerar que cada 
    documento ($d_i$) tiene N palabras ($w_i$) y a partir de los términos únicos se crea el vocabulario ($V_i$) del texto:  
    """)  

    col1, col2, col3 = st.columns(3)

    with col2:
        st.write(r"""
        $d_i = (w_1, w_2, w_3, w_4, w_5)$  
        $d_i=(a,b,a,a,c)$  
        $V_i = (a,b,c)$
        """)

    st.write(r"""
    Ahora, para transformar un documento en forma vectorial, se toman todas sus palabras como se muestra en $d_i$ y 
    se eliminan las *stopwords*, ósea, todos los términos que no son relevante para la comprensión del texto general 
    (por ejemplo: el, la, un, nos, entre otros), y se procesan los datos hasta que queden en lemas. Luego, se les asigna un peso a los términos y para esto, se usa el método **TF x IDF**:  
    """)

    st.write(f"""
    + **TF (Term Frequency)**: Entrega la cantidad de veces que se encuentra un término en el documento. Suponiendo que 
    W representa el peso, i los términos y j los documentos, se tiene que:  
    """)

    col1, col2, col3 = st.columns(3)

    with col2:
        st.write(r"$TF(i,j)=W(i,j)$")

    st.write("""
    + **IDF (Inverse Document Frequency)**: Es un ponderador que mide que tan común es una palabra dentro del vocabulario. 
    Teniendo N como el número total de documentos y df(i) como el número de documentos en donde ocurre el i-ésimo 
    término:  
    """)

    col1, col2, col3 = st.columns(3)

    with col2:
        st.write(r"$IDF =log \frac{N}{df(i)}$")

    st.write("""
    Finalmente, al multiplicar ambos valores (TF y IDF), se asigna un peso representativo a cada término presente en los documentos 
    con respecto al vocabulario total. Esto, permite representar los documentos como vectores, que pueden compararse usando 
    medidas de similitud. Los documentos cuya distancia en un espacio 
    multidimensional (compuesto por las palabras del vocabulario) sea menor, son los más similares entre sí, ósea, los que tengan un valor 
    más cercano a 1 son más similares.  


    """)

    st.subheader("Aplicación TFxIDF")  

    st.write("""
    En Chile, los discursos de los presidentes suelen ser almacenados en https://prensa.presidencia.cl/discursos.aspx, 
    y durante el 2021 se pensaba que muchos de los discursos ya habían sido escuchados antes. Existen más 700 discursos 
    para ser analizados y buscar alguna similitud en ellos, pero realizar esta tarea a mano puede demorar mucho tiempo. 
    Dado esto, se usó Python para buscar la similitud entre pares de discursos, en donde se aplicó todo lo mencionado 
    antes. Como resultado, se obtiene la tabla que se muestra a continuación con el porcentaje de similitud entre ambos 
    discursos:
    """)

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'textos.csv')
    df = pd.read_csv(file_path) 



    porcentaje_similitud = st.slider('Porcentaje de Similitud', min_value=0.0, max_value=100.0, value=70.0, step=1.0)

    
    resultados_filtrados = df[df['similarity'] >= porcentaje_similitud/100.0] 

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(resultados_filtrados [["pair", "similarity"]])
    
    with col2:
        st.write(f"Se encontraron {resultados_filtrados.shape[0]} pares de discursos con una similitud mayor a {porcentaje_similitud}%")

if __name__ == "__main__":
    main()