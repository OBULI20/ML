import os
import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as pximport 
import folium
from streamlit.components.v1 import html as components_html
from sklearn import preprocessing
from prediction import predict
import category_encoders as ce



df_adult_new = pd.read_excel('Cleaned_Census_Data.xlsx')
df_Undersample = pd.read_excel('Undersampled_Model.xlsx')

st.set_page_config(layout="wide")

# tab 1: EDA summary-2-3 graphs
# tab 2: Correlation plot, chloropeth map, any other insight related graph, show outliers
# tab 3: Scikit learn model

tab1, tab2, tab3, tab4, tab5,tab6,tab7,tab8 = st.tabs(["Feature Importance","Features","Correlation Matrix", "Key Feature","KNN", "Prediction Model","remarks", decision])

with tab1:
    sns.set_palette(sns.color_palette("Dark2", 8))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))
    sns.boxplot(x='education', y='age', hue='income', data=df_adult_new, palette='twilight', ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution by Education and Income')

    sns.boxplot(x='workclass', y='age', hue='workclass', data=df_adult_new, palette='Set3', ax=axes[1, 1])
    axes[1, 1].set_title('Age Distribution by Workclass')

    edu_income = df_adult_new.groupby('education')['income'].value_counts(normalize=True).unstack()
    edu_income.plot(kind='bar', stacked=True, color=['grey', 'purple'], ax=axes[1, 0])
    axes[1, 0].set_title('Income Levels by Education')
    axes[1, 0].set_xlabel('Education')
    axes[1, 0].legend(title='Income')

    workclass_income = df_adult_new.groupby('workclass')['income'].value_counts(normalize=True).unstack()
    workclass_income.plot(kind='bar', color=['grey', 'blue'], ax=axes[0, 1])
    axes[0, 1].set_title('Income Levels by Workclass')
    axes[0, 1].set_xlabel('Workclass')
    axes[0, 1].legend(title='Income')
    st.write("A quick analysis of the 4 most common influencers of income showed that this data set did not contain any of the usual markers of income. Education seems to have a mild positive influence on income, but in all other cases, there are no clear trends.")
    plt.tight_layout()  
    st.pyplot(plt.gcf())

with tab2:
    sns.set_palette(sns.color_palette("twilight", 8))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    sns.scatterplot(x='age', y='hours_per_week', hue='income', palette ='Purples',data=df_adult_new,ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution by Hours per Week and Income')

    sns.barplot(x='education', y='hours_per_week', hue='income', data=df_adult_new,palette ='Purples', ax=axes[1, 1])
    axes[1, 1].set_title('Hours per Week by Education and Income')

    sns.boxplot(x='income', y='age', data=df_adult_new,hue = 'sex',palette ='Purples', ax=axes[0, 1])
    axes[0, 1].set_title('Age vs Income')

    race_count= df_adult_new['race'].value_counts()
    axes[1,0].pie(race_count, labels=race_count.index, autopct ='%.2f%%',wedgeprops={'edgecolor': 'black'})
    axes[1, 0].set_title('Race Distribution')
    plt.tight_layout()  
    st.pyplot(fig)

with tab3:
    cat_col = df_adult_new.select_dtypes(include=['object']).columns
    num_col = df_adult_new.select_dtypes(exclude=['object']).columns
    encoder = ce.OrdinalEncoder()
    encoded_cat= encoder.fit_transform(df_adult_new[cat_col])
    combined_df = pd.concat([df_adult_new[num_col], encoded_cat], axis=1)
    correlation_matrix = combined_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='twilight',fmt=".2f", annot_kws={"size": 8})
    plt.title('Heatmap')
    st.pyplot(plt.gcf())



with tab4:
    st.write("We then used fnlwgt which is the number of people a variable represents to find per capita income and project density of incomes over 50K and under 50K by using a world map. What we saw was that almost anyone with income over 50K was in the US or Canada, even after adjusting for population size. This is therefore our key feature")
    df_adult_new['fnl_pct']=(df_adult_new['fnlwgt'] / df_adult_new['fnlwgt'].sum())*100    
    df_country_wt=df_adult_new.groupby(['nativecountry','income'])[['fnl_pct','fnlwgt']].sum().reset_index()
    df_country_wt['income'] = df_country_wt['income'].replace({'<=50K': 50000, '>50K': 10000})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'United-States': 'USA'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Dominican-Republic': 'Dominican Republic'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Holand-Netherlands': 'Netherlands'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Outlying-US(Guam-USVI-etc)': 'Puerto Rico'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'El-Salvador': 'El Salvador'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Trinadad&Tobago': 'Trinidad and Tobago'})
    df_country_wt['nativecountry'] = df_country_wt['nativecountry'].replace({'Puerto-Rico': 'Puerto Rico'})
    
    df_country_wt['Wtd_income']=df_country_wt['income']*df_country_wt['fnl_pct']
    
    bins = df_country_wt['fnlwgt'].quantile([0, 0.5, 0.7, 0.9, 1]).tolist()
    
    world_geo = r'world_countries.json'

    world_map = folium.Map(location=[0, 0], zoom_start=2, width='100%', height=800)  

    folium.Choropleth(
        geo_data=world_geo,
        data=df_country_wt,
        columns=['nativecountry', 'fnlwgt'],
        key_on='feature.properties.name',
        fill_color='RdYlGn',
        nan_fill_color="beige",
        fill_opacity=0.9,
        line_opacity=0.2,
        bins=bins).add_to(world_map)

    map_html = 'world_map.html'
    world_map.save(map_html)

    with open(map_html, 'r') as f:
        html_code = f.read()

    components_html(html_code, height=600)
with tab5:
    df_knn = pd.read_csv("MeanAccKNN.csv")

    values = df_knn['Value']
    serial_numbers = df_knn['Serial Number']

    fig1 = plt.figure()  
    plt.plot(serial_numbers, values, 'g', marker='o')
    plt.xlabel('Serial Number')
    plt.ylabel('Value')
    plt.title('Plot of Values vs Serial Numbers')
    plt.tight_layout()
    st.pyplot(fig1)

    st.write("We also tried imputing the missing values using KNN, after determining that 79% accuracy could be achieved using 12 neighbours. But 79% accuracy was already achieved through our original cluster as well")
    
with tab6:

    age1 = st.selectbox("Age", df_Undersample['age'].unique()) 
    workclass1 = st.selectbox("Workclass", df_Undersample['workclass'].unique()) 
    occupation1 = st.selectbox("Occupation", df_Undersample['occupation'].unique()) 
    hours_per_week1 = st.selectbox("Hours per Week", df_Undersample['hours_per_week'].unique())
    education1 = st.selectbox("Education", df_Undersample['education'].unique()) 
    nativecountry1 = st.selectbox("Native Country", df_Undersample['nativecountry'].unique()) 
    maritalstatus1 = st.selectbox("Marital Status", df_Undersample['marital_status'].unique()) 
    gender = st.selectbox("Gender", df_Undersample['sex'].unique()) 
    CG = st.selectbox("Capital Gains: Yes Or No", df_Undersample['CG_Category'].unique()) 


    
    if st.button("Predict Income"):
        encoder = ce.OrdinalEncoder(cols=['workclass', 'education', 'nativecountry', 'occupation','marital_status','sex'])
        data = {
            'workclass': [workclass1],
            'education': [education1],
            'nativecountry': [nativecountry1],
            'occupation': [occupation1],
            'marital_status':[maritalstatus1],
            'sex':[gender]

               }
        encoded_data = encoder.fit_transform(pd.DataFrame(data))
        input_data = np.concatenate((encoded_data.values[0], [hours_per_week1],[age1],[CG]))
        result = predict(np.array([input_data]))
        st.text(result[0])
with tab7:
    st.write(" Income depends on how your work-hard/smart")
