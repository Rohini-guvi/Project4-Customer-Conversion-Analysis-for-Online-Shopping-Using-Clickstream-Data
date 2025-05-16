from matplotlib.colors import colorConverter
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class Kmean_cluster:
  def __init__(self,df):
    self.df = df

  def Predict(self):
    
    filename = '/content/drive/MyDrive/Project4-Clickstream/Kmeans_bs_model.sav'
    model = pickle.load(open(filename, 'rb'))
      
    self.df['Predicted_cluster'] = model.fit_predict(self.df)

    uni = self.df['Predicted_cluster'].to_list()
    uni = len(list(set(uni)))
    st.subheader(f"Total Clusters by K-Means algorithm:{uni}")

    return self.df

  def Plot_graph(self,df1):
      
    fig1= px.scatter_3d(df1,x= 'price',y= 'page2_clothing_model',z='price_2',color='Predicted_cluster')
    fig2= px.scatter_3d(df1,x= 'price',y= 'page1_main_category',z = 'price_2',color='Predicted_cluster')

    title_name = ["Clusters based on clothing model","Clusters based on Main category"]

    fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],column_titles=title_name
    )

    # Add first 3D scatter to first subplot
    fig.add_trace(fig1.data[0],row=1, col=1)

    # Add second 3D scatter to second subplot
    fig.add_trace(fig2.data[0],row=1, col=2)
    
    # Update layout as needed
    fig.update_layout(height=600, width=1000, title_text="KMeans Cluster plots")
    # Update layout to set axis titles
    fig.update_layout(
      scene1=dict(xaxis_title='price', yaxis_title='clothing_model', zaxis_title='price_2'),
      scene2=dict(xaxis_title='price', yaxis_title='main_category', zaxis_title='price_2'))
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True,theme='streamlit')

    return 

class DBSCAN_cluster:
  def __init__(self,df):
    self.df = df
    
  def Predict(self):  
    filename = '/content/drive/MyDrive/Project4-Clickstream/DB_scan_bs_model.sav'
    model = pickle.load(open(filename, 'rb'))

    self.df['Predicted_cluster'] = model.fit_predict(self.df)
 
    uni = self.df['Predicted_cluster'].to_list()
    if -1 in uni:
      uni = len(list(set(uni)))-1
    else:
      uni = len(list(set(uni)))

    st.subheader(f"Total Clusters by DBSCAN algorithm:{uni}")
    return self.df

  def Plot_graph(self,df1):

    fig1= px.scatter_3d(df1,x= 'price',y= 'page2_clothing_model',z='price_2',color='Predicted_cluster')
    fig2= px.scatter_3d(df1,x= 'price',y= 'page1_main_category',z = 'price_2',color='Predicted_cluster')

    title_name = ["Clusters based on clothing model","Clusters based on Main category"]

    fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],column_titles=title_name
    )

    # Add first 3D scatter to first subplot
    fig.add_trace(fig1.data[0],row=1, col=1)

    # Add second 3D scatter to second subplot
    fig.add_trace(fig2.data[0],row=1, col=2)
    
    # Update layout as needed
    fig.update_layout(height=600, width=1000, title_text="DBSCAN Cluster plots")
    
    # Update layout to set axis titles
    fig.update_layout(
      scene1=dict(xaxis_title='price', yaxis_title='clothing_model', zaxis_title='price_2'),
      scene2=dict(xaxis_title='price', yaxis_title='main_category', zaxis_title='price_2'))
    
    # Display in Streamlit
    st.plotly_chart(fig,use_container_width=True,theme='streamlit')

    return

st.subheader("**Prediction of Customer Clusters**")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
  delimiter = st.text_input("Enter delimiter (e.g., ',', ';', '\\t')",',')
  try:
    df = pd.read_csv(uploaded_file, delimiter=delimiter)
    st.write("Data Preview")
    st.dataframe(df)
  except pd.errors.ParserError:
    st.error("Error: Could not parse CSV with the given delimiter. Please check the delimiter and file format.")
  except Exception as e:
    st.error(f"An error occurred: {e}")

if 'clicked' not in st.session_state:
  st.session_state.clicked = False

def click_button():
  st.session_state.clicked = True

st.button('Submit', on_click=click_button)

if st.session_state.clicked:
  data = pd.read_csv("/content/drive/MyDrive/Project4-Clickstream/train_data.csv",delimiter=',')
    
  # Drop policy start date and id columns
  data.drop(["year","session_id"],axis=1,inplace=True)
  df.drop(["year","session_id"],axis=1,inplace=True)
  
  # Transformation of Numerical columns
  num_col = ['month', 'day', 'order', 'country', 'page1_main_category',
  'colour', 'location', 'model_photography', 'price', 'price_2', 'page']
  scaler = MinMaxScaler()
  data[num_col]=scaler.fit_transform(data[num_col])
  df[num_col] = scaler.transform(df[num_col])

  # Transformation of Categorical Columns
  le = LabelEncoder()
  data['page2_clothing_model'] = le.fit_transform(data['page2_clothing_model'])
  df['page2_clothing_model'] = le.fit_transform(df['page2_clothing_model'])

  kc = Kmean_cluster(df)

  df_kmean = kc.Predict()

  df_kmean['page2_clothing_model'] = le.inverse_transform(df_kmean['page2_clothing_model'])
  df_kmean[num_col] = scaler.inverse_transform(df_kmean[num_col])

  df_kmean['price_2'] = df_kmean['price_2'].map({1:"above_avg",2:"below_avg"})
  df_kmean['page1_main_category'] = df_kmean['page1_main_category'].map({1:'Trousers',2:'Skirts',3:'Blouses',4:'Sale'})
  kc.Plot_graph(df_kmean)

  df['price_2'] = df['price_2'].map({"above_avg":1,"below_avg":2})
  df['page1_main_category'] = df['page1_main_category'].map({'Trousers':1,'Skirts':2,'Blouses':3,'Sale':4})
  df[num_col] = scaler.transform(df[num_col])
  df['page2_clothing_model'] = le.fit_transform(df['page2_clothing_model'])
  
  dc = DBSCAN_cluster(df)
  df_dbscan = dc.Predict()
  df_dbscan['page2_clothing_model'] = le.inverse_transform(df_dbscan['page2_clothing_model'])
  df_dbscan[num_col] = scaler.inverse_transform(df_dbscan[num_col])

  df_dbscan['price_2'] = df_dbscan['price_2'].map({1:"above_avg",2:"below_avg"})
  df_dbscan['page1_main_category'] = df_dbscan['page1_main_category'].map({1:'Trousers',2:'Skirts',3:'Blouses',4:'Sale'})
  dc.Plot_graph(df_dbscan)

else:
  st.write('Kindly click Submit button to predict Customer clusters!')

