import streamlit as st
import pandas as pd
import numpy as np
import pickle

class DataPreprocessing:

  def Mapping(self,df):
    
    # Drop policy start date and id columns
    df.drop(["year","session_id"],axis=1,inplace=True)

    # Mapping of categorical variables
    df['month'] = df['month'].map({'April':4,'May':5,'June':6,'July':7,'August':8})
    df['page1_main_category'] = df['page1_main_category'].map({'Trousers':1, "Skirts":2, "Blouses":3, 'Sale':4})
    df['colour'] = df['colour'].map({'beige':1,'black':2,'blue':3,'brown':4,'burgundy':5,'gray':6,'green':7,
    'navy blue':8,'many colors':9,'olive':10,'pink':11,'red':12,'violet':13,'white':14})
    df['location'] = df['location'].map({'top left':1, "top in the middle":2, 'top right':3,'bottom left':4,'bottom in the middle':5,'bottom right':6})
    df['model_photography'] = df['model_photography'].map({'en face':1, "profile":2})
    df['price_2'] = df['price_2'].map({"yes":1,"no":2})
    df['country'] = df['country'].map({'Australia':1,'Austria':2,'Belgium':3,"British Virgin Islands":4,"Cayman Islands":5,"Christmas Island":6,'Croatia':7,'Cyprus':8,
    "Czech Republic":9,'Denmark':10,'Estonia':11,'unidentified':12,"Faroe Islands":13,'Finland':14,'France':15,'Germany':16,'Greece':17,'Hungary':18,
    'Iceland':19,'India':20,'Ireland':21,'Italy':22,'Latvia':23,'Lithuania':24,'Luxembourg':25,'Mexico':26,'Netherlands':27,'Norway':28,'Poland':29,'Portugal':30,
    'Romania':31,'Russia':32,'San Marino':33,'Slovakia':34,'Slovenia':35,'Spain':36,'Sweden':37,'Switzerland':38,'Ukraine':39,'United Arab Emirates':40,
    "United Kingdom":41,'USA':42,"biz (.biz)":43,"com (.com)":44,"int (.int)":45,"net (.net)":46,"org (''*''.org)":47})

    return df

  def Prediction(self,df):
    df = self.Mapping(df)

    filename = '/content/drive/MyDrive/Project4-Clickstream/XGB_regbs_model.sav'
    model = pickle.load(open(filename, 'rb'))

    output = model.predict(df)
    return output

st.subheader("**Prediction of the Price of Product purchased**")

year = st.selectbox("Enter the Year of Purchase",("2008"))
month = st.selectbox("Enter the Month of Purchase",("April","May","June","July","August"))
day = st.selectbox("Enter the Day of Purchase",(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
st.caption("Months-April and June have 30 days. Months-May,July and August have 31 days")
order = st.text_input("Enter Number of Orders you made")
country = st.selectbox("Select your Country",('Australia','Austria','Belgium',"British Virgin Islands","Cayman Islands","Christmas Island",'Croatia','Cyprus',
"Czech Republic",'Denmark','Estonia','unidentified',"Faroe Islands",'Finland','France','Germany','Greece','Hungary',
'Iceland','India','Ireland','Italy','Latvia','Lithuania','Luxembourg','Mexico','Netherlands','Norway','Poland','Portugal',
'Romania','Russia','San Marino','Slovakia','Slovenia','Spain','Sweden','Switzerland','Ukraine','United Arab Emirates',
"United Kingdom",'USA',"biz (.biz)","com (.com)","int (.int)","net (.net)","org (''*''.org)"))
sess_id = st.text_input("Enter your Session ID")
main_cat = st.selectbox("Select your Category",('Trousers', "Skirts", "Blouses", 'Sale'))
cloth_model = st.selectbox("Select your Clothing model",('C20',
'B26',
'C13',
'B11',
'B31',
'C38',
'C24',
'C45',
'B24',
'A11',
'P39',
'P18',
'P16',
'P11',
'A3',
'P1',
'A13',
'C26',
'B17',
'A7',
'C12',
'A2',
'P2',
'P4',
'C18',
'P3',
'P43',
'C41',
'C10',
'C25',
'P60',
'P77',
'C33',
'A10',
'B34',
'P8',
'A25',
'A6',
'B10',
'P12',
'A30',
'C14',
'C19',
'C40',
'A8',
'A21',
'A22',
'A5',
'C11',
'A16',
'A29',
'B20',
'C5',
'P55',
'P80',
'P51',
'B25',
'C35',
'C2',
'C17',
'P14',
'P5',
'A39',
'C7',
'P20',
'P67',
'P49',
'P15',
'C44',
'A14',
'C9',
'P57',
'P7',
'A1',
'A38',
'B2',
'P25',
'B27',
'P10',
'P72',
'B32',
'A33',
'P17',
'C54',
'C56',
'B4',
'A4',
'C27',
'A15',
'C4',
'A17',
'A41',
'P62',
'A35',
'P48',
'C46',
'C6',
'A18',
'A37',
'A12',
'P26',
'P63',
'B14',
'C15',
'P40',
'A36',
'B15',
'P34',
'A42',
'C55',
'B21',
'P61',
'C8',
'A9',
'P33',
'B8',
'B23',
'B1',
'B13',
'C53',
'P29',
'C16',
'B6',
'P73',
'C50',
'B16',
'A20',
'P42',
'P74',
'P35',
'A31',
'A26',
'B30',
'P50',
'A28',
'A32',
'C59',
'P75',
'P70',
'C48',
'P47',
'C58',
'P6',
'C51',
'A27',
'P68',
'C21',
'P38',
'C32',
'C30',
'P23',
'P9',
'P19',
'P65',
'C23',
'B29',
'B28',
'B19',
'C34',
'C49',
'C57',
'P64',
'B7',
'C52',
'P44',
'P71',
'P59',
'A23',
'P82',
'P36',
'B12',
'B33',
'B9',
'C1',
'P32',
'C42',
'C36',
'P30',
'P37',
'C43',
'C39',
'P56',
'B3',
'A34',
'P76',
'B22',
'A43',
'C3',
'P13',
'B5',
'C28',
'A40',
'C22',
'C47',
'C29',
'P24',
'A24',
'P58',
'A19',
'P53',
'C37',
'P46',
'P69',
'C31',
'P45',
'P52',
'P78',
'P21',
'P81',
'P41',
'P66',
'P27',
'P31',
'P79',
'P22',
'P54'))
color = st.selectbox("Select Color of the Product",('beige','black','blue','brown','burgundy','gray',
'green','navy blue','many colors','olive','pink','red','violet','white'))
loc = st.selectbox("Select Product Location",('top left', "top in the middle", 'top right','bottom left','bottom in the middle','bottom right'))
photo = st.selectbox("Select Product Model Photography",('en face', "profile"))
pref = st.selectbox("Select whether your product price is higher than average price",("yes","no"))
page = st.selectbox("Select your Product Page Number",(1,2,3,4,5))

if 'clicked' not in st.session_state:
  st.session_state.clicked = False

def click_button():
  st.session_state.clicked = True

st.button('Submit', on_click=click_button)

if st.session_state.clicked:

  try:
    data=pd.DataFrame({'year' : [year],'month' : [month],'day': [day],'order':[order],
                       'country' : [country], 'session_id' : [sess_id ],'page1_main_category' : [main_cat],
                       'page2_clothing_model': [cloth_model],'colour': [color ],'location': [loc],
                       'model_photography':[photo],'price_2': [pref ],'page': [page]})
  except:
    st.write('After changing the details,Kindly click Submit button to predict your Mental Health!')

  dc = DataPreprocessing()
  output = dc.Prediction(data).astype(int)

  st.subheader(f"Predicted Price of the Purchase is {output[0]} US Dollars")

else:
  st.write('Kindly click Submit button to predict your Purchase Price!')

