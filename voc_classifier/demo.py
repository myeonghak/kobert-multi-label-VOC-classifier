import streamlit as st

from inference import load_model

model=load_model()

bert=model.get_model()

bert.eval()


import pickle

raw=pickle.load(open("../data/raw_text.pkl","rb"))

data=pickle.load(open("../data/preprocessed_text.pkl","rb"))

def process_for_inference(num_of_text):
    raw_to_show=raw.text.iloc[num_of_text]
    text_input=data.text.iloc[num_of_text]
    input_text_label=data.iloc[num_of_text,1:].tolist()
    return raw_to_show, text_input, input_text_label


number_for_demo = st.text_input('태그를 생성할 텍스트의 번호를 입력해 주세요.')
if number_for_demo:
    num_of_text=int(number_for_demo)
    Raw,Input,Label=process_for_inference(num_of_text)
    for i in range(62,len(Raw),62):
        Raw=Raw[:i]+"\n"+Raw[i:]
    st.text(Raw)
    st.text("\n")
    st.text("==================예측 결과==================")
    st.text("\n")
    
    a,b=model.get_prediction_from_txt(Input,Label)
    st.text(a)
    st.text(b)