import pandas as pd
import numpy as np

import os
import re

import config

DATA_PATH=config.DATA_PATH



def remove_linebreak(string):
    return string.replace('\r',' ').replace('\n',' ')

def split_table_string(string):
    trimmedTableString=string[string.rfind("PNR :"):]
    string=string[:string.rfind("PNR :")]
    return (string, trimmedTableString)

def remove_multispace(x):
    x = str(x).strip()
    x = re.sub(' +', ' ',x)
    return x





class preprocess:
    
    def __init__(self):
        self.voc_total=pd.read_excel(DATA_PATH,sheet_name=None, engine='openpyxl')

    
    def make_table(self):
        
        voc_data=self.voc_total["종합본"]

        DATA_PATH=config.DATA_PATH

        voc_total=pd.read_excel(DATA_PATH,sheet_name=None, engine='openpyxl')

        voc_data=voc_total["종합본"]

        voc_data=voc_data.drop(["Unnamed: 11","Unnamed: 12"],axis=1)

        voc_data.columns=["case_no","sender_type","message","cat1","cat2","cat3","cat4","cat5","cat6","cat7","cat8"]

        case_start_idx=voc_data.loc[:,"case_no"][voc_data.loc[:,"case_no"].notnull()].index.tolist()

        voc_data.loc[case_start_idx,"seq"]=1

        nan_val=voc_data.iloc[2][0]

        result=[]
        case_no=False
        cnt_case=0
        for _,txt in voc_data.iterrows():
            if (cnt_case==0)&(txt.seq==1):
                new_chat_corpus=[]

                cat1=txt.cat1
                cat2=txt.cat2
                cat3=txt.cat3
                cat4=txt.cat4
                cat5=txt.cat5
                cat6=txt.cat6
                cat7=txt.cat7
                cat8=txt.cat8

                new_chat_corpus.append([txt.sender_type,txt.message])
                cnt_case+=1

            elif txt.seq==1:

                result.append([new_chat_corpus,cat1,cat2,cat3,cat4,cat5,cat6,cat7,cat8])

                cnt_case+=1
                # 기존의 말뭉치 append
                # 새로운 말뭉치 구분 시작
                new_chat_corpus=[]

                cat1=txt.cat1
                cat2=txt.cat2
                cat3=txt.cat3
                cat4=txt.cat4
                cat5=txt.cat5
                cat6=txt.cat6
                cat7=txt.cat7
                cat8=txt.cat8

                new_chat_corpus.append([txt.sender_type,txt.message])

            else:
                new_chat_corpus.append([txt.sender_type,txt.message])


        total_result=[]    

        for i in range(len(result)):
            talk=" ".join([str(txt[1]) for txt in result[i][0]])
            label=result[i][-8:]
            total_result.append([talk,label])



            result_data=pd.DataFrame(total_result)

        label_total=result_data[1]

        result_data[["label_cat1","label_cat2","label_cat3","label_cat4","label_cat5","label_cat6","label_cat7","label_cat8"]]=pd.DataFrame(result_data[1].tolist(), index=result_data.index)

        result_data=result_data.drop(1,axis=1)

        result_data.columns=["text","label_cat1","label_cat2","label_cat3","label_cat4","label_cat5","label_cat6","label_cat7","label_cat8"]

        result_data=result_data[result_data["label_cat1"].isna()==False]

        result_data.label_cat1=result_data.label_cat1.fillna("결측")
        result_data.label_cat1=result_data.label_cat1.apply(str.strip)
        result_data.label_cat2=result_data.label_cat2.fillna("결측")
        result_data.label_cat2=result_data.label_cat2.apply(str.strip)
        result_data.label_cat3=result_data.label_cat3.fillna("결측")
        result_data.label_cat3=result_data.label_cat3.apply(str.strip)
        result_data.label_cat4=result_data.label_cat4.fillna("결측")
        result_data.label_cat4=result_data.label_cat4.apply(str.strip)
        result_data.label_cat5=result_data.label_cat5.fillna("결측")
        result_data.label_cat5=result_data.label_cat5.apply(str.strip)
        result_data.label_cat6=result_data.label_cat6.fillna("결측")
        result_data.label_cat6=result_data.label_cat6.apply(str.strip)
        result_data.label_cat7=result_data.label_cat7.fillna("결측")
        result_data.label_cat7=result_data.label_cat7.apply(str.strip)
        result_data.label_cat8=result_data.label_cat8.fillna("결측")
        result_data.label_cat8=result_data.label_cat8.apply(str.strip)

        result_data.label_cat1=result_data.label_cat1.apply(str.replace, args=(" ",""))
        result_data.label_cat2=result_data.label_cat2.apply(str.replace, args=(" ",""))
        result_data.label_cat3=result_data.label_cat3.apply(str.replace, args=(" ",""))
        result_data.label_cat4=result_data.label_cat4.apply(str.replace, args=(" ",""))
        result_data.label_cat5=result_data.label_cat5.apply(str.replace, args=(" ",""))
        result_data.label_cat6=result_data.label_cat6.apply(str.replace, args=(" ",""))
        result_data.label_cat7=result_data.label_cat7.apply(str.replace, args=(" ",""))
        result_data.label_cat8=result_data.label_cat8.apply(str.replace, args=(" ",""))

        self.table=result_data
        return True

    def label_process(self, num_labels=17 ,except_labels=None):
        result_data=self.table

        total_label_cases=set(result_data.label_cat1.unique().tolist())|set(result_data.label_cat2.unique().tolist())|set(result_data.label_cat3.unique().tolist())|\
        set(result_data.label_cat4.unique().tolist())|set(result_data.label_cat5.unique().tolist())|set(result_data.label_cat6.unique().tolist())|\
        set(result_data.label_cat7.unique().tolist())|set(result_data.label_cat8.unique().tolist())

        total_label_cases=list(total_label_cases)

        final_label=[]

        for _,txt in result_data.iterrows():
            label_sum="|"+txt.label_cat1+"|"+txt.label_cat2+"|"+txt.label_cat3+"|"+txt.label_cat4+"|"+txt.label_cat5+"|"+txt.label_cat6+"|"+txt.label_cat7+"|"+txt.label_cat8
            final_label.append([_,label_sum])

        total_label_cases_dict={}

        for col in total_label_cases:
            total_label_cases_dict[col]=len([i[0] for i in final_label if f"|{col}|" in i[1]])

        label_cases_dict_cnt=[case for case in {k: v for k, v in sorted(total_label_cases_dict.items(), key=lambda item: item[1], reverse=True)}.items()]

        label_cases_sorted=[case[0] for case in {k: v for k, v in sorted(total_label_cases_dict.items(), key=lambda item: item[1], reverse=True)}.items()]
        
        if except_labels:
            for label in except_labels:
                label_cases_sorted.remove(label)
                
        label_cases_sorted_target=label_cases_sorted[1:num_labels+1]

        label_cols=label_cases_sorted_target
        self.label_cols=label_cases_sorted_target
        
        for col in label_cases_sorted_target:
            result_data.loc[[i[0] for i in final_label if f"|{col}|" in i[1]], col]=1
            result_data[col]=result_data[col].fillna(0)
            result_data[col]=result_data[col].astype(int)


        result_data=result_data.drop(["label_cat1","label_cat2","label_cat3","label_cat4","label_cat5","label_cat6","label_cat7","label_cat8"],axis=1)
        # 단 하나의 답도 없는 경우에는 일단 제거

        result_data=result_data[(result_data.iloc[:,1:].eq(0).sum(axis=1)!=num_labels)]

        self.data=result_data
        
        return True
    
