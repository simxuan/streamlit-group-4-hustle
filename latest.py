#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from boruta import BorutaPy
from tqdm import tqdm_notebook, tqdm
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LinearRegression
#import warnings
#warnings.filterwarnings('ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)
# @cache
@st.cache

#cases_malaysia dataset
def load_data(n_rows):
    cases_malaysia = pd.read_csv("cases_malaysia.csv")
    cases_malaysia.isna().sum()
    cases_malaysia = cases_malaysia.fillna(0)
    cases_malaysia['cluster_import'] = cases_malaysia['cluster_import'].astype('int64')
    cases_malaysia['cluster_religious'] = cases_malaysia['cluster_religious'].astype('int64')
    cases_malaysia['cluster_community'] = cases_malaysia['cluster_community'].astype('int64')
    cases_malaysia['cluster_highRisk'] = cases_malaysia['cluster_highRisk'].astype('int64')
    cases_malaysia['cluster_education'] = cases_malaysia['cluster_education'].astype('int64')
    cases_malaysia['cluster_detentionCentre'] = cases_malaysia['cluster_detentionCentre'].astype('int64')
    cases_malaysia['cluster_workplace'] = cases_malaysia['cluster_workplace'].astype('int64')
    new_cases_malaysia = cases_malaysia.iloc[227:].reset_index(level=0,drop=True) 
    return new_cases_malaysia

new_cases_malaysia = load_data(374)

#cases_state dataset
def load_data2(n_rows):
    cases_state = pd.read_csv("cases_state.csv")
    cases_state.isna().sum()
    new_cases_state = cases_state[3632:].reset_index(level=0,drop=True) 
    return new_cases_state

new_cases_state = load_data2(5984)

# clusters dataset 
def load_data3(n_rows):
    clusters = pd.read_csv("clusters.csv")
    clusters.isna().sum()
    new_clusters = clusters.iloc[106:].reset_index(level=0,drop=True)
    return new_clusters

new_clusters = load_data3(5047)

# test malaysia dataset
def load_data4(n_rows):
    tests_malaysia = pd.read_csv("tests_malaysia.csv")
    tests_malaysia.isna().sum()
    new_tests_malaysia = tests_malaysia.iloc[228:].reset_index(level=0,drop=True)
    return new_tests_malaysia

new_tests_malaysia = load_data4(372)

# tests state dataset 
def load_data5(n_rows):
    tests_state = pd.read_csv("tests_state.csv")
    tests_state.isna().sum()
    new_tests_state = tests_state
    return new_tests_state

new_tests_state = load_data5(1216)

# hospital dataset
def load_data6(n_rows):
    hospital = pd.read_csv("hospital.csv")
    hospital.isna().sum()
    new_hospital = hospital.iloc[2520:].reset_index(level=0,drop=True)
    return new_hospital

new_hospital = load_data6(5739)

# ICU dataset
def load_data7(n_rows):
    icu = pd.read_csv("icu.csv")
    icu.isna().sum()
    new_icu = icu.iloc[2520:].reset_index(level=0,drop=True)
    return new_icu

new_icu = load_data7(5759)

# PKRC dataset
def load_data8(n_rows):
    pkrc = pd.read_csv("pkrc.csv")
    pkrc.isna().sum()
    new_pkrc = pkrc.iloc[1383:].reset_index(level=0,drop=True)
    return new_pkrc

new_pkrc = load_data8(4871)

# Death dataset
def load_data9(n_rows):
    deaths_state = pd.read_csv("deaths_state.csv")
    deaths_state.isna().sum()
    new_deaths_state = deaths_state.iloc[2800:].reset_index(level=0,drop=True)
    return new_deaths_state

new_deaths_state = load_data9(5984)

###################################################################################################################################

# Streamlit 
html_temp = """
<div style ="background-color:white;padding:3.5px">
<h1 style="color:black;text-align:center;">TDS 3301 Data Mining</h1>
</div><br>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.text('Prepared by: Wong Phang Wei 1171103580, Chan Wai Jun 1171103397, Sim Shin Xuan 1181101676')

st.title('Question 3 - Python Programming')

# Define Dataset
st.markdown('#### Datasets')
st.write(
    "The datasets are collected from the official account of the Ministry of Health (MOH) Malaysia in GitHub."
    " It is related to the daily COVID-19 cases, different test cases, healthcare statements, " 
    "vaccination progress, daily death cases, daily checkins, and the population of the community in Malaysia."
    " Data is updated from time to time for a better understanding about the latest trend and patterns of COVID-19 cases."
    " It has divided into types of categories where cases and testing, healthcare, deaths, vaccination, mobility"
    " and contact tracing and static data. Hence, this question will mainly focus on the cases and testing datatsets."
)

# QUESTION 3i 
# EDA
st.markdown('# Exploratory Data Analysis (EDA)')
# Import Dataset
st.markdown('### Import Datasets')
img = Image.open('cases_malaysia.jpg')
st.image(img, width=650)
st.write(
    "Datasets that collected from the official MOH account can be considered as dirty as it may contain incomplete "
    "data and noise which lack of certain attribute values or outliers."
)

# Check for Missing Values
st.markdown('### Check for Missing Values')
img = Image.open('cases_malaysia1.jpg')
st.image(img, width=650)
st.write(
     "For this question, the missing value is detected and summed up with the 'isna.sum()' function in all datasets. "
     "Result shows that only the 'cases_malaysia' dataset contains missing values. There is a total of 342 missing values "
     "from different columns that located in this dataset."
)

# Handle for Missing Values 
st.markdown('### Handle for Missing Values')
img = Image.open('cases_malaysia2.jpg')
st.image(img, width=650)
st.write(
    "The 'fillna()' function is performed to handle the missing values with zero. The data types "
    "is converted from float to integer to ensure the consistency of the dataframe for further "
    "analysis. It is to integrate the measurable of data frame as the COVID-19 cases does not contain float data "
    "types for analysis."
)

# Outliers Detection
st.markdown('# Outliers Detection')
st.text("")
st.write(
    "Outliers are detected in certain datasets. The most significant outliers are detected in the 'new_test_malaysia'"
    " and 'new_test_state' datasets. These datasets recorded the number of test cases of COVID-19 in each state and the"
    " whole Malaysia. Boxplot is plotted to show the interesting outliers that exists in both datasets. The 1st "
    "quartile and the 3rd quartile are calculated in both boxplot for further calculation of the interquartile "
    "range (IQR). The IQR is then used to calculate the upper boundary and lower boundary to determine the possible range "
    "of outliers that exist in the boxplot. "
)

# outliers for test malaysia dataset 
st.markdown('### Test Malaysia Dataset')

# plot
sns.set(rc={'figure.figsize':(10,10)})
ax = new_tests_malaysia.boxplot()
st.pyplot(ax=ax)
st.write(
    "For this dataset, the median of the rtk-ag boxplot is lower than the pcr boxplot and both boxes are overlapped."
    "The rtk-ag boxplot is spread wider than the pcr boxplot. Besides, the rtk-ag boxplot and pcr boxplot are positively"
    " skewed to the right. There is a total of 4 upper outliers exist in the rtk-ag boxplot and there is no outliers exist"
    " in the pcr boxplot. "
)

# test_malaysia RTK outlier detection
tests_malaysia_rtk = new_tests_malaysia.loc[:,['rtk-ag']]
q1 = np.quantile(tests_malaysia_rtk,0.25)
q3 = np.quantile(tests_malaysia_rtk,0.75)
iqr = q3-q1
upper_boundary = q3+(1.5*iqr)
lower_boundary = q1-(1.5*iqr)
outliers_malaysia_rktag = tests_malaysia_rtk[(tests_malaysia_rtk <= lower_boundary) | (tests_malaysia_rtk >= upper_boundary)]
show_outliers_malaysia_rtkag = outliers_malaysia_rktag.dropna()

st.write("The upper outliers for the rtk-ag boxplot in 'new_tests_malaysia' dataset is showed as below:")
if st.checkbox("Upper Outliers for the RTK-ag boxplot in the test malaysia dataset"):
    st.write(show_outliers_malaysia_rtkag)
pass

# tests_malaysia PCR outliers detection 
tests_malaysia_pcr = new_tests_malaysia.loc[:,['pcr']]
q1 = np.quantile(tests_malaysia_pcr,0.25)
q3 = np.quantile(tests_malaysia_pcr,0.75)
iqr = q3-q1
upper_boundary = q3+(1.5*iqr)
lower_boundary = q1-(1.5*iqr)
outliers_malaysia_pcr = tests_malaysia_pcr[(tests_malaysia_pcr <= lower_boundary) | (tests_malaysia_pcr >= upper_boundary)]
show_outliers_malaysia_pcr = outliers_malaysia_pcr.dropna()

#outliers for test_state dataset
st.markdown('### Test State Dataset')

#plot 
sns.set(rc={'figure.figsize':(10,10)})
ax = new_tests_state.boxplot()
st.pyplot(ax=ax)
st.write(
    "The median of the rtk-ag boxplot is lower than the pcr boxplot. The rtk-ag boxplot spread slightly wider than"
    " the pcr boxplot. Both rtk-ag and pcr boxplot are positively skewed to the right as well. There is a total"
    " of 91 upper outliers exist in the rtk-ag boxplot while the pcr boxplot contains 115 upper outliers."
    " Both boxplot does not contain any lower outliers."
)

# tests_state RTK outliers detection
tests_state_rtk = new_tests_state.loc[:,['rtk-ag']]
q1 = np.quantile(tests_state_rtk,0.25)
q3 = np.quantile(tests_state_rtk,0.75)
iqr = q3-q1
upper_boundary = q3+(1.5*iqr)
lower_boundary = q1-(1.5*iqr)
outliers_state_rkt = tests_state_rtk[(tests_state_rtk <= lower_boundary) | (tests_state_rtk >= upper_boundary)]
show_outliers_state_rtk = outliers_state_rkt.dropna()

# tests_state PCR outliers detection 
tests_state_pcr = new_tests_state.loc[:,['pcr']]
q1 = np.quantile(tests_state_pcr,0.25)
q3 = np.quantile(tests_state_pcr,0.75)
iqr = q3-q1
upper_boundary = q3+(1.5*iqr)
lower_boundary = q1-(1.5*iqr)
outliers_state_pcr = tests_state_pcr[(tests_state_pcr <= lower_boundary) | (tests_state_pcr >= upper_boundary)]
show_outliers_state_pcr = outliers_state_pcr.dropna()

st.write("The upper outliers for the rtk-ag boxplot in 'new_tests_state' dataset is showed as below:")
if st.checkbox('Upper Outliers for the RTK-ag boxplot  in tests state dataset'):
    st.write(show_outliers_state_rtk)
pass

st.write("The upper outliers for the pcr boxplot in 'new_tests_state' dataset is showed as below:")
if st.checkbox('Upper Outliers for the PCR boxplot in tests state dataset'):
    st.write(show_outliers_state_pcr)
pass

# QUESTION 3ii
#Data Preprocessing
#Cases State
df1 = new_cases_state.copy()
df1 = pd.pivot_table(df1,columns=['state'], aggfunc=np.sum)
df1 = df1.drop(['cases_import','cases_recovered'])

#Hospital
df2 = new_hospital.copy()
df2 = pd.pivot_table(df2,columns=['state'], aggfunc=np.sum)
df2 = df2.drop(['admitted_covid','admitted_pui','admitted_total','beds','beds_noncrit',
                'discharged_pui','discharged_pui','discharged_total','hosp_noncovid','hosp_pui'])

#ICU
df3 = new_icu.copy()
df3 = pd.pivot_table(df3,columns=['state'], aggfunc=np.sum)
df3 = df3.drop(['beds_icu','beds_icu_total','beds_icu_rep','icu_noncovid',
                'icu_pui','vent','vent_noncovid','vent_port','vent_port_used',
                'vent_pui','vent_used'])

#Pkrc
df4 = new_pkrc.copy()
df4 = pd.pivot_table(df4,columns=['state'], aggfunc=np.sum)
df4 = df4.drop(['admitted_pui','admitted_total','beds','discharge_covid',
                'discharge_pui','discharge_total','pkrc_noncovid','pkrc_pui'])

#Death State 
df5 = new_deaths_state.copy()
df5 = pd.pivot_table(df5,columns=['state'], aggfunc=np.sum)
df5 = df5.drop(['deaths_bid','deaths_bid_dod','deaths_new_dod'])

#Merge all the date set
frames = [df1,df2,df3,df4,df5]
df6 = pd.concat(frames)

#Heatmap
index = ['cases_new', 'beds_covid', 'discharged_covid', 'hosp_covid	', 'beds_icu_covid','icu_covid',
         'vent_covid','admitted_covid','pkrc_covid','deaths_new']
columns = ['Johor', 'Kedah', 'Kelantan', 'Melaka','Negeri Sembilan','Pahang','Perak','Perlis','Pulau Pinang','Sabah','Sarawak'
          ,'Selangor','Terengganu','W.P. Kuala Lumpur','W.P. Labuan','W.P. Putrajaya']

plt.pcolor(df6)
plt.yticks(np.arange(0.5, len(df6.index), 1), df6.index)
plt.xticks(np.arange(0.5, len(df6.columns), 1), df6.columns)

st.markdown('# Correlation within States')
# change the figure size 
sns.set(rc={'figure.figsize':(5,10)})
ax = sns.heatmap(df6.corr(), vmin=-1, vmax=1, annot=True)
st.pyplot(ax=ax)
st.write("To show the states that have strong correlation with Pahang and Johor. The method .corr() has been used to show the correlation "
"between the state. To visualize the data, heatmap has been used to show the confusion matrix of the correlation. ")

st.write("From the heatmap, we can see that Pahang has strong correlation within all of the states except W.P Kuala Lumpur and W.P Putrajaya. "
"Among the strong correlation states, W.P Labuan, Terengganu, Sarawak, Sabah, Perak, Melaka, Kelantan and Johor have a strong correlation as "
"their correlation coefficient value is more than 0.9. For the top four states are Terengganu, Sabah, Kelantan and W.P Labuan. We will choose "
"the top four states as for the 3rd position there are two states have the same correlation coefficient value 0.97 which is Kelantan and W.P Labuan.")

st.write("For Johor, we can see that it has strong correlation will all the states as all the correlation coefficient value are more than 0.7. "
"But among the strong correlation state, we can see that Kedah, Kelantan, Melaka, Negeri Sembilan, Pahang, Perak, Perlis, Sabah and Terengganu "
"have a much higher corelation coefficient value which is more than 0.9. The top three states are Melaka, Negeri Sembilan and Perak.")

# QUESTION 3iii
new_cases_state_df = new_cases_state.copy()
new_hospital_df = new_hospital.copy()
new_icu_df = new_icu.copy()
new_pkrc_df = new_pkrc.copy()

new_johor_case = new_cases_state_df[new_cases_state_df["state"] == 'Johor']
new_pahang_case = new_cases_state[new_cases_state["state"]== 'Pahang']
new_kedah_case = new_cases_state[new_cases_state["state"] == 'Kedah']
new_selangor_case = new_cases_state[new_cases_state["state"] == 'Selangor']

new_johor_hospital = new_hospital_df[new_hospital_df["state"] =='Johor']
new_pahang_hospital = new_hospital[new_hospital["state"] =='Pahang']
new_kedah_hospital = new_hospital[new_hospital["state"] =='Kedah']
new_selangor_hospital = new_hospital[new_hospital["state"] =='Selangor']

new_johor_icu = new_icu_df[new_icu_df["state"] =='Johor']
new_pahang_icu = new_icu[new_icu["state"] =='Pahang']
new_kedah_icu = new_icu[new_icu["state"] =='Kedah']
new_selangor_icu = new_icu[new_icu["state"] =='Selangor']

new_johor_pkrc = new_pkrc_df[new_pkrc_df["state"] =='Johor']
new_pahang_pkrc = new_pkrc[new_pkrc["state"] =='Pahang']
new_kedah_pkrc = new_pkrc[new_pkrc["state"] =='Kedah']
new_selangor_pkrc = new_pkrc[new_pkrc["state"] =='Selangor']

new_frame_johor = [new_johor_case,new_johor_hospital,new_johor_icu,new_johor_pkrc]
new_frame_pahang = [new_pahang_case,new_pahang_hospital,new_pahang_icu,new_pahang_pkrc]
new_frame_kedah = [new_kedah_case,new_kedah_hospital,new_kedah_icu,new_kedah_pkrc]
new_frame_selangor = [new_selangor_case,new_selangor_hospital,new_selangor_icu,new_selangor_pkrc]

new_df_johor = pd.concat(new_frame_johor,join='outer',ignore_index=True,axis=0)         
new_df_pahang = pd.concat(new_frame_pahang,join='outer',ignore_index=True,axis=0)         
new_df_kedah = pd.concat(new_frame_kedah,join='outer',ignore_index=True,axis=0)         
new_df_selangor = pd.concat(new_frame_selangor,join='outer',ignore_index=True,axis=0)         

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

new_df_johor["date"] = LabelEncoder().fit_transform(new_df_johor.date) 
new_df_pahang["date"] = LabelEncoder().fit_transform(new_df_pahang.date) 
new_df_kedah["date"] = LabelEncoder().fit_transform(new_df_kedah.date) 
new_df_selangor["date"] = LabelEncoder().fit_transform(new_df_selangor.date) 

new_df_johor = pd.get_dummies(new_df_johor)
new_df_johor=new_df_johor.fillna(0)

new_df_pahang = pd.get_dummies(new_df_pahang)
new_df_pahang=new_df_pahang.fillna(0)

new_df_kedah = pd.get_dummies(new_df_kedah)
new_df_kedah=new_df_kedah.fillna(0)

new_df_selangor = pd.get_dummies(new_df_selangor)
new_df_selangor= new_df_selangor.fillna(0)

st.markdown('# Features and Indicators')
st.write("The two feature selection model used were Boruta and Recursive Feature Elimination (RFE) which both are wrapper."
        " The datasets used are cases_state.csv, hospital.csv, icu.csv, and pkrc.csv with selected time frame of one month. "
        "The strong features are indicated based on the ranking and score of more than 0.5 in both Boruta and RFE."
        "The top five features of strong features are selected for the four state of Johor, Pahang, Kedah and Selangor and "
        "were listed down below. ")

st.markdown('### Boruta and RFE')
st.text('')
st.markdown("#### Boruta for Johor")

X_johor = new_df_johor.drop(columns = ['cases_new','date','state_Johor','cases_import','cases_recovered'], axis = 1)
y_johor = new_df_johor['cases_new']
colnames = X_johor.columns

rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5)
feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)
feat_selector.fit(X_johor.values,y_johor.values.ravel())

boruta_score = ranking(list(map(float, feat_selector.ranking_)),colnames,order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values('Score',ascending=False)

ax= sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", 
                                        height=14, aspect=1.9, palette='coolwarm')    
plt.title("Boruta Top 30 Features")                       
st.pyplot(ax=ax)    

#RFE
st.markdown('#### RFE for Johor')
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5, n_estimators=50)
rf.fit(X_johor,y_johor)
rfe = RFECV(rf, min_features_to_select=1, cv=3)
rfe.fit(X_johor,y_johor)
rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)
ax = sns_boruta_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:30], kind = "bar", 
                                        height=14, aspect=1.9, palette='coolwarm')
plt.title("RFE Top 30 Features")
st.pyplot(ax=ax)               

st.write("Top five Strong feature for Johor:")
st.write("i) Boruta: ")
st.write(boruta_score.head())
st.write("ii) RFE: ")
st.write (rfe_score.head())

#Pahang
st.markdown('### Boruta and RFE for Pahang')
st.markdown('#### Boruta for Pahang')

X_pahang = new_df_pahang.drop(columns = ['cases_new','date','state_Pahang','cases_import','cases_recovered'], axis = 1)
y_pahang = new_df_pahang['cases_new']
colnames = X_pahang.columns

rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5)
feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)
feat_selector.fit(X_pahang.values,y_pahang.values.ravel())

boruta_score = ranking(list(map(float, feat_selector.ranking_)),colnames,order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values('Score',ascending=False)

ax= sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", 
                                        height=14, aspect=1.9, palette='coolwarm')    
plt.title("Boruta Top 30 Features")                       
st.pyplot(ax=ax)    
            
st.markdown("#### RFE for Pahang")
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5, n_estimators=50)
rf.fit(X_pahang,y_pahang)
rfe = RFECV(rf, min_features_to_select=1, cv=3)
rfe.fit(X_pahang,y_pahang)
rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

ax = sns_boruta_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:30], kind = "bar", 
                                        height=14, aspect=1.9, palette='coolwarm')
plt.title("RFE Top 30 Features")
st.pyplot(ax=ax)      

st.write("Top five Strong feature for Pahang:")
st.write("i) Boruta: ")
st.write(boruta_score.head())
st.write("ii) RFE: ")
st.write (rfe_score.head())         

#Kedah

st.markdown('### Boruta and RFE for Kedah')
st.markdown('#### Boruta for Kedah')
st.text('')

X_kedah = new_df_kedah.drop(columns = ['cases_new','date','state_Kedah','cases_import','cases_recovered'], axis = 1)
y_kedah = new_df_kedah['cases_new']
colnames = X_kedah.columns

rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5)
feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)
feat_selector.fit(X_kedah.values,y_kedah.values.ravel())

boruta_score = ranking(list(map(float, feat_selector.ranking_)),colnames,order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values('Score',ascending=False)

ax= sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", 
                                        height=14, aspect=1.9, palette='coolwarm')    
plt.title("Boruta Top 30 Features")                       
st.pyplot(ax=ax)    
            
st.markdown("#### RFE for Kedah")
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5, n_estimators=50)
rf.fit(X_kedah,y_kedah)
rfe = RFECV(rf, min_features_to_select=1, cv=3)
rfe.fit(X_kedah,y_kedah)
rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

ax = sns_boruta_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:30], kind = "bar", 
                                        height=14, aspect=1.9, palette='coolwarm')
plt.title("RFE Top 30 Features")
st.pyplot(ax=ax)             
    
st.write("Top five Strong feature for Kedah:")
st.write("i) Boruta: ")
st.write(boruta_score.head())
st.write("ii) RFE: ")
st.write (rfe_score.head())  

#Selangor

st.markdown('### Boruta and RFE for Selangor')
st.markdown('#### Boruta for Selangor')

X_selangor = new_df_selangor.drop(columns = ['cases_new','date','state_Selangor','cases_import','cases_recovered'], axis = 1)
y_selangor = new_df_selangor['cases_new']
colnames = X_selangor.columns

rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5)
feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)
feat_selector.fit(X_selangor.values,y_selangor.values.ravel())

boruta_score = ranking(list(map(float, feat_selector.ranking_)),colnames,order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values('Score',ascending=False)

ax= sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", 
                                        height=14, aspect=1.9, palette='coolwarm')    
plt.title("Boruta Top 30 Features")                       
st.pyplot(ax=ax)    
            
st.markdown("#### RFE for Selangor")
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5, n_estimators=50)
rf.fit(X_selangor,y_selangor)
rfe = RFECV(rf, min_features_to_select=1, cv=3)
rfe.fit(X_selangor,y_selangor)
rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

ax = sns_boruta_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:30], kind = "bar", 
                                        height=14, aspect=1.9, palette='coolwarm')
plt.title("RFE Top 30 Features")
st.pyplot(ax=ax)     

st.write("Top five Strong feature for Selangor:")
st.write("i) Boruta: ")
st.write(boruta_score.head())
st.write("ii) RFE: ")
st.write (rfe_score.head())          

st.write("As shown in the results the top features were very similar with slight difference between each state. "
"This indicates that these features can be used to improve the accuracy when use in prediciting the daily cases for this four states.")

# QUESTION 3iv 
# Extract out Pahang, Kedah, Johor Selangor 
johor_case = new_cases_state[new_cases_state["state"] == 'Johor']
pahang_case = new_cases_state[new_cases_state["state"]== 'Pahang']
kedah_case = new_cases_state[new_cases_state["state"] == 'Kedah']
selangor_case = new_cases_state[new_cases_state["state"] == 'Selangor']
frame = [johor_case,pahang_case,kedah_case,selangor_case]
new_cases_states = pd.concat(frame,join='inner',ignore_index=True,axis=0)

# Test Train Split for Classification Model
# label encoder for clssification model
new_cases_states["date"] = LabelEncoder().fit_transform(new_cases_states.date) 
X = new_cases_states.drop(['state'], axis=1)
y = new_cases_states['state']
print(X.shape)
print(y.shape)

st.markdown('# Classification and Regression Models')
st.markdown('### Label Encoder for Classification Model')
st.write("This dataset consists of the data in 4 different states where Johor, Pahang, Kedah and Selangor. "
        "The dataset is undergo label encoder process before the model implementation to normalize the date column into labels. "
        "The date columns in this datatsets has labeled from date time labels to numerical labels.")
st.write(new_cases_states)

st.markdown('### Train Test Split')
st.write('The datasets are split into train sets and test sets with 70%'
        " and 30%" " respectively. The dataset is splitted randomly for model implementation.")

st.markdown('## Classification Models')
st.write("There are several types of classification models has been train and test in this section."
        " The best model that fit the data well in this section are the K-Nearest Neighbors algorithms "
        "and the Decision Tree Classifiers.")

# KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# model accuracy 
train = knn.score(X_train, y_train)
test = knn.score(X_test, y_test)
accurate = '{:.3f}'.format(accuracy_score(y_test, y_pred))
confusion_majority = confusion_matrix(y_test, y_pred)                                   
report = classification_report(y_test, y_pred)

st.markdown('### K-Nearest Neighbors (KNN)')
st.write("The parameter that pass in the KNN model is 5 for the number of neighbors. "
        "The overall performance of the KNN model, confusion matrix and the classification report are showed as below:")

st.markdown('#### Evaluation Metrics for KNN')
st.text('')
if st.checkbox('Accuracy Score of KNN'):
    st.write("The overall performance of the KNN model is",accurate)
pass

if st.checkbox('Confusion Matrix of KNN'):
    st.text(confusion_majority)
pass

if st.checkbox('Classification Report of KNN'):
    st.text(report)
pass 

st.text('')
st.markdown('#### Explanation')
st.text('')
st.write('The KNN model fits the data averaging as it contains the precision, recall and f1 score nearer to the perfect score of 1.0.'
        " The Selangor state consists of the most accurate state in predicting the daily cases with its high precision scores. "
        " Thus, Selangor state has the best f1-score among all states as it have a lower false postive score and lower false "
        " negative values. The macro average computed from the confusion matrix independently for each data in the KNN model and return "
        " the average score equally. The important data that recognised by the KNN model was showed 70% " "accurately in the weighted average." 
        " With this, it has the sufficient support that shows the prediction of daily cases performes well in the Selangor state. "
        ""
)

# Decision Tree Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
dtree = DecisionTreeClassifier(criterion="gini", max_depth=10, splitter='best')
dtree.get_params()
dtree = dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
train = dtree.score(X_train, y_train)
test = dtree.score(X_test, y_test)
accurate = '{:.3f}'.format(accuracy_score(y_test, y_pred))
confusion_majority = confusion_matrix(y_test, y_pred)                       
report = classification_report(y_test, y_pred)

st.markdown("### Decision Tree Classifier")
st.write("There are several types of parameters have passed in this model for training. "
          "The best parameter that tested out is the gini criteria, the best split for each node and the 10 counts for maximum depth."
          " The accuracy score, confusion matrix and the classification report of this model has shown as below:")

st.markdown('#### Evaluation Metrics for Decision Tree Classifier')
st.text('')
if st.checkbox('Accuracy Score of Decision Tree Classifier'):
    st.write("The overall performance for the Decision Tree Classifier is",accurate)
pass

if st.checkbox('Confusion Matrix of Decision Tree Classifier'):
    st.text(confusion_majority)
pass

if st.checkbox('Classification Report of Decision Tree Classifier'):
    st.text(report)
pass

st.text('')
st.markdown('#### Explanation')
st.text('')
st.write("The performance of the Decision Tree Classifier is well as the score in the classfication report is mostly to the perfect score."
         " The Selangor state contains the most accurate and positive result in this model with its high precision and recall value. Besides, "
         " Selangor state has the highest f1-score compared to the other states. The macro average value and the weighted average value scores "
         " 74%" " accurately in this model. There is a good support that the Decision Tree Classifier has good prediction for the daily cases "
         " in Selangor state." 
)

st.write("As compared, the performance of the Decision Tree Classifier is better than the KNN model in predicting the daily cases"
         " from the actual cases.")

# new label encode for model evaluation - classification model
new_cases_states["date"] = LabelEncoder().fit_transform(new_cases_states.date) 
new_cases_states["state"] = LabelEncoder().fit_transform(new_cases_states.state) 
X = new_cases_states.drop(['state'],axis=1)
y = new_cases_states['state']

# Model Evaluation for KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Johor
fpr_0, tpr_0, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=0)
prob_KNN_0 = knn.predict_proba(X_test)
prob_KNN_0 = prob_KNN_0[:,1]
auc_KNN_0 = '{:.3f}'.format(metrics.auc(fpr_0,tpr_0))
# ROC 
fpr_KNN_0, tpr_KNN_0, thresholds_KNN_0 = metrics.roc_curve(y_test, prob_KNN_0, pos_label=0) 
# precision-Recall 
prec_KNN_0, rec_KNN_0, threshold_KNN_0 = metrics.precision_recall_curve(y_test, prob_KNN_0, pos_label=0)

# Pahang
fpr_1, tpr_1, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=1)
prob_KNN_1 = knn.predict_proba(X_test)
prob_KNN_1 = prob_KNN_1[:,1]
auc_KNN_1 = '{:.3f}'.format(metrics.auc(fpr_1,tpr_1))
# ROC
fpr_KNN_1, tpr_KNN_1, thresholds_KNN_1 = metrics.roc_curve(y_test, prob_KNN_1, pos_label=1) 
# precision-recall
prec_KNN_1, rec_KNN_1, threshold_KNN_1 = metrics.precision_recall_curve(y_test, prob_KNN_1, pos_label=1)

# Kedah
fpr_2, tpr_2, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=2)
prob_KNN_2 = knn.predict_proba(X_test)
prob_KNN_2 = prob_KNN_2[:,1]
auc_KNN_2 = '{:.3f}'.format(metrics.auc(fpr_2,tpr_2))
# ROC
fpr_KNN_2, tpr_KNN_2, thresholds_KNN_2 = metrics.roc_curve(y_test, prob_KNN_2, pos_label=2) 
# precision-recall
prec_KNN_2, rec_KNN_2, threshold_KNN_2 = metrics.precision_recall_curve(y_test, prob_KNN_2, pos_label=2)

# Selangor  
fpr_3, tpr_3, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=3)
prob_KNN_3 = knn.predict_proba(X_test)
prob_KNN_3 = prob_KNN_3[:,1]
auc_KNN_3 = '{:.3f}'.format(metrics.auc(fpr_3,tpr_3))
# ROC curve
fpr_KNN_3, tpr_KNN_3, thresholds_KNN_3 = metrics.roc_curve(y_test, prob_KNN_3, pos_label=3) 
# precision-recall
prec_KNN_3, rec_KNN_3, threshold_KNN_3 = metrics.precision_recall_curve(y_test, prob_KNN_3, pos_label=3)


# Model Evaluation for Decision Tree Classifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
dtree = DecisionTreeClassifier(criterion="gini", max_depth=10, splitter='best')
dtree.get_params()
dtree = dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)

# Johor
fpr_0, tpr_0, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=0)
prob_DT_0 = dtree.predict_proba(X_test)
prob_DT_0 = prob_DT_0[:,1]
auc_DT_0 = '{:.3f}'.format(metrics.auc(fpr_0,tpr_0))
# roc 
fpr_DT_0, tpr_DT_0, thresholds_DT_0 = metrics.roc_curve(y_test, prob_DT_0, pos_label=0) 
# precision
prec_DT_0, rec_DT_0, threshold_DT_0 = metrics.precision_recall_curve(y_test, prob_DT_0, pos_label=0)

# Pahang
fpr_1, tpr_1, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=1)
prob_DT_1 = dtree.predict_proba(X_test)
prob_DT_1 = prob_DT_1[:,1]
auc_DT_1 = '{:.3f}'.format(metrics.auc(fpr_1,tpr_1))
# roc 
fpr_DT_1, tpr_DT_1, thresholds_DT_1 = metrics.roc_curve(y_test, prob_DT_1, pos_label=1) 
# precision
prec_DT_1, rec_DT_1, threshold_DT_1 = metrics.precision_recall_curve(y_test, prob_DT_1, pos_label=1)

# Kedah
fpr_2, tpr_2, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=2)
prob_DT_2 = dtree.predict_proba(X_test)
prob_DT_2 = prob_DT_2[:,1]
auc_DT_2 = '{:.3f}'.format(metrics.auc(fpr_2,tpr_2))
# roc
fpr_DT_2, tpr_DT_2, thresholds_DT_2 = metrics.roc_curve(y_test, prob_DT_2, pos_label=2) 
# precision
prec_DT_2, rec_DT_2, threshold_DT_2 = metrics.precision_recall_curve(y_test, prob_DT_2, pos_label=2)

# Selangor 
fpr_3, tpr_3, thresholds = metrics.roc_curve(y_test,y_pred, pos_label=3)
prob_DT_3 = dtree.predict_proba(X_test)
prob_DT_3 = prob_DT_3[:,1]
auc_DT_3 = '{:.3f}'.format(metrics.auc(fpr_3,tpr_3))
# roc
fpr_DT_3, tpr_DT_3, thresholds_DT_3 = metrics.roc_curve(y_test, prob_DT_3, pos_label=3) 
# precision
prec_DT_3, rec_DT_3, threshold_DT_3 = metrics.precision_recall_curve(y_test, prob_DT_3, pos_label=3)

# Comparison between KNN and Decision Tree Classifier 
st.markdown('### Comparison for Model Evaluation of Classificaton Models')
st.text('')
st.markdown('#### Receiver Operating Characteristic (ROC) Curve')
st.text('')
st.text('ROC curve shows the overall performance of the classification models.')

option = st.selectbox('Select a state for ROC visualization :', ['Johor','Pahang','Kedah','Selangor'])

# Johor
if option == 'Johor':
    st.write('The readings of Area Under Curve (AUC) in Johor is',auc_KNN_0)
    plt.plot(fpr_KNN_0, tpr_KNN_0, color='orange', label='KNN (Johor)') 
    plt.plot(fpr_DT_0, tpr_DT_0, color='blue', label='DT (Johor)')  
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()
pass

# Pahang 
if option == 'Pahang':
    st.write('The readings of Area Under Curve (AUC) in Pahang is',auc_KNN_1)
    plt.plot(fpr_KNN_1, tpr_KNN_1, color='orange', label='KNN (Pahang)') 
    plt.plot(fpr_DT_1, tpr_DT_1, color='blue', label='DT (Pahang)')  
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()
pass

# Kedah 
if option == 'Kedah':
    st.write('The readings of Area Under Curve (AUC) in Kedah is',auc_KNN_2)
    plt.plot(fpr_KNN_2, tpr_KNN_2, color='orange', label='KNN (Kedah)') 
    plt.plot(fpr_DT_2, tpr_DT_2, color='blue', label='DT (Kedah)')  
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()
pass

# Selangor
if option == 'Selangor':
    st.write('The readings of Area Under Curve (AUC) in Selangor is',auc_KNN_3)
    plt.plot(fpr_KNN_3, tpr_KNN_3, color='orange', label='KNN (Selangor)') 
    plt.plot(fpr_DT_3, tpr_DT_3, color='blue', label='DT (Selangor)')  
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()
pass

st.write('The true postive rate and the false positive rate for each state has been calculated with different types of thresholds. '
         "The area covered below the line is defined as area under the curve. AUC has been calculated to evaluate the "
         "performance of each state in the classificaton model. Based on the ROC curve plotting, the Selangor state has the higher AUC"
         " readings as it contains a better model in recognising the daily cases. Although the Pahang state having a lower AUC reading "
         "in this ROC curve graph, whereas it was the best ROC curve graph among the other states. The ROC curve graph of Pahang state "
         "shows the trade off of the between the true positive rate and the false positive rate."
)

st.markdown('#### Precision-Recall Curve')
st.text('')
st.text('Precision-recall curve shows the relationship between the precision and the recall readings.')

option = st.selectbox('Select a state for Precision-Recall visualization :', ['Johor','Pahang','Kedah','Selangor'])

# Johor
if option == 'Johor':
    st.write('The readings of Area Under Curve (AUC) in Johor is',auc_DT_0)
    plt.plot(prec_KNN_0, rec_KNN_0, color='blue', label='KNN (Johor)') 
    plt.plot(prec_DT_0, rec_DT_0, color='orange', label='DT (Johor)') 
    plt.plot([1, 0], [0.1, 0.1], color='black', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()
pass

# Pahang
if option == 'Pahang':
    st.write('The readings of Area Under Curve (AUC) in Pahang is',auc_DT_1)
    plt.plot(prec_KNN_1, rec_KNN_1, color='blue', label='KNN (Pahang)') 
    plt.plot(prec_DT_1, rec_DT_1, color='orange', label='DT (Pahang)') 
    plt.plot([1, 0], [0.1, 0.1], color='black', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()
pass

# Kedah 
if option == 'Kedah':
    st.write('The readings of Area Under Curve (AUC) in Kedah is',auc_DT_2)
    plt.plot(prec_KNN_2, rec_KNN_2, color='blue', label='KNN (Kedah)') 
    plt.plot(prec_DT_2, rec_DT_2, color='orange', label='DT (Kedah)') 
    plt.plot([1, 0], [0.1, 0.1], color='black', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()
pass

# Selangor 
if option == 'Selangor':
    st.write('The readings of Area Under Curve (AUC) in Selangor is',auc_DT_3)
    plt.plot(prec_KNN_3, rec_KNN_3, color='blue', label='KNN (Selangor)') 
    plt.plot(prec_DT_3, rec_DT_3, color='orange', label='DT (Selangor)') 
    plt.plot([1, 0], [0.1, 0.1], color='black', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()
pass

st.write('The precision-recall curve shows the trade off between the precision and the recall'
         " readings with different threshold values. Based on the results, the Selangor state"
         " contains the highest precision and recall readings. The higher precision readings "
         " indicate to a low false positive rate and low false nagative rate. The Pahang state "
         "having the best precision and recall graph among all the other states although it "
         "contains a lower AUC readings."
)

st.write('As result, the Selangor state has the higher AUC readings in both classification model but it does not showed a good'
          " ROC curve graph and precsion-recall curve graph. With this, the Pahang state having the best ROC curve graph and precision-recall"
          " curve graph compared to the other states. The performance of the KNN model is better than the Decision Tree Classifier for both "
          " RoC curve graph and precision-recall curve graph.")

# Regression Model
new_cases_states["date"] = LabelEncoder().fit_transform(new_cases_states.date) 
new_cases_states["state"] = LabelEncoder().fit_transform(new_cases_states.state) 

X = new_cases_states.drop(['state'],axis=1)
y = new_cases_states['state']

st.markdown('### Label Encoder for Regression Model')
st.write('The dataset that used for regression model is maintain in the 4 different states as well. Label encoder is performed'
         "for both date columns and state columns. The state columns has labeled from string data types to numerical labels.")
st.write(new_cases_states)

st.markdown('## Regression Models')
st.write('There most suitable regression model that fit the data well are the Decision Tree Regressor and the Linear Regression.')

# Decision Tree Regressor 
X_train_dtreg, X_test_dtreg, y_train_dtreg, y_test_dtreg = train_test_split(X, y, test_size=0.3, random_state=0)
dtreg = DecisionTreeRegressor(max_depth=8, criterion="mse", splitter="best")  
dtreg.fit(X_train, y_train) 
y_pred_dtreg = dtreg.predict(X_test)

r2 = '{:.3f}'.format(r2_score(y_test_dtreg, y_pred_dtreg))
mae = '{:.3f}'.format(mean_absolute_error(y_test_dtreg, y_pred_dtreg))
mse = '{:.3f}'.format(mean_squared_error(y_test_dtreg, y_pred_dtreg))
rmse = '{:.3f}'.format(np.sqrt(mean_squared_error(y_test_dtreg, y_pred_dtreg)))

st.markdown('### Decision Tree Regressor')
st.write("The parameters that used in this model is the 8 values of maximum depth of the tree, mse criteria and the best split for each node.")

st.markdown('#### Evaluation Metrics for Decision Tree Regressor')
st.text('')
if st.checkbox('Accuracy Score of Decision Tree Regressor'):
    st.write("The overall performance of the Decision Tree Regressor is",r2)
pass

if st.checkbox('MAE of Decision Tree Regressor'):
    st.write("The Mean Absolute Error (MAE) of the Decision Tree Regressor is", mae)
pass

if st.checkbox('MSE of Decision Tree Regressor'):
    st.write("The Mean Squared Error (MSE) of the Decision Tree Regressor is", mse)
pass

if st.checkbox('RMSE of Decision Tree Regressor'):
    st.write("The Root Mean Squared Error (RMSE) of the Decision Tree Regressor is", rmse)
pass

# Linear Regression
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.3, random_state=0)
lm = LinearRegression()
lm.fit(X_train_lr, y_train_lr) 
y_pred_lr = lm.predict(X_test_lr)
coef = lm.coef_
intercept = lm.intercept_
r2 = '{:.3f}'.format(r2_score(y_test_lr, y_pred_lr))
mae = '{:.3f}'.format(mean_absolute_error(y_test_lr, y_pred_lr))
mse = '{:.3f}'.format(mean_squared_error(y_test_lr, y_pred_lr))
rmse = '{:.3f}'.format(np.sqrt(mean_squared_error(y_test_lr, y_pred_lr)))

st.markdown('### Linear Regression')
st.markdown('#### Evaluation Metrics for Linear Regression')
st.text('')
if st.checkbox('Accuracy Score of Linear Regression'):
    st.write("The overall performance of the Linear Regression is",r2)
pass

if st.checkbox('MAE of Linear Regression'):
    st.write("The Mean Absolute Error (MAE) of the Linear Regression is", mae)
pass

if st.checkbox('MSE of Linear Regression'):
    st.write("The Mean Squared Error (MSE) of the Linear Regression is", mse)
pass

if st.checkbox('RMSE of Linear Regression'):
    st.write("The Root Mean Squared Error (RMSE) of the Linear Regression is", rmse)
pass

# Model Evaluation for Decision Tree Regression
st.markdown('### Comparison for Model Evaluation of Regression Models')
sns.regplot(x=y_test_dtreg, y=y_pred_dtreg, x_jitter=.01, scatter_kws={"color": "red"}, line_kws={"color": "red"},scatter=True)
sns.regplot(x=y_test_lr, y=y_pred_lr, x_jitter=.005, scatter_kws={"color": "blue"}, line_kws={"color": "blue"}, scatter=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Plot for Actual Values and Predicted Values ')
st.pyplot()
st.write('The regression plot is to show how the regression model fits the data well. From the graph above, the'
          " red regression plot represent the Decision Tree Regressor while the blue regression plot represent "
          "the Linear Regression. This plot shows the overall performance of the actual values and the predicted "
          "values of the two regression models. "
)
st.text('')
st.write(
          " For comparison, the performance of Decision Tree Regressor"
          " is better than the Linear Regression model as it contains a higher r2 score. Besides, the MSE score"
          " of the Decision Tree Regressor is smaller than the Linear Regression to represent a good performance."
          " Despite, the smaller RMSE score represent the better the model as it defined the standard deviation "
          " of the prediction errors in both regression model. Data in Decision Tree Regressor is spread wider "
          "than the data in Linear Regression model. The data points in Decision Tree Regressor is concentrates "
          " around the best fit line compared to the Linear Regression model. "
)

st.write("As conclusion, the overall performance of the classification model will be better in predicting the daily"
         " cases compared to the regression model for this dataset. The classification models consist a better "
         " accuracy score in evaluation metrics compared to the regression models."
)

st.write('-------------------------------------------------------------------------------------------------------')
st.text('Prepared by: Wong Phang Wei 1171103580, Chan Wai Jun 1171103397, Sim Shin Xuan 1181101676')
