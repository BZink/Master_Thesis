#-*- coding:utf-8 -*-

import pandas as pd
import os
import numpy as np
import scipy
import seaborn as sns
import statsmodels.api as sm
import statsmodels.discrete as smd
from patsy import dmatrices
import matplotlib.pyplot as plt
import importlib

#--------------------------------------------------------------------------------------------------------------------------
#Directory, where figures are saved (saving is commented out):
sav_dir=r'C:\Users\benit\Documents\Uni\Master\Masterarbeit\LaTex'

#Datasets
df = pd.read_excel("Daten.xlsx", sheet_name="neu", header=0, index_col=None) #Main dataset
df_Alter = pd.read_excel("Daten.xlsx", sheet_name="Alter", header=None)
df_Abschluss = pd.read_excel("Daten.xlsx", sheet_name="Abschluss", header=None)
df_Tätigkeit = pd.read_excel("Daten.xlsx", sheet_name="Tätigkeit", header=None)
df_Berufsfeld = pd.read_excel("Daten.xlsx", sheet_name="Berufsfeld", header=None)
df_Studium = pd.read_excel("Daten.xlsx", sheet_name="Studium", header=None)
df_Einkommen = pd.read_excel("Daten.xlsx", sheet_name="Einkommen", header=None)

#Creation of new dummy variables - NaNs are transfered to dummy variables
df['Student']=((df['Tätigkeit']==1)&(df['Studium']!=1)&(df['Studium'].notnull())).astype(int)
df['Student']=np.where((df['Tätigkeit'].isnull()|df['Studium'].isnull()),np.nan,df['Student'])
df['Working']=((df['Tätigkeit']!=1)&(df['Tätigkeit']!=7)&(df['Tätigkeit']!=8)).astype(int)
df['Working']=np.where(df['Tätigkeit'].isnull(),np.nan,df['Working'])
df['Akademiker']=((df['Student']==1)|(df['Abschluss']>6)).astype(int)
df['Akademiker']=np.where(((df['Akademiker']==0) & (df['Tätigkeit'].isnull()|df['Studium'].isnull())),np.nan,df['Akademiker'])
df['Selbst']=((df['Tätigkeit']==4)|(df['Tätigkeit']==5)).astype(int)
df['Selbst']=np.where(df['Tätigkeit'].isnull(),np.nan,df['Selbst'])
df['Sozial']=((df['Berufsfeld']==2)|(df['Berufsfeld']==4)).astype(int)
df['Sozial']=np.where(df['Berufsfeld'].isnull(),np.nan,df['Sozial'])
df['Marktberuf']=((df['Berufsfeld']==1)|(df['Berufsfeld']==3)).astype(int)
df['Marktberuf']=np.where(df['Berufsfeld'].isnull(),np.nan,df['Marktberuf'])
df['WiStu']=(df['Studium']==2).astype(int)
df['WiStu']=np.where(df['Studium'].isnull(),np.nan,df['WiStu'])
df['SoStu']=((df['Studium']==4)|(df['Studium']==8)|(df['Studium']==9)).astype(int)
df['SoStu']=np.where(df['Studium'].isnull(),np.nan,df['SoStu'])
df['NatStu']=((df['Studium']==3)|(df['Studium']==5)|(df['Studium']==6)|(df['Studium']==10)).astype(int)
df['NatStu']=np.where(df['Studium'].isnull(),np.nan,df['NatStu'])
df['JurStu']=(df['Studium']==7).astype(int)
df['JurStu']=np.where(df['Studium'].isnull(),np.nan,df['JurStu'])
df['NoStu']=(df['Studium']==1).astype(int)
df['NoStu']=np.where(df['Studium'].isnull(),np.nan,df['NoStu'])

#Cleaning of the main dataset
df=df.drop(columns=['FMIS_1', 'FMIS_2','FMIS_3','FMIS_4','FMIS_5','FMIS_6', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Entscheidung'])

#New datasets 
df_ama = df[df['Amazon']==0].copy().dropna(subset=['Amazon'])
df_waste = df[df['Müll']==0].copy().dropna(subset=['Müll'])
df_secure = df_ama[df_ama['Müll']==0].copy().dropna(subset=['Müll'])
df_Markt = df_ama[df_ama['Markt']==1].copy()

list_df=[df, df_ama, df_waste, df_secure]
list_dfname=['Total', 'w/o Amazon-despisers', 'w/o waste perfectionists', 'w/o both of the above']

#---------------------------------------------------------------------------------------------------------------------------
def Sum_Var():
    summary={'Observations': df.count(),
         'Type2': df.dtypes,
         'Type': ['Dummy', 'Dummy', 'Scale 1-11', 'Scale 1-11', 'Integer 0-1000', 'Dummy','Ordered Groups 1-7', 'Groups', 'Groups', 'Groups', 'Groups', 'Ordered Groups 1-8', 'Scale 1-7', 'Scale 1-7', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy', 'Dummy'],
         'Summe': df.sum(),
         'Mean': df.mean()}
    Var_sum = pd.DataFrame (summary, columns = ['Observations','Type','Summe', 'Mean'])
    return Var_sum
    
def Fair(df):
    Fairshare_market=(df['Dec'][(df['Dec'] == 1) & (df['Markt'] ==1)].count())/(df['Dec'][df['Markt'] ==1].count())*100
    Fairshare_baseline=(df['Dec'][(df['Dec'] == 1) & (df['Markt'] ==0)].count())/(df['Dec'][df['Markt'] ==0].count())*100
    Fairshare=[Fairshare_baseline, Fairshare_market]
    return Fairshare

def Unfair(df):
    Unfairshare_market=(df['Dec'][(df['Dec'] == 0) & (df['Markt'] ==1)].count())/(df['Dec'][df['Markt'] ==1].count())*100
    Unfairshare_baseline=(df['Dec'][(df['Dec'] == 0) & (df['Markt'] ==0)].count())/(df['Dec'][df['Markt'] ==0].count())*100
    Unfairshare=[Unfairshare_baseline, Unfairshare_market]
    return Unfairshare

def barchart(df):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar([0,1], Fair(df), width=0.8)
    ax.bar([0,1], Unfair(df), width=0.8,bottom=Fair(df), color='r')
    ax.set_ylabel('Percent')
    #ax.set_title('Figure 1: Decisions in the baseline and market treatment.')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Baseline', 'Market'])
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend(labels=['Fair', 'Unfair'], loc=(0.63,0.02))
    #plt.savefig(sav_dir+'\Hist1.png', bbox_inches="tight")
    plt.show()
    
def Pearson(df):
    return scipy.stats.pearsonr(df.Dec, df.Markt)

def MWU(df):
    return scipy.stats.mannwhitneyu(df.Dec[df.Markt==0],df.Dec[df.Markt==1],alternative='two-sided')
    
def sum_dfs():
    Fairshare_df = []
    for x in list_df:
        Fairshare_df.append(Fair(x)+list(MWU(x)))
    print("Table 1: Percentage of people deciding to donate")    
    return pd.DataFrame(index=(list_dfname), data = Fairshare_df, columns=('Baseline', 'Market', 'Mann-Whitney-U', 'p-Value'))    
#-------------------Table 3----------------------------
def Reg31():
    y, X = dmatrices('Dec ~ Markt', df, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())
    
def Reg32():
    y, X = dmatrices('Dec ~ Markt + Altru_1 + Altru_2 + Geld + Müll', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())
    
def Reg33():
    y, X = dmatrices('Dec ~ Markt + Altru_1 + Altru_2', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())    

#------------------Table 4--------------------------------    
def Reg41():
    y, X = dmatrices('Dec ~ FMIS_Index*Markt', df, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())
    
def Reg42():
    y, X = dmatrices('Dec ~ FMIS_Index*Markt', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())    
    
def Reg43():
    y, X = dmatrices('Dec ~ FMIS_Index*Markt + Altru_1 + Altru_2', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())   
    
def Reg44():
    y, X = dmatrices('Dec ~ WiStu*Markt + Student', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary()) 
    
def Reg45():
    y, X = dmatrices('Dec ~ Marktgeschehen*Markt', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary()) 
    
def Reg46():
    y, X = dmatrices('Dec ~ Altru_1*Markt', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())     

#------------------------Figure 2--------------------------    
def FMIS_plot():
    y, X = dmatrices('Dec ~ FMIS_Index*Markt', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    #print(res.summary())
    #------------------------------------------------------------------
    def f_Markt(FMIS):
        return res.predict([1,FMIS,1,FMIS])
    def f_Baseline(FMIS):
        return res.predict([1,FMIS,0,0])
    #-------------Vorbereitung--------------------------------
    FMIS_list=list(range(1, 12))
    y_Markt=[None]*len(FMIS_list)
    y_Base=[None]*len(FMIS_list)
    for i in FMIS_list:
        y_Markt[i-1]=f_Markt(i)
    for i in FMIS_list:
        y_Base[i-1]=f_Baseline(i)
    #----------------Estimated Likelihood----------------------    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(FMIS_list, y_Markt, color='tab:blue', linewidth=2)
    ax.plot(FMIS_list, y_Base, color='tab:orange', linewidth=2)
    ax.axhline(y=df_ama['Dec'].mean(), color='gray', linewidth=1, linestyle='--')
    ax.axvline(x=df_ama.FMIS_Index.mean(), color='gray', linewidth=1, linestyle='--')
    #ax.set_title('Likelihood of unfair decision dependent on FMIS')
    ax.set_ylabel('Estimated Probability of Fair Decision')
    ax.set_xlabel('Fair Market Ideology Index')
    ax.legend(labels=['Market', 'Baseline'])
    #plt.savefig(sav_dir+'\FMIS_plot.png', bbox_inches="tight")
    plt.show()
    
def WiStu_plot():
    y, X = dmatrices('Dec ~ WiStu*Markt + Student', df_ama, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    #print(res.summary())
    #--------------------------------------
    def f_Markt(VI):
        return res.predict([1,VI,1,VI,1])
    def f_Baseline(VI):
        return res.predict([1,VI,0,0,1])
    #---------------Vorbereitung ------------
    VI_list=list(range(0, 2))
    y_Markt=[None]*len(VI_list)
    y_Base=[None]*len(VI_list)
    for i in VI_list:
        y_Markt[i]=f_Markt(i)
    for i in VI_list:
        y_Base[i]=f_Baseline(i)
    #--------------Estimated Likelihood----------
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(np.arange(0, 2, step=1))
    ax.plot(VI_list, y_Markt, color='tab:blue', linewidth=2)
    ax.plot(VI_list, y_Base, color='tab:orange', linewidth=2)
    ax.axhline(y=df_ama['Dec'].mean(), color='gray', linewidth=1, linestyle='--')
    ax.axvline(x=df_ama.WiStu.mean(), color='gray', linewidth=1, linestyle='--')
    #ax.set_title('Likelihood of unfair decision dependent on FMIS')
    ax.set_ylabel('Estimated Probability of Fair Decision')
    ax.set_xlabel('Economics Student')
    ax.legend(labels=['Market', 'Baseline'])
    #plt.savefig(sav_dir+'\WiStu_plot.png', bbox_inches="tight")
    plt.show()
    
#------------------------------------------Table 5------------------------------------------------   
def Reg51():
    y, X = dmatrices('Dec ~ Altru_1 + Altru_2 + FMIS_Index + Marktgeschehen + pol_rechts + Gespendet + Geld + weiblich + Alter + Selbst + Sozial + Marktberuf + WiStu + NatStu + SoStu + JurStu + Akademiker', df_Markt, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())
    
def Reg52():
    y, X = dmatrices('Dec ~ Altru_1 + Altru_2 + FMIS_Index + pol_rechts + Gespendet + Geld + weiblich + Alter + Selbst + Sozial + Marktberuf + Akademiker ', df_Markt, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())
    
def Reg53():
    y, X = dmatrices('Dec ~ FMIS_Index + pol_rechts + Gespendet + Geld + weiblich + Alter + Selbst + Sozial + Marktberuf + Akademiker ', df_Markt, return_type = 'dataframe')
    probit = sm.Probit(y, X, missing='drop')
    res=probit.fit()
    print(res.summary())