import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm
import pandas as pd

import pickle


model =None
sin_df=pd.read_csv("Sinhala_dataset.csv",header=None)
sin_df.columns=['data']


sin_df["data"] = sin_df["data"].str.replace('?','')
sin_df["data"] = sin_df["data"].str.replace('/','')
sin_df["data"] = sin_df["data"].str.replace(',','')
sin_df["data"] = sin_df["data"].str.replace('!','')
sin_df["data"] = sin_df["data"].str.replace('.','')
sin_df["data"] = sin_df["data"].str.replace('-','')
sin_df["data"] = sin_df["data"].str.replace('#','')
sin_df["data"] = sin_df["data"].str.replace('=','')
sin_df["data"] = sin_df["data"].str.replace(')','')
sin_df["data"] = sin_df["data"].str.replace('(','')
sin_df["data"] = sin_df["data"].str.replace('\'','')
sin_df["data"] = sin_df["data"].str.replace('\"','')
sin_df["data"] = sin_df["data"].str.replace('\u200d','')
sin_df["data"] = sin_df["data"].str.replace('\n','')



sin_df['data'] = sin_df['data'].str.replace('\d+', '')



sin_df.to_csv(r'output.csv')

cap=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
sim=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

sin_df['data'] = sin_df['data'].str.replace('[a-zA-Z]', '')

sin_df.to_csv(r'output1.csv')

A=set(sin_df['data'].str.lower().str.split().sum())

en_df=pd.read_csv("English_dataset.csv",header=None)

en_df.head(10)

en_df.columns=['data']

en_df["data"] = en_df["data"].str.replace('?','')
en_df["data"] = en_df["data"].str.replace('.','')
en_df["data"] = en_df["data"].str.replace(',','')
en_df["data"] = en_df["data"].str.replace('-','')
en_df["data"] = en_df["data"].str.replace('/','')
en_df["data"] = en_df["data"].str.replace('\\','')
en_df["data"] = en_df["data"].str.replace('\'','')
en_df["data"] = en_df["data"].str.replace('\"','')
en_df["data"] = en_df["data"].str.replace('(','')
en_df["data"] = en_df["data"].str.replace(')','')
en_df['data'] = en_df['data'].str.replace('\d+', '')

B=set(en_df['data'].str.lower().str.split().sum())

df_singlish=pd.read_csv("./Datasets/Singlish_dataset.csv",header=None)

df_singlish.columns=['data']

df_singlish.head(10)

df_singlish['data'] = df_singlish['data'].str.replace('?','')
df_singlish['data'] = df_singlish['data'].str.replace('.','')
df_singlish['data'] = df_singlish['data'].str.replace(',','')
df_singlish['data'] = df_singlish['data'].str.replace('\"','')
df_singlish['data'] = df_singlish['data'].str.replace('\'','')
df_singlish['data'] = df_singlish['data'].str.replace('-','')
df_singlish['data'] = df_singlish['data'].str.replace('=','')
df_singlish['data'] = df_singlish['data'].str.replace('!','')
df_singlish['data'] = df_singlish['data'].str.replace('(','')
df_singlish['data'] = df_singlish['data'].str.replace(')','')
df_singlish['data'] = df_singlish['data'].str.replace('\d+', '')
df_singlish['data'] = df_singlish['data'].str.lower()

df_singlish['data']

C=set(df_singlish['data'].str.split().sum())


df_english_coloumn = pd.DataFrame(B)
df_sinhala_coloumn =  pd.DataFrame(A)
df_singlish_coloumn =  pd.DataFrame(C)

df_english_coloumn["class"]="English"
df_singlish_coloumn["class"]="Singlish"
df_sinhala_coloumn["class"]="Sinhala"


df_combined = df_english_coloumn.append(df_sinhala_coloumn)
df_combined2 = df_combined.append(df_singlish_coloumn)


  
df_combined2['En_Suf_eer'] = np.where(df_combined2[0].str.endswith('eer'),1,0);
df_combined2['En_Suf_ion'] = np.where(df_combined2[0].str.endswith('ion'),1,0);
df_combined2['En_Suf_ism'] = np.where(df_combined2[0].str.endswith('ism'),1,0);
df_combined2['En_Suf_ity'] = np.where(df_combined2[0].str.endswith('ity'),1,0);
df_combined2['En_Suf_or'] = np.where(df_combined2[0].str.endswith('or'),1,0);
df_combined2['En_Suf_sion'] = np.where(df_combined2[0].str.endswith('sion'),1,0);
df_combined2['En_Suf_ship'] = np.where(df_combined2[0].str.endswith('ship'),1,0);
df_combined2['En_Suf_th'] = np.where(df_combined2[0].str.endswith('th'),1,0);
df_combined2['En_Suf_able'] = np.where(df_combined2[0].str.endswith('able'),1,0);
df_combined2['En_Suf_ible'] = np.where(df_combined2[0].str.endswith('ible'),1,0);
df_combined2['En_Suf_al'] = np.where(df_combined2[0].str.endswith('al'),1,0);
df_combined2['En_Suf_ant'] = np.where(df_combined2[0].str.endswith('ant'),1,0);
df_combined2['En_Suf_ary'] = np.where(df_combined2[0].str.endswith('ary'),1,0);
df_combined2['En_Suf_ful'] = np.where(df_combined2[0].str.endswith('ful'),1,0);
df_combined2['En_Suf_ic'] = np.where(df_combined2[0].str.endswith('ic'),1,0);
df_combined2['En_Suf_ious'] = np.where(df_combined2[0].str.endswith('ious'),1,0);
df_combined2['En_Suf_ous'] = np.where(df_combined2[0].str.endswith('ous'),1,0);
df_combined2['En_Suf_y'] = np.where(df_combined2[0].str.endswith('y'),1,0);
df_combined2['En_Suf_ed'] = np.where(df_combined2[0].str.endswith('ed'),1,0);
df_combined2['En_Suf_en'] = np.where(df_combined2[0].str.endswith('en'),1,0);
df_combined2['En_Suf_er'] = np.where(df_combined2[0].str.endswith('er'),1,0);
df_combined2['En_Suf_ing'] = np.where(df_combined2[0].str.endswith('ing'),1,0);
df_combined2['En_Suf_ise'] = np.where(df_combined2[0].str.endswith('ise'),1,0);
df_combined2['En_Suf_ly'] = np.where(df_combined2[0].str.endswith('ly'),1,0);
df_combined2['En_Suf_ward'] = np.where(df_combined2[0].str.endswith('ward'),1,0);
df_combined2['En__SUf_wise'] = np.where(df_combined2[0].str.endswith('wise'),1,0);
df_combined2['En_Pre_anti'] = np.where(df_combined2[0].str.startswith('anti'),1,0);
df_combined2['En_Pre_auto'] = np.where(df_combined2[0].str.startswith('auto'),1,0);
df_combined2['En_Pre_circum'] = np.where(df_combined2[0].str.startswith('circum'),1,0);
df_combined2['En_Pre_dis'] = np.where(df_combined2[0].str.startswith('dis'),1,0);
df_combined2['En_Pre_ex'] = np.where(df_combined2[0].str.startswith('ex'),1,0);
df_combined2['En_Pre_extra'] = np.where(df_combined2[0].str.startswith('extra'),1,0);
df_combined2['En_Pre_hetero'] = np.where(df_combined2[0].str.startswith('hetero'),1,0);
df_combined2['En_Pre_homo'] = np.where(df_combined2[0].str.startswith('homo'),1,0);
df_combined2['En_Pre_hyper'] = np.where(df_combined2[0].str.startswith('hyper'),1,0);
df_combined2['En_Pre_em'] = np.where(df_combined2[0].str.startswith('em'),1,0);
df_combined2['En_Pre_ir'] = np.where(df_combined2[0].str.startswith('ir'),1,0);
df_combined2['En_Pre_inter'] = np.where(df_combined2[0].str.startswith('inter'),1,0);
df_combined2['En_Pre_intra'] = np.where(df_combined2[0].str.startswith('intra'),1,0);
df_combined2['En_Pre_macro'] = np.where(df_combined2[0].str.startswith('macro'),1,0);
df_combined2['En_Pre_mono'] = np.where(df_combined2[0].str.startswith('mono'),1,0);
df_combined2['En_Pre_non'] = np.where(df_combined2[0].str.startswith('non'),1,0);
df_combined2['En_Pre_omni'] = np.where(df_combined2[0].str.startswith('omni'),1,0);
df_combined2['En_Pre_post'] = np.where(df_combined2[0].str.startswith('post'),1,0);
df_combined2['En_Pre_pre'] = np.where(df_combined2[0].str.startswith('pre'),1,0);
df_combined2['En_Pre_pro'] = np.where(df_combined2[0].str.startswith('pro'),1,0);
df_combined2['En_Pre_sub'] = np.where(df_combined2[0].str.startswith('sub'),1,0);
df_combined2['En_Pre_sym'] = np.where(df_combined2[0].str.startswith('sym'),1,0);
df_combined2['En_Pre_syn'] = np.where(df_combined2[0].str.startswith('syn'),1,0);
df_combined2['En_Pre_tele'] = np.where(df_combined2[0].str.startswith('tele'),1,0);
df_combined2['En_Pre_trans'] = np.where(df_combined2[0].str.startswith('trans'),1,0);
df_combined2['En_Pre_tri'] = np.where(df_combined2[0].str.startswith('tri'),1,0);
df_combined2['En_Pre_uni'] = np.where(df_combined2[0].str.startswith('uni'),1,0);
df_combined2['En_Pre_up'] = np.where(df_combined2[0].str.startswith('up'),1,0);
df_combined2['En_Pre_tele'] = np.where(df_combined2[0].str.startswith('tele'),1,0);
df_combined2['Singlish_suffix_a'] = np.where(df_combined2[0].str.endswith('a'),1,0);

df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0);
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined2['???'] = np.where(df_combined2[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined['???'] = np.where(df_combined[0].str.contains('???'),1,0)
df_combined2['a'] = np.where(df_combined2[0].str.contains('a'),1,0);
df_combined2['b'] = np.where(df_combined2[0].str.contains('b'),1,0);
df_combined2['c'] = np.where(df_combined2[0].str.contains('c'),1,0);
df_combined2['d'] = np.where(df_combined2[0].str.contains('d'),1,0);
df_combined2['e'] = np.where(df_combined2[0].str.contains('e'),1,0);
df_combined2['f'] = np.where(df_combined2[0].str.contains('f'),1,0);
df_combined2['g'] = np.where(df_combined2[0].str.contains('g'),1,0);
df_combined2['h'] = np.where(df_combined2[0].str.contains('h'),1,0);
df_combined2['i'] = np.where(df_combined2[0].str.contains('i'),1,0);
df_combined2['j'] = np.where(df_combined2[0].str.contains('j'),1,0);
df_combined2['k'] = np.where(df_combined2[0].str.contains('k'),1,0);
df_combined2['l'] = np.where(df_combined2[0].str.contains('l'),1,0);
df_combined2['m'] = np.where(df_combined2[0].str.contains('m'),1,0);
df_combined2['n'] = np.where(df_combined2[0].str.contains('n'),1,0);
df_combined2['o'] = np.where(df_combined2[0].str.contains('o'),1,0);
df_combined2['p'] = np.where(df_combined2[0].str.contains('p'),1,0);
df_combined2['q'] = np.where(df_combined2[0].str.contains('q'),1,0);
df_combined2['r'] = np.where(df_combined2[0].str.contains('r'),1,0)
df_combined2['s'] = np.where(df_combined2[0].str.contains('s'),1,0)
df_combined2['t'] = np.where(df_combined2[0].str.contains('t'),1,0)
df_combined2['u'] = np.where(df_combined2[0].str.contains('u'),1,0)
df_combined2['v'] = np.where(df_combined2[0].str.contains('v'),1,0)
df_combined2['w'] = np.where(df_combined2[0].str.contains('w'),1,0);
df_combined2['x'] = np.where(df_combined2[0].str.contains('x'),1,0);
df_combined2['y'] = np.where(df_combined2[0].str.contains('y'),1,0)
df_combined2['z'] = np.where(df_combined2[0].str.contains('z'),1,0)



X=df_combined2[['En_Suf_eer','En_Suf_ion','En_Suf_ism','En_Suf_ity','En_Suf_or','En_Suf_sion','En_Suf_ship',
                  'En_Suf_th','En_Suf_able','En_Suf_ible','En_Suf_al','En_Suf_ant','En_Suf_ary','En_Suf_ful',
                  'En_Suf_ic','En_Suf_ious','En_Suf_ous','En_Suf_y','En_Suf_ed','En_Suf_en','En_Suf_er','En_Suf_ing',
                  'En_Suf_ise','En_Suf_ly','En_Suf_ward','En__SUf_wise','En_Pre_anti','En_Pre_auto','En_Pre_circum',
                  'En_Pre_dis','En_Pre_ex','En_Pre_extra','En_Pre_hetero','En_Pre_homo','En_Pre_hyper','En_Pre_em',
                  'En_Pre_ir','En_Pre_inter','En_Pre_intra','En_Pre_macro','En_Pre_mono','En_Pre_non','En_Pre_omni',
                  'En_Pre_post','En_Pre_pre','En_Pre_pro','En_Pre_sub','En_Pre_sym','En_Pre_syn','En_Pre_tele','En_Pre_trans',
                  'En_Pre_tri','En_Pre_uni','En_Pre_up','En_Pre_tele',
                  '???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???', '???',	'???',	'???',	'???',	'???', '???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',	'???',
                  'a',	'b',	'c',	'd',	'e',	'f',	'g',	'h',	'i',	'j',	'k',	'l',	'm',	'n',	'o',	'p',	'q',	'r',	's',	't',	'u',	'v',	'w',	'x',	'y',	'z','Singlish_suffix_a']]  # Features
y=df_combined2['class']  # Labels
len(X)

  # Import train_test_split function


  # Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

  #Import Random Forest Model


  #Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

  #Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

  #Import scikit-learn metrics module for accuracy calculation

  # Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

df_predicted_vs_actual=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})

df = pd.DataFrame(df_predicted_vs_actual, columns=['Actual','Predicted'])

confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

  #Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

  #Train the model using the training sets
clf.fit(X_train, y_train)

  #Predict the response for test dataset
y_pred = clf.predict(X_test)

  #Import scikit-learn metrics module for accuracy calculation


  # Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
pickle.dump(clf, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


