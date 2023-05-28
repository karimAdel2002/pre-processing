import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix

df = pd.read_csv("dataset.csv")
#--------------------------------------------------------------------------------
#                          Karim functions
#--------------------------------------------------------------------------------

print("---------------------------------------------------------------")
print("-----------------------Date-------------------------------")
print(df)

# ----------------------------- Clean first ------------------------------#
def mean() :
 df1 = pd.read_csv("data.csv")
 imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
 df1.iloc[:, 1:5] = pd.DataFrame(imputer.fit_transform(df1.iloc[:, 1:5]))
 imp = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value="empty")
 df1.iloc[:,0:5]= pd.DataFrame(imp.fit_transform(df1.iloc[:,0:5]))
 print('-------------Data After Imputer (mean)-------------')
 print(df1)

def median() :
 df2 = pd.read_csv("data.csv")
 imputer = SimpleImputer(missing_values=np.nan, strategy='median')
 df2.iloc[:, 1:5] = pd.DataFrame(imputer.fit_transform(df2.iloc[:, 1:5]))
 imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="empty")
 df2.iloc[:,0:5] = pd.DataFrame(imp.fit_transform(df2.iloc[:,0:5]))
 print('-------------Data After Imputer (median)-------------')
 print(df2)

def most_frequent() :
 # Imputer (most_frequent)
 df6 = pd.read_csv("data.csv")
 imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
 df6.iloc[:, 1:5] = pd.DataFrame(imputer.fit_transform(df6.iloc[:, 1:5]))
 imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="empty")
 df6.iloc[:,0:5] = pd.DataFrame(imp.fit_transform(df6.iloc[:,0:5]))
 print('-------------Data After Imputer (most_frequent)-------------')
 print(df6)

def constant() :
 df7 = pd.read_csv("data.csv")
 imputer = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value="empty")
 df7.iloc[:,0:5] = pd.DataFrame(imputer.fit_transform(df7.iloc[:,0:5]))
 print('-------------Data After Imputer (constant)-------------')
 print(df7)

def one_Hot_Decoder() :
 df0 = pd.read_csv("data.csv")
 ct = ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore'), [int(0)])], remainder='passthrough')
 df0 = pd.DataFrame(ct.fit_transform(df0))
 print('-------------Data After one Hot Decoder-------------')
 print(df0)

def LableDecoder() :
 df3 = pd.read_csv("data.csv")
 le = LabelEncoder()
 df3.iloc[:,-1] = pd.DataFrame( le.fit_transform(df3.iloc[:, -1]))
 print('-------------Data After LableDecoder-------------')
 print(df3)



df = pd.read_csv("dataset.csv")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df.iloc[:, 1:] = pd.DataFrame(imputer.fit_transform(df.iloc[:, 1:]))

le = LabelEncoder()
df.iloc[:, -1] = pd.DataFrame(le.fit_transform(df.iloc[:, -1]))
ct = ColumnTransformer([('encoder', OneHotEncoder(handle_unknown='ignore'), [int(0)])], remainder='passthrough')
df = pd.DataFrame(ct.fit_transform(df))
print('-------------Data After pre_processing-------------')
print(df )

#--------------------------------------------------------------------------------
#                          Fathy functions (linear & knn)
#--------------------------------------------------------------------------------


data = pd.read_csv("dataset.csv")
X = data.iloc[:, :-1].values
print(x)
y = data.iloc[:, -1].values
print(y)
train = 0.7
x_train,x_test,y_train,y_test= train_test_split(X,y,train_size=train)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print (prediction)

#--------------------------------------------------------------------------------
#                          Fatma functions (SVM)
#--------------------------------------------------------------------------------
def holdout() :
 data=pd.read_csv('dataset.csv')
 x = data.iloc[:, :-1]
 print(x.shape)
 y = data.iloc[:, -1]
 print(y.shape)
 x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
 # لكن لو اختار ال  kfold  ينفذ دا
 from sklearn.model_selection import KFold
 k = 5
 kfold = KFold(n_splits=k, random_state=None, shuffle=True)
 for train_index, test_index in kfold.split(x):
  x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
  print(x_train)
  y_train, y_test = y[train_index], y[test_index]
  break
 # rba ,linear هيبقى قيه اتنين راديو باتن واحد
 # لو اختار rba
 # Create SVM classifier object
 classifier = svm.SVC(kernel='rbf')
 # Train SVM classifier on training data
 classifier.fit(x_train, y_train)
 y_pred = classifier.predict(x_test)
 matrix = confusion_matrix(y_test, y_pred)
 print(matrix)
 # accuracy score
 from sklearn.metrics import accuracy_score
 acc = accuracy_score(y_test, y_pred)
 print("accuracy", acc)  # دا هيظهر ف شكل label فيه كلمه accuracy و
 # text=accفيه قيمه ال
 # precisio
 from sklearn.metrics import precision_score
 pre = precision_score(y_test, y_pred, average='micro')  # دا هيظهر ف شكل label فيه كلمه precision و
 print("precision", pre)  # text=preفيه قيمه ال
 # recall
 from sklearn.metrics import recall_score
 rec = recall_score(y_test, y_pred, average='micro')
 print("recall", rec)  # زى اللى فوق
 # f1-measure
 from sklearn.metrics import f1_score
 f1 = f1_score(y_test, y_pred, average='micro')
 print("measure", f1)
#
def k_fold() :
 # linearلكن لو اختار
 data = pd.read_csv('dataset.csv')
 x = data.iloc[:, :-1]
 y = data.iloc[:, -1]
 from sklearn.model_selection import train_test_split
 x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
 # Create SVM classifier object
 classifier = svm.SVC(kernel='linear')
 # Train SVM classifier on training data
 classifier.fit(x_train, y_train)
 y_pred = classifier.predict(x_test)
 from sklearn.metrics import confusion_matrix
 matrix = confusion_matrix(y_test, y_pred)
 print(matrix)
 # accuracy score
 from sklearn.metrics import accuracy_score
 acc = accuracy_score(y_test, y_pred)
 print("accuracy", acc)
 # precision
 from sklearn.metrics import precision_score
 pre = precision_score(y_test, y_pred, average='micro')
 print("precision", pre)
 # recall
 from sklearn.metrics import recall_score
 rec = recall_score(y_test, y_pred, average='micro')
 print("recall", rec)
 # f1-measure
 from sklearn.metrics import f1_score
 f1 = f1_score(y_test, y_pred, average='micro')
 print("measure", f1)


# linearRegression()
# knn()
# holdout()
# k_fold()
