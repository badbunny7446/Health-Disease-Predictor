
import pandas as pd
import matplotlib.pyplot as plt 

# Load the data
a = pd.read_csv('improved_disease_dataset.csv')
b = pd.read_csv('improved_disease_dataset.csv')

# Clean column names
a.columns = a.columns.str.strip()
b.columns = b.columns.str.strip()

# Remove duplicates
a.drop_duplicates(inplace=True)
b.drop_duplicates(inplace=True)

# Drop duplicate columns from 'b' except 'disease'
columns_to_drop = [col for col in b.columns if col != 'disease']
b = b[['disease']]  # Only keep 'disease' column

# Merge without duplication
ab = pd.merge(a, b, on='disease', how='left')

#Graph 1 find top disease
top_disease=ab['disease'].value_counts().head(10)

#Graph 2 find common symtoms
symtoms_col=[col for col in ab.columns if col!='disease']
symtoms_sum=ab[symtoms_col].sum().sort_values(ascending=False)

# Graph 3 Symtoms score
ab['symtoms_score']=ab[symtoms_col].sum(axis=1)

#Graph 4 average symtoms by disease
avg_symptom_by_disease=ab.groupby('disease')['symtoms_score'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
avg_symptom_by_disease.plot(kind='barh',color='green')
plt.title('Average symtoms by disease')
plt.xlabel('Average Symptom Score')
plt.ylabel('Disease')
plt.tight_layout()
plt.show()


#risk calculation 
risk=(avg_symptom_by_disease/len(symtoms_col))*100
avg_symptom_by_disease=ab.groupby('disease')['symtoms_score'].mean()
risk=risk.sort_values(ascending=False)

#Top 10 Diseases by Number of Cases
top=ab['disease'].value_counts().head(10)
plt.figure(figsize=(8,8))
plt.pie(top,labels=top.index,autopct='%1.1f%%')
plt.title('Top 10 Disease')
plt.tight_layout()
plt.show()

#Top 10 Common Symtoms
top_symtoms=symtoms_sum.head(10)
plt.figure(figsize=(8,8))
plt.pie(top_symtoms,labels=top_symtoms.index,autopct='%1.1f%%')
plt.title('Top 10 Common Symtoms')
plt.tight_layout()
plt.show()

#Top 10 High Risk Disease
risk=risk.head(10)
plt.figure(figsize=(8,8))
plt.pie(risk,labels=risk.index,autopct='%1.1f%%')
plt.title('Top 10 High Risk Disease')
plt.tight_layout()
plt.show()

#export ab to csv
ab.to_csv("final_health_data.csv", index=False)
risk.to_csv("disease_risk.csv")

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# set target input or output
symtoms_col = [col for col in ab.columns if col not in ['disease', 'symtoms_score']]
x = ab[symtoms_col]
y=ab['disease']

#split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#create model
model=RandomForestClassifier()

#Train model
model.fit(x_train,y_train)

# Test / predict
y_pred=model.predict(x_test)



#Check accurecy
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import pickle
pickle.dump(model, open("disease_model.pkl", "wb"))
