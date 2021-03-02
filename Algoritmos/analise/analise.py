import pandas as pd;

svm_linear = pd.read_csv("C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Algoritmos/svm/diferentes.csv", index_col=0);
svm_rbf = pd.read_csv("C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Algoritmos/svm_rbf/diferentes.csv", index_col=0);
knn = pd.read_csv("C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Algoritmos/knn/diferentes.csv", index_col=0);
random_forest = pd.read_csv("C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Algoritmos/random_forest/diferentes.csv", index_col=0);
gradient_boosting = pd.read_csv("C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Algoritmos/gradient_boosting/diferentes.csv", index_col=0);
labels = pd.read_csv('C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Base de dados/Test dataset/labels_langone.csv', index_col=0);

d = {'SVM(Linear)': svm_linear['SVM(Linear)'], 'SVM(RBF)': svm_rbf['SVM(RBF)'], 'KNN': knn['KNN'], 'RandomForest': random_forest['RandomForest'], 'XGBoost': gradient_boosting['XGBoost'], 'PAM50': labels['PAM50']};

total = pd.DataFrame(data=d);

pd.set_option('display.max_rows', None);

basal = total.loc[total['PAM50'] == 'Basal'];
print(basal);
print("");
Her2 = total.loc[total['PAM50'] == 'Her2'];
print(Her2);
print("");
LumA = total.loc[total['PAM50'] == 'LumA'];
print(LumA);
print("");
LumB = total.loc[total['PAM50'] == 'LumB'];
print(LumB);
print("");

abnormal = [];

for i in range(77):
 if((int(total['SVM(Linear)'].iloc[i] != total['PAM50'].iloc[i]) + int(total['SVM(RBF)'].iloc[i] != total['PAM50'].iloc[i]) + int(total['KNN'].iloc[i] != total['PAM50'].iloc[i]) + int(total['RandomForest'].iloc[i] != total['PAM50'].iloc[i]) + int(total['XGBoost'].iloc[i] != total['PAM50'].iloc[i])) >= 4):
  abnormal += [i];

if (len(abnormal) > 0):
    print("As seguinte amostras foram preditas de forma errada por pelo menos 4 dos 5 algoritmos utilizados:")
    print(abnormal);

total.to_csv('diferentes.csv');