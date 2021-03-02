import pandas as pd;

import csv;

from sklearn.ensemble import RandomForestClassifier;

from sklearn.metrics import classification_report;
from sklearn.metrics import f1_score;
from sklearn.metrics import precision_score;
from sklearn.metrics import recall_score;
from sklearn.metrics import confusion_matrix;

import seaborn as sns;
import matplotlib.pyplot as plt;

#Lendo os datasets de treino.
df_cptac = pd.read_csv("C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Base de dados/Train dataset/cptac.csv",index_col = 0);
df_cptac_subtype = pd.read_csv("C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Base de dados/Train dataset/cptac_subtype.csv", index_col = 0);

#Adicionando coluna referente ao PAM50
df_cptac['PAM50'] = df_cptac_subtype['PAM50'];

#Divisão do dataset em relação a cada subtipo

cptac_basal = df_cptac.loc[df_cptac['PAM50'] == 'Basal'];
cptac_basal = cptac_basal.drop(['PAM50'], axis=1);

cptac_her2 = df_cptac.loc[df_cptac['PAM50'] == 'Her2'];
cptac_her2 = cptac_her2.drop(['PAM50'], axis=1);

cptac_luma = df_cptac.loc[df_cptac['PAM50'] == 'LumA'];
cptac_luma = cptac_luma.drop(['PAM50'], axis=1);

cptac_lumb = df_cptac.loc[df_cptac['PAM50'] == 'LumB'];
cptac_lumb = cptac_lumb.drop(['PAM50'], axis=1);

#Leitura do dataset de teste
df_rna = pd.read_csv('C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Base de dados/Test dataset/langone.csv', index_col=0);
df_rna_subtype = pd.read_csv('C:/Users/joaof/Documents/Joao/Ufam/Pibic/Cancer/Base de dados/Test dataset/labels_langone.csv', index_col=0);

#Processamento miscelânico
df_rna = df_rna.T;
df_rna.rename(columns={'ORC6L':'ORC6'}, inplace=True);

#igualando a quantidade de colunas entre os datasets
common_columns = df_cptac.columns & df_rna.columns;
df_cptac = df_cptac[common_columns];
df_rna = df_rna[common_columns];

#Selecionando os genes a serem classificados
pam50_genes = ['ACTR3B', 'ANLN', 'BAG1', 'BCL2', 'BIRC5', 'BLVRA', 'CCNB1', 'CCNE1',
       'CDC20', 'CDC6', 'CDH3', 'CENPF', 'CEP55', 'CXXC5', 'EGFR', 'ERBB2',
       'ESR1', 'EXO1', 'FGFR4', 'FOXA1', 'FOXC1', 'GPR160', 'GRB7', 'KIF2C',
       'KRT14', 'KRT17', 'KRT5', 'MAPT', 'MDM2', 'MELK', 'MIA', 'MKI67',
       'MLPH', 'MMP11', 'MYBL2', 'MYC', 'NAT1', 'NDC80', 'NUF2', 'ORC6', 'PGR',
       'PHGDH', 'PTTG1', 'RRM2', 'SFRP1', 'SLC39A6', 'TMEM45B', 'TYMS',
       'UBE2C', 'UBE2T'];

#ajustando o dataset de treino
df_cptac_final = df_cptac[pam50_genes];
df_cptac_final = df_cptac_final.fillna(df_cptac_final.mean());

#Ajustando o dataset de teste
df_rna_final = df_rna[pam50_genes];
df_rna_final = df_rna_final.fillna(df_rna_final.mean());

#selecionando os valores de treino(x_train e y_train) e de teste(X_test e y_test)
x_train = df_cptac_final;

vals = {'Basal': 1, 'Her2': 2, 'LumA': 3, 'LumB': 4};

df_classes_cptac = df_cptac_subtype.replace({'PAM50': vals});
df_classes_cptac = df_classes_cptac['PAM50'].values;
y_train = df_classes_cptac;

x_test = df_rna_final;

df_classes = df_rna_subtype.replace({'PAM50': vals});
y_test = df_classes['PAM50'].values;
#Fim do tratamento de dados

##Algoritmo de classificação utilizando Random Forest
clf = RandomForestClassifier(n_estimators = 20, random_state=0);
clf.fit(x_train, y_train);

pred_amostra = clf.predict(x_test);

##Salvando em disco quais genes estão diferentes
rev = {1:'Basal', 2:'Her2', 3:'LumA', 4:'LumB'};
diferentes = pred_amostra;

d = {'RandomForest': diferentes}

result = pd.DataFrame(data=d);
result = result.replace({'RandomForest': rev});
result.to_csv('diferentes.csv');

##Plotando a matriz de confusão com os resultados encontrados##
cfm1 = confusion_matrix(y_test, pred_amostra);
print(classification_report(y_test,pred_amostra));
f1 = f1_score(y_test, pred_amostra, average=None);
precision = precision_score(y_test, pred_amostra, average=None);
recall = recall_score(y_test, pred_amostra, average=None);

df_cm = pd.DataFrame(cfm1, range(4), range(4));
df_cm.index = ['Basal', 'Her2', 'LumA', 'LumB'];
df_cm.columns = ['Basal', 'Her2', 'LumA', 'LumB'];

sns.set(font_scale=1.4);

sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cbar=False, cmap="YlGnBu");

plt.show();