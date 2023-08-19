from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.decomposition import PCA 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import plot_roc_curve, accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_curve, plot_roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import time
from tqdm import tqdm

#Get the data
features = pd.read_table('uterus_rnaseq_VST.txt', sep="\t", index_col=0)
features = features[features['label']!="G2"]
features=features.replace("G1",0)
features=features.replace("G3",1)

print(features["label"].value_counts())

labels = np.array(features['label'])

features= features.drop('label', axis = 1)


#train and test

features1 = StandardScaler().fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features1, labels, test_size=0.2, random_state=1)
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

svc = SVC(kernel='sigmoid', cache_size=6000, probability=True).fit(X_train, y_train)

y_pred_proba = svc.predict_proba(X_test)[:,1]
acc = svc.score(X_test, y_test)
auc_1 = roc_auc_score(y_test, y_pred_proba)


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax.legend(loc = 'lower right')
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("sigmoid-s_1/test_roc.pdf")
#plt.show()
plt.close('all')
print("best auc {}\nbest accuracy {}".format(auc_1, acc))

#Histological grade prediction for G2 group
featuresG2 = pd.read_table('uterus_rnaseq_VST_G2.txt', sep="\t", index_col=0)
print(featuresG2["label"].value_counts())

y_test_G2 = np.array(featuresG2['label'])
X_test_G2= featuresG2.drop('label', axis = 1)

X_test_G2_ = StandardScaler().fit_transform(X_test_G2)

y_pred_proba=pd.DataFrame(svc.predict_proba(X_test_G2_)[:,1])
y_pred_proba.columns=["pred_proba"]


y_pred_proba["samples"]=X_test_G2.index.values
y_pred_proba.to_csv("sigmoid-s_1/G2_preds.txt", sep='\t', index=False)


#cross-validation 
cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=1)
aucs = []
accs = []
tprs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
i=1

for train_idx, val_idx in tqdm(cv_outer.split(X_train, y_train)):
    train_data, val_data = X_train[train_idx], X_train[val_idx]
    train_target, val_target = y_train[train_idx], y_train[val_idx]
    
    svc = SVC(kernel='sigmoid', cache_size=6000, probability=True).fit(train_data, train_target)

    y_pred_proba = svc.predict_proba(val_data)[:,1]
    acc = svc.score(val_data, val_target)
    auc_1 = roc_auc_score(val_target, y_pred_proba)
    accs.append(acc)
    
    viz = plot_roc_curve(svc, val_data, val_target, name='ROC fold {}'.format(i), alpha=0.5, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    i=i+1
    
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
print("Mean AUC: ", mean_auc)
mean_acc = np.mean(accs)
print("Mean ACC:", mean_acc)

std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
plt.show()
plt.savefig("sigmoid-s_1/crossval_roc.pdf")

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax.legend(loc="lower right")
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
#plt.show()
plt.savefig("sigmoid-s_1/crossval_roc_mean_only.pdf")
plt.close('all')


#train and test with 12 genes
gene_mask = ['ENSG00000104371', 'ENSG00000178053', 'ENSG00000255545','ENSG00000124205', 'ENSG00000227097', 'ENSG00000148702','ENSG00000242265', 'ENSG00000166426', 'ENSG00000154277','ENSG00000171956', 'ENSG00000172005', 'ENSG00000227063']
#gene_mask = ['ENSG00000124205', 'ENSG00000163501', 'ENSG00000148702', 'ENSG00000261122','ENSG00000166426', 'ENSG00000104371', 'ENSG00000255545', 'ENSG00000124939', 'ENSG00000007306', 'ENSG00000123838', 'ENSG00000161055', 'ENSG00000134873']
#gene_mask = ['ENSG00000163501', 'ENSG00000242265', 'ENSG00000148702', 'ENSG00000171956', 'ENSG00000166426', 'ENSG00000104371', 'ENSG00000255545', 'ENSG00000178053', 'ENSG00000124939', 'ENSG00000172005', 'ENSG00000161055', 'ENSG00000134873']

features2 = features[gene_mask]
features2 = StandardScaler().fit_transform(features2)
X_train, X_test, y_train, y_test = train_test_split(features2, labels, test_size=0.2, random_state=1)
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)


svc = SVC(kernel='poly', cache_size=6000, probability=True).fit(X_train, y_train)

y_pred_proba = svc.predict_proba(X_test)[:,1]
acc = svc.score(X_test, y_test)
auc_1 = roc_auc_score(y_test, y_pred_proba)


fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax.legend(loc = 'lower right')
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("sigmoid-s_1/test_roc_12.pdf")
#plt.show()
plt.close('all')
print("best auc {}\nbest accuracy {}".format(auc_1, acc))

#Histological grade prediction for G2 group with 12 genes
featuresG2 = pd.read_table('uterus_rnaseq_VST_G2.txt', sep="\t", index_col=0)
print(featuresG2["label"].value_counts())

y_test_G2 = np.array(featuresG2['label'])
X_test_G2= featuresG2.drop('label', axis = 1)
X_test_G2_= X_test_G2[gene_mask]
X_test_G2_ = StandardScaler().fit_transform(X_test_G2_)

y_pred_proba=pd.DataFrame(svc.predict_proba(X_test_G2_)[:,1])
y_pred_proba.columns=["pred_proba"]


y_pred_proba["samples"]=X_test_G2.index.values
y_pred_proba.to_csv("sigmoid-s_1/G2_preds_12.txt", sep='\t', index=False)
