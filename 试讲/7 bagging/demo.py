

from sklearn.metrics import f1_score
y_true = [0, 0, 0, 0,0, 0]
y_pred = [0, 0, 0, 0, 0, 0]
print(f1_score(y_true,y_pred,average='binary'))