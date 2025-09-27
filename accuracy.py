import pandas as pd

df = pd.read_csv("runs/detect/train/results.csv")

best_epoch = df["metrics/mAP50(B)"].idxmax()
print("Best Epoch:", df.loc[best_epoch, "epoch"])
print("Best mAP50:", df.loc[best_epoch, "metrics/mAP50(B)"])
print("Best mAP50-95:", df.loc[best_epoch, "metrics/mAP50-95(B)"])
print("Precision:", df.loc[best_epoch, "metrics/precision(B)"])
print("Recall:", df.loc[best_epoch, "metrics/recall(B)"])


#Precision= True Positives / (False Positives + True Positives​)