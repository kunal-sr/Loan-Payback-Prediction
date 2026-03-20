import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape, test.shape)
train.head()

train["loan_paid_back"].value_counts(normalize=True)

X = train.drop(columns=["loan_paid_back", "id"])
y = train["loan_paid_back"]

X_test = test.drop(columns=["id"])

X.isnull().sum()

X = X.fillna(-999)
X_test = X_test.fillna(-999)

from sklearn.preprocessing import LabelEncoder

cat_cols = X.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

    from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

val_preds = model.predict_proba(X_val)[:, 1]
score = roc_auc_score(y_val, val_preds)

print("Validation AUC:", score)

model.fit(X, y)

test_preds = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_preds
})

submission.to_csv("submission.csv", index=False)