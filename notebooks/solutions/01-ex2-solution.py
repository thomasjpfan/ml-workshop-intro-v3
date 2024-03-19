from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

wine = load_wine(as_frame=True)
X, y = wine.data, wine.target

y

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

knn.score(X_test, y_test)

rfc.score(X_test, y_test)

log_reg.score(X_test, y_test)
