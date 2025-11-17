import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def LogisticRegressionTest(model, test):
    predictions = model.predict(test)
    return predictions

#Linear regression model
def LogisticRegressionTrain(X, y):
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nLogistic Regression\nAccuracy:", round(accuracy_score(y_test, y_pred),3))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

def RandomForestTest(model, test):
    predictions = model.predict(test)
    return predictions

#Gradient Descent model
def RandomForestTrain(X, y):
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rfmodel = RandomForestClassifier(random_state=42)

    # 5-fold cross-validation
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(rfmodel, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)  # fit on training set only

    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    best_rf = grid_search.best_estimator_

    y_pred = best_rf.predict(X_test)

    print("\nRandom Forest\nAccuracy:", round(accuracy_score(y_test, y_pred),2))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return best_rf

def XGBtrain(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier(random_state = 42, eval_metric = 'logloss')
    

    # 5-fold cross-validation
    param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
    }

    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", round(grid_search.best_score_, 3))

    # Evaluate on hold-out test set
    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)

    print("\nXGBoost Accuracy:", round(accuracy_score(y_test, y_pred), 2))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return best_xgb

def XGBTest(model, test):
    predictions = model.predict(test)
    return predictions



#Typical data preprocessing function, filling NA vals
def preprocessing(df):
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 
                                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                    'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
    df['Age_Pclass'] = df['Age'] * df['Pclass']
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['Pclass_Sex'] = df['Pclass'].astype(str) + "_" + df['Sex'].astype(str)
    df = pd.get_dummies(df, columns=['Pclass_Sex'], drop_first=True)
    
    features = df[['Pclass', 'Sex', 'Age', 'Parch', 'SibSp', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'Age_Pclass', 'FarePerPerson'] + [col for col in df.columns if col.startswith('Pclass_Sex')]]
    if 'Survived' in df:
        target = df['Survived']
        return features, target
    else:
        return features
    
def Export(predictions, test):
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('XGBoost.csv', index=False)


#reading train set
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')

#creating features which will be used in model
train, target = preprocessing(dftrain)

#print(train.head())

test = preprocessing(dftest)

#lrmodel = LogisticRegressionTrain(train, target)
#rfmodel = RandomForestTrain(train, target)
xgbmodel = XGBtrain(train, target)


#lrpredictions = LogisticRegressionTest(lrmodel, test)
#rfpredictions = RandomForestTest(rfmodel, test)
xgbpredictions = XGBTest(xgbmodel, test)

Export(xgbpredictions, dftest)


'''importances = rfmodel.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(10,6))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.show()'''









