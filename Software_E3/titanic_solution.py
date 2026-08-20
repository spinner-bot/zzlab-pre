# titanic_solution.py —— 泰坦尼克号生还预测（Kaggle 竞赛复现）
# 流程：数据清洗 -> 特征工程 -> 多模型对比(>=2类) -> 网格搜索调参 -> 生成 submission.csv
# 依赖：pip install pandas numpy scikit-learn
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data', 'titanic.csv')


def load_and_clean():
    df = pd.read_csv(DATA)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)   # 从姓名提取称谓
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    return df


def featurize(df):
    feat = pd.get_dummies(df[['Sex', 'Embarked', 'Title']], drop_first=True).astype(int)
    feat['Pclass'] = df['Pclass']
    feat['Age'] = df['Age']
    feat['Fare'] = df['Fare']
    feat['HasCabin'] = df['HasCabin']
    feat['FamilySize'] = df['FamilySize']
    feat['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return feat


def main():
    df = load_and_clean()
    X = featurize(df)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ---- 1. 多种不同类别模型对比 ----
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
    }
    print('=== 模型对比（交叉验证）===')
    scores = {}
    for name, m in models.items():
        cv = cross_val_score(m, X_train, y_train, cv=5).mean()
        scores[name] = cv
        print(f'  {name:20s} CV acc = {cv:.4f}')
    best_model = max(scores, key=scores.get)
    print(f'最佳基础模型: {best_model}\n')

    # ---- 2. 网格搜索调参 ----
    print('=== 网格搜索调参 ===')
    if best_model == 'RandomForest':
        grid = {'n_estimators': [100, 200], 'max_depth': [5, 8, None],
                'min_samples_split': [2, 5]}
        base = RandomForestClassifier(random_state=42)
    elif best_model == 'GradientBoosting':
        grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]}
        base = GradientBoostingClassifier(random_state=42)
    else:
        grid = {'C': [0.01, 0.1, 1, 10]}
        base = LogisticRegression(max_iter=1000, random_state=42)
    gs = GridSearchCV(base, grid, cv=5, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f'  最优参数: {gs.best_params_}  最优CV acc = {gs.best_score_:.4f}')

    # ---- 3. 在留出集上评估 ----
    y_pred = gs.best_estimator_.predict(X_test)
    print(f'\n留出集测试准确率: {accuracy_score(y_test, y_pred):.4f}')

    # ---- 4. 生成 Kaggle 格式 submission.csv ----
    # 说明：此处用留出验证集作为"测试集"演示生成流程；接入真实 test.csv 时替换即可。
    sub = pd.DataFrame({'PassengerId': df.loc[X_test.index, 'PassengerId'],
                        'Survived': y_pred})
    out = os.path.join(BASE, 'submission.csv')
    sub.to_csv(out, index=False)
    print(f'submission.csv 已生成: {out}（{len(sub)} 行）')


if __name__ == '__main__':
    main()
