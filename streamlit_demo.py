from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import re
import pandas as pd
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="BigData Team: streamlit demo",
    page_icon="🦁",
    layout="wide",
)

st.title("🦁 BigData Team: Streamlit Demo")
st.header("01. Введение в библиотеки ML", divider=True)
st.subheader("Titanic dataset, train sample", divider=True)

train = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
st.write(train)


def preprocess_data(data, has_survived=True):
    columns_to_drop = ["Ticket", "PassengerId"]
    df = data.copy()

    survived = None
    if has_survived and "Survived" in df.columns:
        survived = df["Survived"].copy()

    df.drop(columns_to_drop, axis=1, inplace=True)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"] = df["Fare"].astype(int)

    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    df["Cabin"] = df["Cabin"].fillna("U0")
    df["Deck"] = df["Cabin"].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    df["Deck"] = df["Deck"].map(deck)
    df["Deck"] = df["Deck"].fillna(0).astype(int)

    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Cap",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].map(titles).fillna(0).astype(int)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-1, 0, 5, 12, 18, 35, 60, 100],
        labels=[
            "Unknown",
            "Baby",
            "Child",
            "Teenager",
            "YoungAdult",
            "Adult",
            "Senior",
        ],
    )
    df["FareGroup"] = pd.qcut(
        df["Fare"], 4, labels=["Low", "Medium_Low", "Medium_High", "High"]
    )

    df["AgePclassInteraction"] = df["Age"] * df["Pclass"]
    df["FamilySizePclass"] = df["FamilySize"] * df["Pclass"]
    df["WomenAndChildren"] = ((df["Sex"] == "female") | (df["Age"] < 16)).astype(int)

    df["Title_Raw"] = df.Name.str.split(",").str[1].str.split(".").str[0].str.strip()
    df["IsWomanOrBoy"] = ((df.Title_Raw == "Master") | (df.Sex == "female")).astype(int)
    df["LastName"] = df.Name.str.split(",").str[0]

    family_groups = df.groupby("LastName")

    if survived is not None:
        df["Survived"] = survived

        df["WomanOrBoyCount"] = (
            df.groupby("LastName")["IsWomanOrBoy"].transform("sum") - df["IsWomanOrBoy"]
        )

        family_survived = df.groupby("LastName")["Survived"].transform(
            lambda x: x.mul(df.loc[x.index, "IsWomanOrBoy"]).sum()
        )
        df["FamilySurvivedCount"] = (
            family_survived - df["Survived"] * df["IsWomanOrBoy"]
        )

        df["WomanOrBoySurvived"] = df["FamilySurvivedCount"] / df[
            "WomanOrBoyCount"
        ].replace(0, np.nan)
        df["WomanOrBoySurvived"].fillna(0, inplace=True)
    else:
        df["WomanOrBoyCount"] = (
            df.groupby("LastName")["IsWomanOrBoy"].transform("sum") - df["IsWomanOrBoy"]
        )
        df["FamilySurvivedCount"] = 0
        df["WomanOrBoySurvived"] = 0

    df["Alone"] = (df["WomanOrBoyCount"] == 0).astype(int)

    df["FamilyMemberCount"] = df.groupby("LastName")["LastName"].transform("count") - 1

    df["WomenAndChildrenFirst"] = ((df["Sex"] == "female") | (df["Age"] < 14)).astype(
        int
    )

    df["Sex"] = df["Sex"].astype(str)
    df["Embarked"] = df["Embarked"].astype(str)
    df["AgeGroup"] = df["AgeGroup"].astype(str)
    df["FareGroup"] = df["FareGroup"].astype(str)
    df["Pclass"] = df["Pclass"].astype(str)

    columns_to_drop = ["Name", "Cabin", "Title_Raw", "LastName"]
    if has_survived and "Survived" in df.columns:
        columns_to_drop.append("Survived")

    df.drop(columns_to_drop, axis=1, inplace=True)

    return df


train_processed = preprocess_data(train)

y_train = train["Survived"]

categorical_features = ["Pclass", "Sex", "Embarked", "AgeGroup", "FareGroup", "Title"]
numerical_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "AgePclassInteraction",
    "FamilySizePclass",
    "WomenAndChildren",
    "Deck",
    "WomenAndChildren",
    "IsWomanOrBoy",
    "WomanOrBoyCount",
    "FamilySurvivedCount",
    "WomanOrBoySurvived",
    "Alone",
    "FamilyMemberCount",
    "WomenAndChildrenFirst",
]

train.drop("Survived", axis=1, inplace=True)
encoder = OneHotEncoder(sparse_output=False, drop="first")
train_cat_encoded = encoder.fit_transform(train_processed[categorical_features])

train_cat_encoded_df = pd.DataFrame(
    train_cat_encoded,
    columns=encoder.get_feature_names_out(categorical_features),
    index=train_processed.index,
)

train_new = pd.concat(
    [train_processed[numerical_features], train_cat_encoded_df], axis=1
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_new)

st.subheader("В поисках лучшей kNN модели", divider=True)

col1, col_, col2 = st.columns([0.5, 0.1, 0.4])

with col1:
    n_neihbors = st.slider("Количество соседей", value=5, min_value=1, max_value=25)
    weights = st.selectbox("weights", options=("uniform", "distance"))
    p = st.number_input("distance_p(ower degree)", value=2, min_value=1)
    st.markdown(
        "Больше о параматрах kNN в sklearn: [по ссылке](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)"
    )

    knn = KNeighborsClassifier(n_neighbors=n_neihbors, weights=weights, p=p)
    cross_val_scores = cross_val_score(
        knn, X_train_scaled, y_train, scoring="accuracy", cv=5
    )

with col2:
    cross_val_score_mean = cross_val_scores.mean()
    delta = None
    if "previous_score" in st.session_state:
        delta = cross_val_score_mean - st.session_state["previous_score"]
        delta = round(delta, 3)

    st.write("Результаты")
    st.metric(
        "Accuracy (mean over 5 folds)",
        round(cross_val_score_mean, 3),
        delta,
        border=True,
    )
    st.write({"score_mean": cross_val_score_mean, "score_std": cross_val_scores.std()})

    st.session_state["previous_score"] = cross_val_score_mean
