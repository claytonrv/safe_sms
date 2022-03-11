from time import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from trainer import ModelTrainer

classification_models = [
    LogisticRegression(multi_class='auto', solver='lbfgs'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    MultinomialNB()
]


def train(model, save):
    print("#################################################################")
    classification_model_name = type(model).__name__
    ModelTrainer.run(model, save)
    print(f"This iteration has run the following model: {classification_model_name}")
    print("#################################################################")


def train_all_models():
    print("BEGINNING THE SCRIPT")
    print()

    init_time = time()
    for model in classification_models:
        train(model=model, save=False)

    finish_time = time()
    print("FINISHING THE SCRIPT")
    print(f"Init time: {init_time}")
    print(f"Finish time: {finish_time}")
    print(f"Total execution time: {finish_time - init_time}")

def train_selected_algorithm(model):
    print("BEGINNING THE SCRIPT")
    print()

    init_time = time()
    train(model=model, save=True)

    finish_time = time()
    print("FINISHING THE SCRIPT")
    print(f"Init time: {init_time}")
    print(f"Finish time: {finish_time}")
    print(f"Total execution time: {finish_time - init_time}")


if __name__ == "__main__":
    # train_all_models()
    selected_algoritm = classification_models[0]
    train_selected_algorithm(selected_algoritm)