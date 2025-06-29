import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import seaborn as sns

def load_and_view(n_rows=1, n_cols=4):
    digits = datasets.load_digits()
    
    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    return digits

def flatten_images(digits):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def create_classifier(gamma=0.001):
    return svm.SVC(gamma=gamma)

def split_data(data, target, test_size=0.5, random_state=None):
    return train_test_split(data, target, test_size=test_size, random_state=random_state)

def split_data_three_way(data, target, test_size=0.2, val_size=0.2, random_state=None):
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_predict(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return predicted

def evaluate_model(clf, X_train, y_train, X_val, y_val):
    clf.fit(X_train, y_train)
    val_score = clf.score(X_val, y_val)
    train_score = clf.score(X_train, y_train)
    return train_score, val_score

def plot_predictions(X_test, predicted, n_rows=1, n_cols=4):
    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

def print_classification_report(y_test, predicted, clf):
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

def plot_confusion_matrix(y_test, predicted):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return disp.confusion_matrix

def rebuild_classification_report_from_cm(confusion_matrix):
    from sklearn import metrics
    y_true = []
    y_pred = []
    
    for gt in range(len(confusion_matrix)):
        for pred in range(len(confusion_matrix)):
            y_true += [gt] * confusion_matrix[gt][pred]
            y_pred += [pred] * confusion_matrix[gt][pred]
    
    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )

def get_param_ranges():
    return {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly']
    }

def incremental_hyperparameter_search(X_train, y_train, X_val, y_val, param_ranges=None):
    if param_ranges is None:
        param_ranges = get_param_ranges()
    
    best_params = {}
    best_score = 0
    best_train_score = 0
    all_results = []
    
    total_combinations = len(param_ranges['C']) * len(param_ranges['gamma']) * len(param_ranges['kernel'])
    current_combination = 0
    
    print(f"Testing {total_combinations} parameter combinations...")
    print("Progress: ", end="")
    
    for c in param_ranges['C']:
        for gamma in param_ranges['gamma']:
            for kernel in param_ranges['kernel']:
                current_combination += 1
                
                if current_combination % max(1, total_combinations // 10) == 0:
                    print(f"{current_combination}/{total_combinations} ", end="", flush=True)
                
                try:
                    clf = svm.SVC(C=c, gamma=gamma, kernel=kernel)
                    train_score, val_score = evaluate_model(clf, X_train, y_train, X_val, y_val)
                    
                    result = {
                        'C': c,
                        'gamma': gamma, 
                        'kernel': kernel,
                        'train_score': train_score,
                        'val_score': val_score,
                        'overfitting': train_score - val_score
                    }
                    all_results.append(result)
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_train_score = train_score
                        best_params = {'C': c, 'gamma': gamma, 'kernel': kernel}
                        
                except Exception as e:
                    print(f"\nError with params C={c}, gamma={gamma}, kernel={kernel}: {e}")
                    continue
    
    print("\nSearch completed!")
    return best_params, best_score, best_train_score, all_results

def print_top_results(results, top_n=5):
    df = pd.DataFrame(results)
    top_results = df.nlargest(top_n, 'val_score')
    
    print(f"\nTop {top_n} parameter combinations:")
    print("-" * 80)
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"{i}. C={row['C']}, gamma={row['gamma']}, kernel='{row['kernel']}'")
        print(f"   Validation Score: {row['val_score']:.4f}")
        print(f"   Training Score: {row['train_score']:.4f}")
        print(f"   Overfitting: {row['overfitting']:.4f}")
        print()

def create_best_model(best_params):
    return svm.SVC(**best_params)
