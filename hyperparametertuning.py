"""
================================
Incremental Hyperparameter Tuning for Digit Classification
================================

This script demonstrates incremental hyperparameter tuning using manual 
for-loop search with train/validation/test splits.

"""

import matplotlib.pyplot as plt
import time
import utils

def run_incremental_hyperparameter_tuning():
    print("Loading and preparing data...")
    digits = utils.load_and_view()
    data, target = utils.flatten_images(digits)
    
    print("Splitting data into train/validation/test sets (60%/20%/20%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data_three_way(
        data, target, test_size=0.2, val_size=0.2, random_state=42
    )
    
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")  
    print(f"Test set size: {len(X_test)}")
    
    print("\n" + "="*60)
    print("INCREMENTAL HYPERPARAMETER SEARCH")
    print("="*60)
    
    start_time = time.time()
    best_params, best_score, best_train_score, all_results = utils.incremental_hyperparameter_search(
        X_train, y_train, X_val, y_val
    )
    search_time = time.time() - start_time
    
    print(f"\nSearch completed in {search_time:.2f} seconds")
    print(f"\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Best training score: {best_train_score:.4f}")
    print(f"Overfitting (Train - Val): {best_train_score - best_score:.4f}")
    
    utils.print_top_results(all_results, top_n=5)
    
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    best_model = utils.create_best_model(best_params)
    test_predictions = utils.train_and_predict(best_model, X_train, y_train, X_test)
    test_score = best_model.score(X_test, y_test)
    
    print(f"Final test accuracy: {test_score:.4f}")
    
    print("\nDetailed classification report on test set:")
    utils.print_classification_report(y_test, test_predictions, best_model)
    
    print("\n" + "="*60)
    print("BASIC VISUALIZATION")
    print("="*60)
    
    print("Plotting test predictions...")
    utils.plot_predictions(X_test, test_predictions)
    
    print("Plotting confusion matrix...")
    utils.plot_confusion_matrix(y_test, test_predictions)
    
    print("\nPerformance Summary:")
    print(f"  Training Score: {best_train_score:.4f}")
    print(f"  Validation Score: {best_score:.4f}")  
    print(f"  Test Score: {test_score:.4f}")
    print(f"  Search Time: {search_time:.2f} seconds")
    
    return {
        'best_params': best_params,
        'best_model': best_model,
        'all_results': all_results,
        'scores': {
            'train': best_train_score,
            'validation': best_score,
            'test': test_score
        }
    }

if __name__ == "__main__":
    results = run_incremental_hyperparameter_tuning()
    plt.show()
