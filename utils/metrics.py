from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def calculate_metrics(y_true, y_pred, target_names=None):
    """
    Computes standard classification evaluation metrics.
    Accuracy, Precision, Recall, and F1-Score metrics are returned as a dictionary.
    Includes a verbose sklearn classification report for analysis.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate Precision, Recall, F1 for Macro average (weighted identically irrespective of class sizes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Confusion matrix for identifying misclassifications and false positives specifically
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'report': report
    }
