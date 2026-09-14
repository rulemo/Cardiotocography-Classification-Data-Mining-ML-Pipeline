import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('cardiotocography_v2.csv')

print("=" * 60)
print("TASK 1: DATA MINING - BASIC STATISTICS")
print("=" * 60)

print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"Target column: CLASS")
print()

# Basic info about the dataset
print("Dataset Info:")
print(df.info())
print()

# Check for missing values
print("Missing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Percentage of missing values: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
print()

# Target class distribution
print("Class distribution:")
class_counts = df['CLASS'].value_counts().sort_index()
print(class_counts)
print()
print("Class distribution percentages:")
class_percentages = df['CLASS'].value_counts(normalize=True).sort_index() * 100
print(class_percentages.round(2))
print()

# Basic statistics for features
print("Basic statistics for numerical features:")
print(df.describe().round(2))

# Visualizations
plt.figure(figsize=(15, 10))

# Class distribution plot
plt.subplot(2, 3, 1)
class_counts.plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Missing values plot
plt.subplot(2, 3, 2)
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
plt.bar(range(len(missing_data)), missing_data.values, color='coral')
plt.title('Missing Values per Feature')
plt.xlabel('Features')
plt.ylabel('Missing Count')
plt.xticks(range(len(missing_data)), missing_data.index, rotation=45)

# Correlation heatmap
plt.subplot(2, 3, 3)
key_features = ['LB', 'ASTV', 'MSTV', 'Width', 'Mean', 'Median', 'CLASS']
corr_matrix = df[key_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix (Key Features)')

plt.tight_layout()
plt.show()

# TASK 2
print("=" * 60)
print("TASK 2: DATA PREPARATION")
print("=" * 60)

# Separate features and target
X = df.drop('CLASS', axis=1)
y = df['CLASS']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print()

# Handle missing values using imputation
print("=== HANDLING MISSING VALUES ===")
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

print("Missing values after imputation:")
print("Training set:", X_train_imputed.isnull().sum().sum())
print("Test set:", X_test_imputed.isnull().sum().sum())
print()

# Method 1: Standardization
print("=== METHOD 1: STANDARDIZATION ===")
scaler_std = StandardScaler()
X_train_std = pd.DataFrame(scaler_std.fit_transform(X_train_imputed), 
                          columns=X.columns, index=X_train_imputed.index)
X_test_std = pd.DataFrame(scaler_std.transform(X_test_imputed), 
                         columns=X.columns, index=X_test_imputed.index)

# Method 2: Min-Max Normalization
print("=== METHOD 2: MIN-MAX NORMALIZATION ===")
scaler_minmax = MinMaxScaler()
X_train_norm = pd.DataFrame(scaler_minmax.fit_transform(X_train_imputed), 
                           columns=X.columns, index=X_train_imputed.index)
X_test_norm = pd.DataFrame(scaler_minmax.transform(X_test_imputed), 
                          columns=X.columns, index=X_test_imputed.index)

# Method 3: PCA
print("=== METHOD 3: PCA DIMENSIONALITY REDUCTION ===")
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print(f"Original features: {X_train_std.shape[1]}")
print(f"PCA components: {X_train_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

# Store all prepared datasets
datasets = {
    'Original (Imputed)': (X_train_imputed, X_test_imputed),
    'Standardized': (X_train_std, X_test_std),
    'Normalized': (X_train_norm, X_test_norm),
    'PCA': (X_train_pca, X_test_pca)
}

# TASK 3

print("=" * 60)
print("TASK 3: CLASSIFICATION EXPERIMENTS")
print("=" * 60)

def evaluate_classifier(name, clf, X_train, X_test, y_train, y_test):
    """Evaluate classifier and return metrics"""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Define classifiers with different hyperparameters
classifiers = {
    # Naive Bayes with different parameters
    'NB_default': GaussianNB(),
    'NB_var_smoothing_1e-8': GaussianNB(var_smoothing=1e-8),
    'NB_var_smoothing_1e-6': GaussianNB(var_smoothing=1e-6),
    
    # Decision Tree with different parameters
    'DT_default': DecisionTreeClassifier(random_state=42),
    'DT_max_depth_5': DecisionTreeClassifier(max_depth=5, random_state=42),
    'DT_max_depth_10_min_samples_10': DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42),
}

# Initialize results storage
results = {}

# Test each dataset with each classifier
for dataset_name, (X_train_data, X_test_data) in datasets.items():
    print(f"Dataset: {dataset_name}")
    print("-" * 40)
    
    results[dataset_name] = {}
    
    # Test all classifiers
    for clf_name, clf in classifiers.items():
        metrics = evaluate_classifier(clf_name, clf, X_train_data, X_test_data, y_train, y_test)
        results[dataset_name][clf_name] = metrics
        
        print(f"{clf_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1_score']:.4f}")
        print()
    
    print()

    # TASK 4


print("=" * 60)
print("TASK 4: CLASSIFICATION EVALUATION & RESULTS SUMMARY")
print("=" * 60)

# Create comprehensive results table
results_list = []
for dataset_name, classifiers in results.items():
    for clf_name, metrics in classifiers.items():
        results_list.append({
            'Dataset': dataset_name,
            'Classifier': clf_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })

results_df = pd.DataFrame(results_list)

# Round values for better display
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    results_df[col] = results_df[col].round(4)

print("=== COMPREHENSIVE RESULTS TABLE ===")
print(results_df.to_string(index=False))
print()

# Find best performing combinations
print("=== TOP 10 BEST PERFORMING MODELS ===")
top_models = results_df.nlargest(10, 'Accuracy')[['Dataset', 'Classifier', 'Accuracy', 'F1-Score']]
print(top_models.to_string(index=False))
print()

# Best model detailed analysis
best_model = results_df.loc[results_df['Accuracy'].idxmax()]
print("=== BEST MODEL DETAILED ANALYSIS ===")
print(f"Best Model: {best_model['Classifier']} on {best_model['Dataset']} dataset")
print(f"Accuracy: {best_model['Accuracy']:.4f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall: {best_model['Recall']:.4f}")
print(f"F1-Score: {best_model['F1-Score']:.4f}")