import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    """
    Creates the perfect illusion to show why accuracy is a lie :)
    
    1. Generates a 'clean' training dataset.
    2. Trains a RandomForest that gets >90% accuracy.
    3. Generates a 'drifted' testing dataset where one feature went rogue.
    4. Saves everything to disk for EvalForge CLI to rip apart.
    """
    print("🧪 Booting up the Illusion...")
    np.random.seed(42)

    # 1. Create Clean Synthetic Data
    X, y = make_classification(
        n_samples=2000, 
        n_features=10, 
        n_informative=5, 
        n_redundant=2, 
        random_state=42
    )

    feature_names = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # We use 1000 for training, 1000 for testing
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)
    
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    # 2. Train an Overconfident Model
    print("🧠 Training the Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Baseline check on clean test data
    X_test_clean = df_test.drop(columns=["target"])
    y_test_clean = df_test["target"]
    clean_acc = accuracy_score(y_test_clean, model.predict(X_test_clean))
    print(f"✅ Baseline Clean Accuracy: {clean_acc * 100:.1f}%")

    # 4. Introduce Chaos (Drift & Noise) to the Test Set
    print("😈 Introducing silent drift to the real-world data...")
    df_test_corrupted = df_test.copy()
    
    # Feature 3 drifts significantly (mean shift)
    df_test_corrupted["feature_3"] = df_test_corrupted["feature_3"] + np.random.normal(2.5, 0.5, size=len(df_test))
    
    # Let's save the datasets
    df_train.to_csv("demo_train.csv", index=False)
    df_test_corrupted.to_csv("demo_test.csv", index=False)
    
    # Let's save the model
    with open("demo_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    X_test_corrupt = df_test_corrupted.drop(columns=["target"])
    corrupt_acc = accuracy_score(y_test_clean, model.predict(X_test_corrupt))
    print(f"⚠️ Apparent Accuracy on Corrupted Data: {corrupt_acc * 100:.1f}%")
    print("\nAccuracy says ship it.")
    print("\nLet's see what EvalForge says:")
    print("Run: evalforge analyze --model demo_model.pkl --data demo_test.csv --target target --train-data demo_train.csv --visualize")

if __name__ == "__main__":
    main()
