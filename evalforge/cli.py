import argparse
import pickle
import pandas as pd
from evalforge.metrics import calculate_metrics
from evalforge.bootstrap import compute_bootstrap_ci
from evalforge.drift import detect_drift
from evalforge.mismatch import detect_confidence_accuracy_mismatch
from evalforge.fragility import calculate_adversarial_fragility
from evalforge.health_score import compute_health_score
from evalforge.report_card import generate_report_card
from evalforge.visualize import plot_fragility_drop, plot_drift_histogram

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    """
    Main entry point for EvalForge CLI.
    If it breaks here, we blame the user's dataset :)
    """
    parser = argparse.ArgumentParser(description="EvalForge: Model Health & Evaluation Intelligence Engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # "analyze" command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a model against a dataset")
    analyze_parser.add_argument("--model", type=str, required=True, help="Path to the model .pkl file")
    analyze_parser.add_argument("--data", type=str, required=True, help="Path to the testing dataset .csv")
    analyze_parser.add_argument("--target", type=str, required=True, help="Target column name")
    analyze_parser.add_argument("--train-data", type=str, required=False, help="Optional: Path to training dataset .csv for drift detection")
    analyze_parser.add_argument("--visualize", action="store_true", help="Generate and save PNG plots to reports/ directory")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        print("🚀 Booting up EvalForge Analysis...")
        
        # 1. Load Model
        try:
            model = load_pkl(args.model)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
            
        # 2. Load Data
        try:
            df_test = pd.read_csv(args.data)
            if args.target not in df_test.columns:
                print(f"❌ Target column '{args.target}' not found in {args.data}")
                return
                
            y_true = df_test[args.target]
            X_test = df_test.drop(columns=[args.target])
        except Exception as e:
            print(f"❌ Failed to load testing dataset: {e}")
            return
            
        # 3. Predict
        try:
            print("⏳ Running predictions...")
            y_pred = model.predict(X_test)
            # Try to get probabilities for AUC/Mismatch
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            return
            
        # 4. Compute Metrics
        print("📊 Computing Base Metrics...")
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        accuracy = metrics.get("accuracy", 0.0)
        
        # 5. Diagnostics
        print("🕵️  Detecting Mismatches...")
        mismatch_data = None
        if y_prob is not None:
             mismatch_data = detect_confidence_accuracy_mismatch(y_true, y_pred, y_prob)
             
        print("💥 Testing Fragility...")
        fragility_data = calculate_adversarial_fragility(model, X_test, y_true)
        
        print("🌊 Checking for Drift...")
        drift_data = None
        if args.train_data:
            try:
                df_train = pd.read_csv(args.train_data)
                X_train = df_train.drop(columns=[args.target], errors='ignore')
                drift_data = detect_drift(X_train, X_test)
            except Exception as e:
                print(f"⚠️ Failed to load training data for drift: {e}")
                
        # 6. Compute Health Score
        print("❤️  Calculating Health Score...")
        mismatch_rate = mismatch_data["mismatch_rate"] if mismatch_data else None
        fragility_score = fragility_data["fragility_score"] if fragility_data else None
        
        health_score_data = compute_health_score(
            accuracy_metric=accuracy,
            mismatch_rate=mismatch_rate,
            fragility_score=fragility_score,
            drift_report=drift_data
        )
        
        # 7. Generate Reporting Layer
        print("✅ Analysis Complete. Generating Report...\n")
        report = generate_report_card(health_score_data, mismatch_data, drift_data, fragility_data)
        
        # 8. Visulization dump
        if args.visualize:
            print("🎨 Generating Visualizations...")
            if fragility_data and "drops" in fragility_data:
                plot_fragility_drop(fragility_data.get("baseline_score", accuracy), fragility_data["drops"])
                print("   Saved -> reports/fragility_drop.png")
            
            if drift_data:
                for feature, metrics in drift_data.items():
                    if metrics.get("drift_detected", False):
                        try: # Catch in case df_train/test scope is lost
                           plot_drift_histogram(pd.read_csv(args.train_data), X_test, feature)
                           print(f"   Saved -> reports/drift_{feature}.png")
                        except:
                           pass
        
        # Output Results
        print("==================================================")
        print(report)
        print("==================================================")

if __name__ == "__main__":
    main()
