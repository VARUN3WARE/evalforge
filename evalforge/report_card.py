def generate_report_card(health_score_data, mismatch_data=None, drift_data=None, fragility_data=None, fairness_data=None, stability_data=None):
    """
    Generates a structured narrative report because nobody wants to read 
    raw JSON when the model goes off the rails :)

    Args:
        health_score_data (dict): The output directly from `compute_health_score`.
        mismatch_data (dict, optional): The output from `detect_confidence_accuracy_mismatch`.
        drift_data (dict, optional): The output from `detect_drift`.
        fragility_data (dict, optional): The output from `calculate_adversarial_fragility`.
        fairness_data (dict, optional): The output from `evaluate_fairness`.
        stability_data (dict, optional): The output from `compute_stability_from_scores` (model's emotional state).

    Returns:
        str: A nicely formatted markdown report card.
    """
    if "health_score" not in health_score_data:
        raise ValueError("Invalid health_score_data provided.")

    final_score = health_score_data["health_score"]
    components = health_score_data.get("component_scores", {})
    
    # Determine the vibe of the summary
    if final_score >= 90:
        risk_level = "Excellent"
        base_summary = "Your model performs exceptionally well across all evaluated dimensions."
    elif final_score >= 80:
        risk_level = "Good"
        base_summary = "Your model performs strongly but exhibits some areas for improvement."
    elif final_score >= 60:
        risk_level = "Moderate Risk"
        base_summary = "Your model has acceptable base metrics but shows notable vulnerabilities."
    else:
        risk_level = "High Risk"
        base_summary = "Your model is highly unreliable under scrutiny and requires immediate attention."

    # Build the flags narrative
    flags = []
    
    # 1. Base Accuracy Flag
    acc = components.get("accuracy", 0.0)
    if acc < 70:
        flags.append(f"Base accuracy is critically low ({acc:.1f}%).")
        
    # 2. Calibration / Mismatch Flag
    if mismatch_data:
        count = mismatch_data.get("mismatch_count", 0)
        rate = mismatch_data.get("mismatch_rate", 0.0)
        if count > 0:
            flags.append(f"Confidence calibration requires improvement due to {count} high-confidence errors ({(rate*100):.1f}% of total).")
    
    # 3. Fragility Flag
    if fragility_data:
        drop = fragility_data.get("average_drop", 0.0)
        frag_score = components.get("fragility", 100.0) # Check component score to see if it was evaluated
        if frag_score < 75:
            flags.append(f"Model exhibits severe fragility under perturbation (Average drop of {(drop*100):.1f}%).")
        elif frag_score < 90:
            flags.append(f"Model exhibits moderate fragility under perturbation (Average drop of {(drop*100):.1f}%).")

    # 4. Drift Flag
    drifted_count = 0
    if drift_data:
        drifted_count = sum(1 for feat in drift_data.values() if feat.get("drift_detected", False))
        if drifted_count > 0:
            flags.append(f"Distribution drift detected in {drifted_count} features.")

    # 5. Fairness Flag
    if fairness_data and fairness_data.get("bias_detected", False):
        penalty = fairness_data.get("bias_penalty", 0.0)
        flags.append(f"CRITICAL: Discriminatory bias detected across demographic groups (-{penalty:.1f} penalty).")
        
    # 6. Stability Flag (for models that can't make up their mind)
    if stability_data:
        stability_score = components.get("stability", 100.0)
        if stability_score < 70:
            flags.append(f"Model stability is low ({stability_score:.1f}%), indicating inconsistent performance across runs.")
        elif stability_score < 90:
            flags.append(f"Model stability is moderate ({stability_score:.1f}%), suggesting some variance in performance.")
    else:
        # We don't have stability data, so the model's consistency remains a mystery (or a secret talent).
        pass

    # Build Recommendations
    recommendations = []
    if acc < 80:
        recommendations.append("- Enhance feature engineering or collect more representative base training data.")
    if mismatch_data and mismatch_data.get("mismatch_count", 0) > 0:
        recommendations.append("- Apply temperature scaling, Platt scaling, or isotonic regression to fix calibration.")
    if fragility_data and components.get("fragility", 100.0) < 90:
        recommendations.append("- Train with data augmentation or adversarial examples to improve robustness.")
    if drifted_count > 0:
        recommendations.append("- Retrain the model on recent data and investigate upstream data pipelines.")
    if fairness_data and fairness_data.get("bias_detected", False):
        recommendations.append("- Audit training data for historical bias and implement fairness constraints during training.")
    if stability_data and components.get("stability", 100.0) < 90:
        recommendations.append("- Investigate training procedure for non-determinism, fix random seeds, or improve model architecture.")
        
    if not recommendations:
        recommendations.append("- Ship it. It's beautiful.")

    # Assemble the final string
    report = f"### Model Health Score: {final_score:.1f} ({risk_level})\n\n"
    
    # Summary paragraph
    report += f"> {base_summary}"
    if flags:
        report += " " + " ".join(flags)
    report += "\n\n"
    
    # Diagnostics Printout
    report += "#### Component Breakdown:\n"
    report += f"- Base Accuracy: {acc / 100.0:.2f}\n"
    if fragility_data:
        report += f"- Fragility Drop: {(fragility_data.get('average_drop', 0.0) * 100):.1f}%\n"
    if drift_data:
        report += f"- Drift Detected: {drifted_count} features\n"
    if mismatch_data:
        report += f"- High-Confidence Errors: {mismatch_data.get('mismatch_count', 0)}\n"
    if stability_data:
        report += f"- Stability Score: {stability_data.get('stability_score', 0.0):.1f}%\n" # The model's inner peace level
        
    report += "\n#### Recommendations:\n"
    report += "\n".join(recommendations)
    
    return report

