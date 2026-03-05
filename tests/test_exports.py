import os
import shutil
from evalforge.exports import generate_html_report

def test_generate_html_report():
    """
    Test generating the HTML file and ensure it writes properly.
    """
    test_dir = "test_html_reports"
    output_path = os.path.join(test_dir, "report.html")
    
    os.makedirs(test_dir, exist_ok=True)
    
    fake_pngs = ["test1.png", "test2.png"]
    fake_text = "### Health Score: 50.0\\n\\nTerrible model."
    
    path = generate_html_report(fake_text, fake_pngs, output_path)
    
    assert os.path.exists(path)
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "EvalForge Diagnostic Report" in content
        assert "test1.png" in content
        assert "Terrible model" in content
        
    # Cleanup
    shutil.rmtree(test_dir)
