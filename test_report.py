#!/usr/bin/env python3
"""Test script to generate a new HTML report with the fixed data"""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from main import MainWindow
from src.core.gait_analyzer import GaitAnalyzer

def test_new_report():
    """Generate a new report with the latest data"""
    try:
        # Create main window instance
        import sys
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        main_window = MainWindow()
        
        # Create gait analyzer
        gait_analyzer = GaitAnalyzer()
        
        # Use the test data file
        data_file = "results/Long_30tuoi_Nam_20250829_140016.txt"
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return
            
        print(f"🔍 Generating report from: {data_file}")
        
        # Load data and generate diagnosis
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the JSON data from the file
        import re
        json_match = re.search(r'({.*})', content, re.DOTALL)
        if not json_match:
            print("❌ Could not find JSON data in file")
            return
            
        data = json.loads(json_match.group(1))
        
        # Generate diagnosis using the gait analyzer  
        diagnosis = gait_analyzer._generate_comprehensive_diagnosis(data)
        
        # Generate HTML report
        html_content = main_window.generate_diagnosis_report(diagnosis, data_file)
        
        # Save the report
        output_file = "test_report_fixed.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ Report generated successfully: {output_file}")
        print(f"📄 You can now open {output_file} in your browser to check the UI formatting")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_report()
