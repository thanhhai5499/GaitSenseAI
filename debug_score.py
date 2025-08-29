#!/usr/bin/env python3
"""
Debug script để kiểm tra chi tiết quá trình tính overall_score
"""

import sys
import os
import json
import re
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.gait_analyzer import GaitAnalyzer
from data.normative_gait_data import NormativeGaitData

def debug_score_calculation(file_name):
    """Debug chi tiết quá trình tính điểm"""
    
    file_path = f"results/{file_name}"
    print(f"🔍 Debug tính điểm cho: {file_name}")
    
    # Đọc file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse JSON data
    json_match = re.search(r'({.*})', content, re.DOTALL)
    if not json_match:
        print(f"❌ Không tìm thấy JSON data")
        return
        
    data = json.loads(json_match.group(1))
    patient_info = data.get('patient_info', {})
    
    # Tạo gait analyzer và get measured data
    gait_analyzer = GaitAnalyzer()
    diagnosis = gait_analyzer._generate_comprehensive_diagnosis(data)
    
    # Get the measured data that was used
    print("\n📊 MEASURED DATA:")
    detailed_findings = diagnosis.get('detailed_findings', {})
    measured_data = {}
    for param_vn, finding in detailed_findings.items():
        if 'measured_value' in finding:
            measured_data[param_vn] = finding['measured_value']
            print(f"  {param_vn}: {finding['measured_value']}")
    
    # Manually calculate overall score step by step
    print("\n🔍 CHI TIẾT TÍNH ĐIỂM:")
    
    normative_db = NormativeGaitData()
    
    # Map Vietnamese to English parameter names
    param_mapping = {
        'Chiều Dài Bước': 'stride_length',
        'Tốc Độ Đi': 'walking_speed', 
        'Thời Gian Đặt Chân': 'stance_phase_percentage',
        'Chiều Cao Nâng Chân': 'foot_clearance',
        'Chiều Rộng Bước': 'step_width'
    }
    
    # Prepare measured_data_en for assessment
    measured_data_en = {}
    for param_vn, value in measured_data.items():
        if param_vn in param_mapping:
            param_en = param_mapping[param_vn]
            measured_data_en[param_en] = value
            print(f"  {param_vn} -> {param_en} = {value}")
    
    # Add joint angles if available
    individual_assessments = diagnosis.get('individual_assessments', {})
    asymmetry_assessments = diagnosis.get('asymmetry_assessments', {})
    
    print(f"\n📈 INDIVIDUAL ASSESSMENTS:")
    total_severity = 0
    count = 0
    severity_map = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    
    for param_vn, assessment in individual_assessments.items():
        status = assessment.get('status', 'unknown')
        severity_score = severity_map.get(status, 0)
        total_severity += severity_score
        count += 1
        print(f"  {param_vn}: {status} -> {severity_score} điểm")
    
    print(f"\n🎯 ASYMMETRY ASSESSMENTS:")
    for joint, assessment in asymmetry_assessments.items():
        status = assessment.get('status', 'unknown')
        severity_score = severity_map.get(status, 0)
        total_severity += severity_score
        count += 1
        print(f"  {joint}: {status} -> {severity_score} điểm")
    
    # Calculate final score
    if count > 0:
        final_overall_score = round(total_severity / count, 1)
    else:
        final_overall_score = 0
    
    print(f"\n🏆 KẾT QUẢ CUỐI CÙNG:")
    print(f"  Tổng điểm nghiêm trọng: {total_severity}")
    print(f"  Số thông số đánh giá: {count}")
    print(f"  Overall Score: {total_severity}/{count} = {final_overall_score}")
    
    # Compare with diagnosis result
    diagnosis_score = diagnosis.get('overall_score', 0)
    print(f"  Điểm từ diagnosis: {diagnosis_score}")
    
    if abs(final_overall_score - diagnosis_score) > 0.01:
        print(f"  ⚠️ KHÁC BIỆT: Expected {final_overall_score}, got {diagnosis_score}")
    else:
        print(f"  ✅ Khớp!")

if __name__ == "__main__":
    # Test với file có dáng đi bình thường (stride=100cm)
    print("="*60)
    print("TEST 1: File có dáng đi BÌNH THƯỜNG")
    debug_score_calculation("Hải_25tuoi_Nam_20250829_082245.txt")
    
    print("\n" + "="*60)
    print("TEST 2: File có dáng đi CÓ VẤN ĐỀ")
    debug_score_calculation("Hải_25tuoi_Nam_20250829_135459.txt")
