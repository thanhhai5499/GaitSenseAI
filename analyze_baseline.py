#!/usr/bin/env python3
"""
Script phân tích toàn bộ dữ liệu để tạo quy trình chuẩn hóa báo cáo cho bác sĩ
"""

import sys
import os
import json
import re
from pathlib import Path
import numpy as np

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.gait_analyzer import GaitAnalyzer

def analyze_all_files():
    """Phân tích tất cả file dữ liệu để tìm baseline và pattern chuẩn"""
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("❌ Thư mục results không tồn tại")
        return
    
    # Lấy tất cả file .txt
    files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
    print(f"🔍 Tìm thấy {len(files)} file dữ liệu:")
    for f in files:
        print(f"  - {f}")
    
    # Khởi tạo analyzer
    gait_analyzer = GaitAnalyzer()
    
    all_results = []
    
    for file_name in files:
        file_path = os.path.join(results_dir, file_name)
        print(f"\n📄 Đang phân tích: {file_name}")
        
        try:
            # Đọc file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse JSON data
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if not json_match:
                print(f"❌ Không tìm thấy JSON data trong {file_name}")
                continue
                
            data = json.loads(json_match.group(1))
            patient_info = data.get('patient_info', {})
            
            # Phân tích
            diagnosis = gait_analyzer._generate_comprehensive_diagnosis(data)
            
            # Lưu kết quả
            result = {
                'file': file_name,
                'patient_name': patient_info.get('name', 'Unknown'),
                'patient_age': patient_info.get('age', 0),
                'patient_gender': patient_info.get('gender', 'Unknown'),
                'overall_score': diagnosis.get('overall_score', 0),
                'measurements': {},
                'assessments': {}
            }
            
            # Extract measurements từ diagnosis
            individual_assessments = diagnosis.get('individual_assessments', {})
            for param_vn, assessment in individual_assessments.items():
                result['assessments'][param_vn] = {
                    'status': assessment.get('status', 'unknown'),
                    'deviation_index': assessment.get('deviation_index', 0),
                    'position_percentage': assessment.get('position_percentage', 0)
                }
            
            # Extract measured values
            detailed_findings = diagnosis.get('detailed_findings', {})
            for param_vn, finding in detailed_findings.items():
                if 'measured_value' in finding:
                    result['measurements'][param_vn] = finding['measured_value']
            
            all_results.append(result)
            
            # In kết quả ngay
            print(f"  👤 {result['patient_name']} ({result['patient_age']} tuổi)")
            print(f"  📊 Điểm tổng: {result['overall_score']:.2f}/3.0")
            
            for param, status_info in result['assessments'].items():
                status = status_info['status']
                deviation = status_info['deviation_index']
                print(f"    • {param}: {status} (lệch: {deviation:.2f})")
                
        except Exception as e:
            print(f"❌ Lỗi khi phân tích {file_name}: {e}")
            continue
    
    print(f"\n" + "="*60)
    print(f"📊 TỔNG KẾT PHÂN TÍCH {len(all_results)} NGƯỜI")
    print(f"="*60)
    
    # Thống kê tổng quan
    scores = [r['overall_score'] for r in all_results]
    ages = [r['patient_age'] for r in all_results if r['patient_age'] > 0]
    
    print(f"🎯 Điểm số trung bình: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"📈 Điểm cao nhất: {max(scores):.2f}")
    print(f"📉 Điểm thấp nhất: {min(scores):.2f}")
    print(f"👥 Độ tuổi trung bình: {np.mean(ages):.1f} ± {np.std(ages):.1f}")
    
    # Phân loại theo mức độ
    normal_count = sum(1 for s in scores if s < 0.5)
    mild_count = sum(1 for s in scores if 0.5 <= s < 1.5)
    moderate_count = sum(1 for s in scores if 1.5 <= s < 2.5)
    severe_count = sum(1 for s in scores if s >= 2.5)
    
    print(f"\n📋 PHÂN LOẠI TÌNH TRẠNG:")
    print(f"  • BÌNH THƯỜNG (< 0.5): {normal_count} người ({normal_count/len(all_results)*100:.1f}%)")
    print(f"  • CẦN CHÚ Ý (0.5-1.5): {mild_count} người ({mild_count/len(all_results)*100:.1f}%)")
    print(f"  • CẦN ĐIỀU TRỊ (1.5-2.5): {moderate_count} người ({moderate_count/len(all_results)*100:.1f}%)")
    print(f"  • NGHIÊM TRỌNG (≥ 2.5): {severe_count} người ({severe_count/len(all_results)*100:.1f}%)")
    
    # Phân tích từng thông số
    print(f"\n📏 PHÂN TÍCH CHI TIẾT TỪNG THÔNG SỐ:")
    
    # Tập hợp tất cả measurements
    all_measurements = {}
    for result in all_results:
        for param, value in result['measurements'].items():
            if param not in all_measurements:
                all_measurements[param] = []
            all_measurements[param].append(value)
    
    for param, values in all_measurements.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = min(values)
            max_val = max(values)
            print(f"  • {param}: {mean_val:.2f} ± {std_val:.2f} (min: {min_val:.2f}, max: {max_val:.2f})")
    
    # Tính tỷ lệ bất thường cho từng thông số
    print(f"\n⚠️ TỶ LỆ BẤT THƯỜNG THEO THÔNG SỐ:")
    
    param_abnormal = {}
    for result in all_results:
        for param, assessment in result['assessments'].items():
            if param not in param_abnormal:
                param_abnormal[param] = {'normal': 0, 'abnormal': 0}
            
            if assessment['status'] in ['normal']:
                param_abnormal[param]['normal'] += 1
            else:
                param_abnormal[param]['abnormal'] += 1
    
    for param, counts in param_abnormal.items():
        total = counts['normal'] + counts['abnormal']
        abnormal_rate = counts['abnormal'] / total * 100 if total > 0 else 0
        print(f"  • {param}: {abnormal_rate:.1f}% bất thường ({counts['abnormal']}/{total})")
    
    # Recommendations cho bác sĩ
    print(f"\n" + "="*60)
    print(f"💡 KHUYẾN NGHỊ QUY TRÌNH CHUẨN CHO BÁC SĨ")
    print(f"="*60)
    
    print(f"""
🎯 NGƯỠNG ĐÁNH GIÁ ĐƯỢC ĐỀ XUẤT:
  • BÌNH THƯỜNG: Điểm < 0.5 (Không cần can thiệp)
  • CẦN CHÚ Ý: Điểm 0.5-1.5 (Theo dõi định kỳ)
  • CẦN ĐIỀU TRỊ: Điểm 1.5-2.5 (Can thiệp tích cực)
  • NGHIÊM TRỌNG: Điểm ≥ 2.5 (Cần điều trị ngay)

📋 CÁC THÔNG SỐ QUAN TRỌNG NHẤT:
  1. Chiều Cao Nâng Chân (foot clearance)
  2. Chiều Dài Bước (stride length)  
  3. Bất Cân Xứng các khớp
  4. Tốc độ đi bộ

⚡ QUY TRÌNH ĐÁNH GIÁ ĐỀ XUẤT:
  1. Kiểm tra điểm tổng trước
  2. Phân tích từng thông số chi tiết
  3. Đánh giá mức độ bất cân xứng
  4. Đưa ra khuyến nghị điều trị cụ thể
  5. Lên lịch tái khám phù hợp
""")
    
    return all_results

if __name__ == "__main__":
    analyze_all_files()
