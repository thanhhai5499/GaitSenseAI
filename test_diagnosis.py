import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from core.gait_analyzer import GaitAnalyzer

print('🔍 Testing diagnosis with actual data file...')
result = GaitAnalyzer.load_and_diagnose('results/Long_30tuoi_Nam_20250829_135812.txt')

if result:
    print('\n=== DIAGNOSIS RESULT ===')
    print(f'Patient: {result.get("patient_name")} - {result.get("patient_age")} years old')
    print(f'Severity Score: {result.get("severity_score")}/3')
    print(f'Assessment: {result.get("assessment_summary", "No assessment")}')
    
    print('\n=== DETAILED FINDINGS ===')
    findings = result.get('detailed_findings', {})
    for key, data in findings.items():
        if isinstance(data, dict) and 'measured_value' in data:
            print(f'{key}: {data.get("measured_value", 0):.2f} {data.get("unit", "")} ({data.get("status", "UNKNOWN")})')
        elif isinstance(data, dict) and 'asymmetry_percent' in data:
            print(f'{key}: {data.get("asymmetry_percent", 0):.1f}% asymmetry ({data.get("status", "UNKNOWN")})')
    
    print('\n=== RAW DATA SUMMARY ===')
    print(f'Available findings keys: {list(findings.keys())}')
else:
    print('❌ Failed to load diagnosis')
