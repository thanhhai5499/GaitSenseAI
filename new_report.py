def generate_new_diagnosis_report(self, diagnosis, data_file):
    """Generate new HTML diagnosis report focusing on asymmetry analysis"""
    
    # Lấy dữ liệu chi tiết từ diagnosis
    findings = diagnosis.get('detailed_findings', {})
    
    # Tính toán độ lệch so với chuẩn
    def calculate_deviation_percentage(measured, norm_mean, norm_std):
        if norm_std == 0:
            return 0
        deviation = abs(measured - norm_mean) / norm_std
        return min(deviation * 100, 999)  # Cap at 999%
    
    def get_status_color(status):
        colors = {
            'BÌNH THƯỜNG': '#28a745',
            'NHẸ': '#ffc107', 
            'TRUNG BÌNH': '#fd7e14',
            'NẶNG': '#dc3545'
        }
        return colors.get(status, '#6c757d')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Báo Cáo Phân Tích Dáng Đi</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                line-height: 1.6; 
                font-size: 16px; 
                margin: 0; 
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
            }}
            .header {{ 
                background: linear-gradient(135deg, #0078d4 0%, #106ebe 100%); 
                color: white;
                padding: 30px; 
                border-radius: 15px; 
                margin-bottom: 30px; 
                box-shadow: 0 4px 15px rgba(0,120,212,0.3);
                text-align: center;
            }}
            .section {{ 
                margin: 20px 0; 
                padding: 25px; 
                border-radius: 12px; 
                background-color: #ffffff; 
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                border: 1px solid #e9ecef;
            }}
            .asymmetry-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .joint-card {{
                background: #ffffff;
                border-radius: 12px;
                padding: 20px;
                border: 2px solid #e9ecef;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                position: relative;
            }}
            .joint-title {{
                font-size: 18px;
                font-weight: bold;
                color: #0078d4;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }}
            .angle-comparison {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 8px 0;
                padding: 12px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }}
            .deviation-visual {{
                width: 100%;
                height: 30px;
                background: #e9ecef;
                border-radius: 15px;
                position: relative;
                margin: 10px 0;
                overflow: hidden;
            }}
            .deviation-fill {{
                height: 100%;
                border-radius: 15px;
                transition: all 0.3s ease;
                position: relative;
            }}
            .status-normal {{ background: linear-gradient(90deg, #28a745, #20c997); }}
            .status-mild {{ background: linear-gradient(90deg, #ffc107, #ffca2c); }}
            .status-moderate {{ background: linear-gradient(90deg, #fd7e14, #ff922b); }}
            .status-severe {{ background: linear-gradient(90deg, #dc3545, #e55353); }}
            .section-title {{ 
                color: #0078d4; 
                font-weight: bold; 
                font-size: 20px; 
                margin-bottom: 20px; 
                padding-bottom: 10px;
                border-bottom: 3px solid #0078d4;
                display: flex;
                align-items: center;
            }}
            .metric-value {{
                font-size: 20px;
                font-weight: bold;
                color: #0078d4;
            }}
            .comparison-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                overflow: hidden;
            }}
            .comparison-table th, .comparison-table td {{
                padding: 12px;
                text-align: center;
                border-bottom: 1px solid #dee2e6;
            }}
            .comparison-table th {{
                background: #f8f9fa;
                font-weight: bold;
                color: #495057;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                border-left: 5px solid #2196f3;
            }}
            .overall-status {{
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            }}
            h2 {{ font-size: 24px; margin: 15px 0; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }}
            h3 {{ font-size: 18px; margin: 12px 0; color: #495057; }}
            strong {{ font-weight: bold; color: #212529; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🏥 BÁO CÁO PHÂN TÍCH DÁNG ĐI CHI TIẾT</h2>
            <div style="margin-top: 20px; font-size: 18px;">
                <p><strong>👤 Bệnh nhân:</strong> {diagnosis.get('patient_name', 'N/A')} - {diagnosis.get('patient_age', 'N/A')} tuổi ({diagnosis.get('patient_gender', 'N/A')})</p>
                <p><strong>📅 Thời gian:</strong> {diagnosis.get('session_date', 'N/A')}</p>
            </div>
        </div>
    """
    
    # Tóm tắt tổng quan
    overall_score = diagnosis.get('severity_score', 0)
    status_classes = {0: 'normal', 1: 'mild', 2: 'moderate', 3: 'severe'}
    status_texts = {0: 'BÌNH THƯỜNG', 1: 'CẦN CHÚ Ý', 2: 'CẦN ĐIỀU TRỊ', 3: 'NGHIÊM TRỌNG'}
    status_colors = {0: '#28a745', 1: '#ffc107', 2: '#fd7e14', 3: '#dc3545'}
    
    html += f"""
        <div class="section">
            <div class="section-title">📊 TỔNG QUAN TÌNH TRẠNG</div>
            <div class="overall-status" style="background: {status_colors.get(overall_score, '#6c757d')}; color: white;">
                {status_texts.get(overall_score, 'KHÔNG XÁC ĐỊNH')} - Điểm số: {overall_score}/3
            </div>
            <div class="summary-card">
                <h3>📋 Đánh giá tổng thể:</h3>
                <p>{diagnosis.get('assessment_summary', 'Không có đánh giá')}</p>
                <p><strong>Cần theo dõi:</strong> {'✅ Có' if diagnosis.get('follow_up_needed', False) else '❌ Không'}</p>
            </div>
        </div>
    """
    
    # Phân tích bất đối xứng các khớp
    html += f"""
        <div class="section">
            <div class="section-title">⚖️ PHÂN TÍCH BẤT ĐỐI XỨNG CÁC KHỚP</div>
            <div class="asymmetry-grid">
    """
    
    # Các khớp cần phân tích
    joints = [
        ('Bất Cân Xứng Gối', '🦵', 'Khớp gối giữa chân trái và chân phải'),
        ('Bất Cân Xứng Hông', '🦴', 'Khớp hông giữa chân trái và chân phải'), 
        ('Bất Cân Xứng Cổ Chân', '🦶', 'Khớp cổ chân giữa chân trái và chân phải')
    ]
    
    for joint_name, emoji, description in joints:
        if joint_name in findings:
            data = findings[joint_name]
            status = data.get('status', 'KHÔNG RÕ')
            color = get_status_color(status)
            asymmetry_percent = data.get('asymmetry_percent', 0)
            recommendation = data.get('recommendation', 'Không có khuyến nghị')
            
            # Tính độ rộng của thanh deviation (max 100%)
            bar_width = min(asymmetry_percent * 10, 100)  # Scale for visualization
            
            html += f"""
                <div class="joint-card" style="border-left: 5px solid {color};">
                    <div class="joint-title">{emoji} {joint_name}</div>
                    <p style="color: #666; font-style: italic; margin-bottom: 15px;">{description}</p>
                    
                    <div class="angle-comparison">
                        <span><strong>Mức bất cân xứng:</strong></span>
                        <span class="metric-value" style="color: {color};">{asymmetry_percent:.1f}%</span>
                    </div>
                    
                    <div class="deviation-visual">
                        <div class="deviation-fill status-{status_classes.get(overall_score, 'unknown')}" 
                             style="width: {bar_width}%; background: {color};">
                        </div>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-top: 10px;">
                        <p style="margin: 0; font-weight: bold; color: {color};">📊 Đánh giá: {status}</p>
                        <p style="margin: 5px 0 0 0; font-size: 14px;"><strong>Giải thích:</strong> 
                            {'Cân xứng tốt' if asymmetry_percent <= 3 else 
                             'Hơi lệch' if asymmetry_percent <= 6 else
                             'Lệch rõ rệt' if asymmetry_percent <= 10 else 'Lệch nghiêm trọng'}
                        </p>
                    </div>
                    
                    <div style="background: #e9ecef; padding: 10px; border-radius: 6px; margin-top: 10px;">
                        <strong>💡 Khuyến nghị:</strong> {recommendation}
                    </div>
                </div>
            """
    
    html += "</div></div>"
    
    # Thông số chung
    html += f"""
        <div class="section">
            <div class="section-title">📏 THÔNG SỐ CHUNG</div>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Thông số</th>
                        <th>Giá trị đo được</th>
                        <th>Giá trị chuẩn</th>
                        <th>Độ lệch</th>
                        <th>Đánh giá</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Các thông số chung cần hiển thị
    general_params = [
        ('Tốc Độ Đi', 'm/s'),
        ('Chiều Dài Bước', 'cm'), 
        ('Thời Gian Đặt Chân', '%'),
        ('Chiều Cao Nâng Chân', 'cm'),
        ('Chiều Rộng Bước', 'cm')
    ]
    
    for param_name, unit in general_params:
        if param_name in findings:
            data = findings[param_name]
            measured = data.get('measured_value', 0)
            norm_mean = data.get('normative_mean', 0)
            norm_std = data.get('normative_std', 0)
            status = data.get('status', 'KHÔNG RÕ')
            color = get_status_color(status)
            
            # Tính độ lệch %
            if norm_mean > 0:
                deviation_percent = abs(measured - norm_mean) / norm_mean * 100
            else:
                deviation_percent = 0
                
            html += f"""
                <tr>
                    <td><strong>{param_name}</strong></td>
                    <td style="color: #0078d4; font-weight: bold;">{measured:.1f} {unit}</td>
                    <td>{norm_mean:.1f} ± {norm_std:.1f} {unit}</td>
                    <td style="color: {color}; font-weight: bold;">{deviation_percent:.1f}%</td>
                    <td style="color: {color}; font-weight: bold;">{status}</td>
                </tr>
            """
    
    html += """
                </tbody>
            </table>
        </div>
    """
    
    # Khuyến nghị
    recommendations = diagnosis.get('recommendations', [])
    if recommendations:
        html += """
            <div class="section">
                <div class="section-title">💊 KHUYẾN NGHỊ ĐIỀU TRỊ</div>
                <ul style="list-style-type: none; padding: 0;">
        """
        for i, rec in enumerate(recommendations, 1):
            html += f"<li style='margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #0078d4; border-radius: 4px;'><strong>{i}.</strong> {rec}</li>"
        html += "</ul></div>"
    
    # Lưu ý quan trọng
    html += f"""
        <div class="section" style="background: #fff3cd; border: 2px solid #ffc107;">
            <div class="section-title" style="color: #856404; border-bottom-color: #856404;">⚠️ LƯU Ý QUAN TRỌNG</div>
            <div style="color: #856404;">
                <p><strong>📋 Kết quả này chỉ mang tính chất tham khảo</strong> và không thay thế cho việc khám bệnh chuyên khoa.</p>
                <p><strong>👨‍⚕️ Vui lòng tham khảo ý kiến bác sĩ</strong> chuyên khoa cơ xương khớp hoặc phục hồi chức năng để được tư vấn và điều trị phù hợp.</p>
                <p><strong>🔄 Nên thực hiện phân tích nhiều lần</strong> trong các điều kiện khác nhau để có kết quả chính xác nhất.</p>
                <p><strong>📄 File dữ liệu:</strong> {data_file}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html