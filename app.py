import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="AI 벽체 철근 배근 간격 측정기", layout="wide")

st.title("🧱 AI 벽체 철근 배근 간격 자동 측정기")
st.markdown("""
**대각선 ArUco 마커($200\text{mm} \times 200\text{mm}$) 2개소** 기반으로 철근 간격을 정밀 산출하고, 
검측 사진대지를 자동으로 생성합니다.
""")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 검측 기준 및 마커 설정")
marker_size_mm = st.sidebar.number_input("마커 한 변 크기 (mm)", value=200, step=10)
target_spacing_mm = st.sidebar.selectbox("설계 배근 간격 (mm)", [150, 200, 250, 300], index=1)
tolerance_mm = st.sidebar.number_input("허용 오차 범위 (±mm)", value=15, step=1)
location_info = st.sidebar.text_input("부재 및 위치 정보", value="101동 3층 W1 벽체")

# --- ArUco 마커 검출 및 스케일/투시 보정 함수 ---
def process_rebar_image(image_bytes, marker_size, target_mm, tol_mm):
    # 이미지 로드
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=uint8) if hasattr(image_bytes, 'read') else image_bytes
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None, "이미지를 불러올 수 없습니다.", None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ArUco 검출기 설정 (OpenCV 버전 호환)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(corners) < 1:
        return None, "❌ ArUco 마커를 인식하지 못했습니다. 마커가 가려지지 않았는지 확인해 주세요.", None

    result_img = img.copy()
    scales = []

    # 마커 감지 시각화 및 스케일 산출
    for i, corner in enumerate(corners):
        pts = corner[0].astype(np.int32)
        cv2.polylines(result_img, [pts], True, (0, 255, 0), 4)
        
        # 마커의 한 변 픽셀 길이 평균 계산
        side1 = np.linalg.norm(pts[0] - pts[1])
        side2 = np.linalg.norm(pts[1] - pts[2])
        avg_side_px = (side1 + side2) / 2.0
        
        mm_per_px = marker_size / avg_side_px
        scales.append(mm_per_px)
        
        marker_id = ids[i][0] if ids is not None else i
        cv2.putText(result_img, f"Marker #{marker_id} ({int(marker_size)}mm)", 
                    (pts[0][0], pts[0][1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 2개 이상 마커 검출 시 평균 스케일 사용하여 오차 최소화
    avg_mm_per_px = np.mean(scales)

    # --- 철근 엣지 및 중심선 추적 ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    
    # 수직 철근 검출 (HoughLinesP)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=120, maxLineGap=25)
    
    x_coords = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 25: # 수직근 필터링
                x_avg = (x1 + x2) // 2
                x_coords.append(x_avg)
                cv2.line(result_img, (x_avg, 0), (x_avg, img.shape[0]), (255, 0, 255), 1)

    # 근접선 군집화 (클러스터링)
    x_coords = sorted(list(set(x_coords)))
    filtered_x = []
    min_gap_px = int(80 / avg_mm_per_px) # 최소 80mm 이상 떨어진 유효 선만 인정
    
    for x in x_coords:
        if not filtered_x or abs(x - filtered_x[-1]) > min_gap_px:
            filtered_x.append(x)

    # --- 배근 간격 측정 및 결과 오버레이 ---
    measurements = []
    pass_count = 0
    fail_count = 0

    for i in range(len(filtered_x) - 1):
        x1, x2 = filtered_x[i], filtered_x[i+1]
        px_dist = abs(x2 - x1)
        real_mm = round(px_dist * avg_mm_per_px)
        
        # 합격 / 불합격 판정
        is_pass = abs(real_mm - target_mm) <= tol_mm
        if is_pass:
            pass_count += 1
            color = (0, 255, 0) # 초록
            status = "PASS"
        else:
            fail_count += 1
            color = (0, 0, 255) # 빨강
            status = "FAIL"

        measurements.append({"index": i+1, "spacing_mm": real_mm, "status": status})

        # 사진상 수치 표시
        y_pos = int(img.shape[0] * 0.4) + (i % 3) * 50
        cv2.line(result_img, (x1, y_pos), (x2, y_pos), color, 3)
        cv2.circle(result_img, (x1, y_pos), 5, color, -1)
        cv2.circle(result_img, (x2, y_pos), 5, color, -1)
        
        label = f"{real_mm}mm [{status}]"
        cv2.putText(result_img, label, (x1 + (px_dist // 2) - 40, y_pos - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- 스마트 사진 보드(현황판) 자동 합성 ---
    board_h = 120
    board = np.zeros((board_h, img.shape[1], 3), dtype=np.uint8)
    info_text = f"Location: {location_info} | Design Spec: @{target_mm}mm (Tol: +/-{tol_mm}mm)"
    result_text = f"Detected Rebars: {len(filtered_x)} | PASS: {pass_count} | FAIL: {fail_count} | Scale: 1px={avg_mm_per_px:.2f}mm"
    
    cv2.putText(board, info_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(board, result_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    final_output = np.vstack([result_img, board])

    summary = {
        "marker_count": len(corners),
        "total_lines": len(filtered_x),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "measurements": measurements,
        "scale": avg_mm_per_px
    }

    return final_output, summary, None

# --- UI 레이아웃 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 현장 배근 사진 업로드 / 촬영")
    img_file = st.file_uploader("스마트폰으로 촬영한 철근 사진 선택", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("카메라로 직접 촬영")

    selected_file = camera_file if camera_file is not None else img_file

with col2:
    st.subheader("📊 AI 간격 분석 및 검측 결과")
    if selected_file is not None:
        with st.spinner("ArUco 마커 감지 및 간격 산출 중..."):
            res_img, summary, err = process_rebar_image(selected_file, marker_size_mm, target_spacing_mm, tolerance_mm)
            
            if err:
                st.error(err)
            else:
                # 결과 이미지 표시
                st.image(res_img, channels="BGR", use_column_width=True)
                
                # 측정 요약 수치
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("인식된 마커", f"{summary['marker_count']} 개소")
                m_col2.metric("합격 구간 (Pass)", f"{summary['pass_count']} 개")
                m_col3.metric("불합격 구간 (Fail)", f"{summary['fail_count']} 개", delta_color="inverse")

                # 측정 데이터 테이블
                if summary['measurements']:
                    st.dataframe(summary['measurements'])

                # 다운로드 버튼
                res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(res_rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                
                st.download_button(
                    label="📥 검측 사진대지(JPG) 다운로드",
                    data=buf.getvalue(),
                    file_name=f"Rebar_Inspection_{location_info}.jpg",
                    mime="image/jpeg"
                )
    else:
        st.info("좌측에서 사진을 업로드하거나 스마트폰 카메라로 촬영해 주세요.")