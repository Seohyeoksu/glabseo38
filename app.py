import json
import re
import time
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pypdf import PdfReader

import streamlit as st
import pandas as pd
from openai import OpenAI, APIConnectionError, RateLimitError


# ───────────── 1. 주요 알레르기 정보 (참고용) ─────────────
MAJOR_ALLERGENS = [
    "우유/유제품", "달걀/난류", "땅콩", "견과류", "밀/글루텐", 
    "대두/콩", "생선/어류", "갑각류(새우,게,랍스터)", "조개류/패류",
    "참깨", "아황산염", "메밀", "돼지고기", "소고기", "닭고기",
    "복숭아", "토마토", "키위", "바나나", "아보카도"
]


# ───────────── 2. AI 기반 이미지 분석 함수 ─────────────
def analyze_food_image_with_ai(
    image_bytes: bytes,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> Dict:
    """AI를 사용하여 이미지에서 음식과 알레르기 정보를 상세 분석"""
    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode()
    
    prompt = f"""
    당신은 학교 급식 영양사이자 알레르기 전문가입니다. 
    이 이미지를 매우 자세히 분석하여 다음 작업을 수행해주세요:

    1. 이미지에 있는 모든 음식/메뉴를 식별하세요
    2. 각 음식의 일반적인 재료와 조리법을 고려하세요
    3. 숨겨진 알레르기 유발 요소까지 찾아내세요
    4. 한국 학교 급식의 특성을 고려하세요

    주요 확인 알레르기: {', '.join(MAJOR_ALLERGENS)}

    다음 JSON 형식으로 매우 상세하게 응답해주세요:
    {{
        "menu_items": [
            {{
                "name": "음식명",
                "ingredients": ["주재료1", "주재료2", "부재료1", ...],
                "cooking_method": "조리 방법",
                "allergens": [
                    {{
                        "allergen": "알레르기 유발 물질",
                        "source": "어떤 재료에서 유래",
                        "risk_level": "고도/중등도/경도",
                        "hidden": true/false,
                        "cross_contamination": true/false
                    }}
                ],
                "nutrition_notes": "영양학적 특징"
            }}
        ],
        "overall_assessment": {{
            "total_allergens": ["전체 알레르기 목록"],
            "high_risk_items": ["고위험 항목들"],
            "hidden_allergens": ["숨겨진 알레르기 유발 요소"],
            "safety_notes": "전반적인 안전 주의사항"
        }},
        "recommendations": {{
            "substitutions": ["대체 가능한 메뉴"],
            "preparation_tips": ["조리시 주의사항"],
            "serving_guidelines": ["배식시 주의사항"]
        }}
    }}

    중요: 
    - 조미료, 소스, 양념에 숨겨진 알레르기 성분도 찾아주세요
    - 교차 오염 가능성도 평가해주세요
    - 한국 음식의 특성(고추장, 된장, 새우젓 등)을 고려하세요
    """
    
    for i in range(max_retries + 1):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                temperature=0.3,
                max_tokens=2000
            )
            break
        except (RateLimitError, APIConnectionError):
            if i == max_retries:
                return {"error": "API 호출 실패"}
            time.sleep(2 ** i)
    
    try:
        content = res.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "응답 파싱 실패", "raw": content}
    except json.JSONDecodeError:
        return {"error": "JSON 파싱 오류", "raw": content}


# ───────────── 3. AI 기반 텍스트 분석 함수 ─────────────
def analyze_text_with_ai(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> Dict:
    """AI를 사용하여 텍스트에서 알레르기 정보를 상세 분석"""
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    당신은 학교 급식 영양사이자 알레르기 전문가입니다.
    다음 텍스트(급식 메뉴 또는 식단표)를 분석하여 알레르기 정보를 추출해주세요.

    텍스트: {text}

    주요 확인 알레르기: {', '.join(MAJOR_ALLERGENS)}

    다음 JSON 형식으로 상세하게 응답해주세요:
    {{
        "identified_menus": [
            {{
                "menu_name": "메뉴명",
                "likely_ingredients": ["추정 재료들"],
                "allergen_analysis": [
                    {{
                        "allergen": "알레르기 유발 물질",
                        "confidence": "확실함/가능성높음/가능성있음",
                        "source_ingredient": "유래 재료",
                        "risk_level": "고도/중등도/경도",
                        "notes": "추가 설명"
                    }}
                ]
            }}
        ],
        "summary": {{
            "total_allergens_found": ["발견된 모든 알레르기"],
            "high_confidence_allergens": ["확실한 알레르기"],
            "possible_allergens": ["가능성 있는 알레르기"],
            "menu_safety_score": "1-10점",
            "special_warnings": ["특별 주의사항"]
        }},
        "detailed_recommendations": {{
            "for_allergic_students": ["알레르기 학생을 위한 조언"],
            "for_kitchen_staff": ["조리실 직원을 위한 조언"],
            "alternative_options": ["대체 메뉴 제안"]
        }}
    }}

    참고사항:
    - 한국 음식의 일반적인 재료를 고려하세요
    - 숨겨진 알레르기 성분(소스, 양념 등)도 추정하세요
    - 조리 과정에서의 교차 오염 가능성도 언급하세요
    """
    
    for i in range(max_retries + 1):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            break
        except (RateLimitError, APIConnectionError):
            if i == max_retries:
                return {"error": "API 호출 실패"}
            time.sleep(2 ** i)
    
    try:
        content = res.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "응답 파싱 실패", "raw": content}
    except json.JSONDecodeError:
        return {"error": "JSON 파싱 오류", "raw": content}


# ───────────── 4. AI 기반 종합 보고서 생성 ─────────────
def generate_ai_report(
    analysis_results: List[Dict],
    source_type: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> str:
    """AI를 사용하여 종합적인 알레르기 분석 보고서 생성"""
    client = OpenAI(api_key=api_key)
    
    # 분석 결과를 텍스트로 정리
    results_summary = json.dumps(analysis_results, ensure_ascii=False, indent=2)
    
    prompt = f"""
    당신은 학교 보건교사이자 알레르기 전문가입니다.
    다음 분석 결과를 바탕으로 학교 급식 알레르기 종합 보고서를 작성해주세요.

    분석 대상: {source_type}
    분석 결과: {results_summary}

    보고서는 다음 형식으로 작성해주세요:

    ═══════════════════════════════════════════════════════════════════
                        학교 급식 알레르기 종합 분석 보고서
    ═══════════════════════════════════════════════════════════════════
    
    📅 분석 정보
    - 분석일시: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}
    - 분석대상: {source_type}
    
    📊 분석 요약
    [전체적인 분석 결과 요약 - 3-4줄]
    
    🚨 알레르기 위험도 평가
    
    【고위험 알레르기】
    [생명을 위협할 수 있는 알레르기 상세 설명]
    
    【중등도 위험 알레르기】
    [주의가 필요한 알레르기 상세 설명]
    
    【경도 위험 알레르기】
    [관리 가능한 알레르기 설명]
    
    🔍 상세 분석 결과
    
    [각 메뉴별 상세 분석]
    
    ⚠️ 숨겨진 알레르기 유발 요소
    [눈에 띄지 않지만 주의해야 할 요소들]
    
    💡 대처 방안
    
    【즉시 시행사항】
    1. [구체적인 조치사항]
    2. [구체적인 조치사항]
    
    【예방 조치】
    - [조리실에서의 예방 조치]
    - [배식 시 주의사항]
    - [학생 지도 사항]
    
    【응급 대응 프로토콜】
    1. 경미한 증상: [대처법]
    2. 중등도 증상: [대처법]
    3. 심각한 증상: [대처법]
    
    📋 대체 메뉴 제안
    [알레르기 학생을 위한 대체 메뉴 제안]
    
    📞 비상 연락망
    - 보건실: 내선 [번호]
    - 119 구급대: 119
    - 학교 인근 병원: [병원명] [전화번호]
    - 응급의료정보센터: 1339
    
    ✅ 체크리스트
    □ 알레르기 학생 명단 확인
    □ 대체 급식 준비
    □ 조리실 직원 브리핑
    □ 담임교사 통보
    □ 보건실 의약품 확인
    
    💬 추가 권고사항
    [전문가로서의 추가 조언]
    
    ═══════════════════════════════════════════════════════════════════
    
    작성 시 주의사항:
    1. 의학적으로 정확하면서도 이해하기 쉽게 작성
    2. 실제 학교 현장에서 바로 활용 가능한 구체적인 내용
    3. 위험도에 따른 우선순위 명확히 구분
    4. 한국 학교 급식 환경에 맞는 현실적인 조언
    """
    
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=3000
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"보고서 생성 실패: {str(e)}"


# ───────────── 5. 의학적 정보 생성 함수 ─────────────
def get_medical_info_for_allergen(
    allergen: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Dict:
    """특정 알레르기에 대한 상세 의학 정보 생성"""
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    당신은 소아 알레르기 전문의입니다.
    {allergen} 알레르기에 대해 학교 관계자들이 알아야 할 의학적 정보를 제공해주세요.

    다음 JSON 형식으로 응답해주세요:
    {{
        "medical_name": "의학적 명칭",
        "prevalence": "한국 학생 유병률",
        "mechanism": "알레르기 발생 기전 (쉽게 설명)",
        "symptoms": {{
            "immediate": ["즉각적 증상들"],
            "delayed": ["지연성 증상들"],
            "severe": ["심각한 증상들"]
        }},
        "onset_time": "증상 발현 시간",
        "cross_reactivity": ["교차 반응 가능 물질들"],
        "diagnosis": "진단 방법",
        "treatment": {{
            "emergency": "응급 처치",
            "medication": "사용 가능 약물",
            "long_term": "장기 관리 방법"
        }},
        "school_management": {{
            "prevention": ["학교에서의 예방 조치"],
            "monitoring": ["관찰 포인트"],
            "documentation": ["필요 서류"]
        }},
        "prognosis": "예후 및 성장에 따른 변화"
    }}
    """
    
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        
        content = res.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "의학 정보 파싱 실패"}
    except Exception as e:
        return {"error": f"의학 정보 생성 실패: {str(e)}"}


# ───────────── 6. PDF 텍스트 추출 함수 ─────────────
def extract_pdf_text(pdf_file) -> str:
    """PDF에서 텍스트 추출"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"PDF 읽기 오류: {str(e)}"


# ───────────── 7. Streamlit UI ─────────────
st.set_page_config(
    page_title="🍽️ AI 학교 급식 알레르기 분석 시스템", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍽️ AI 기반 학교 급식 알레르기 분석 시스템")
st.markdown("인공지능이 급식 메뉴의 알레르기 위험을 상세히 분석하고 종합 보고서를 생성합니다.")

# API 키 확인
if "OPENAI_API_KEY" not in st.secrets:
    st.error("⚠️ OpenAI API 키가 설정되지 않았습니다. Streamlit secrets에 OPENAI_API_KEY를 추가해주세요.")
    st.stop()

api_key = st.secrets["OPENAI_API_KEY"]

# 사이드바 - 알레르기 정보 센터
with st.sidebar:
    st.header("📚 알레르기 정보 센터")
    
    # 주요 알레르기 선택
    st.subheader("🔍 알레르기 상세 정보 조회")
    selected_allergen = st.selectbox(
        "알레르기 선택",
        MAJOR_ALLERGENS,
        key="allergen_select"
    )
    
    if st.button("📋 의학 정보 조회", key="get_medical_info"):
        with st.spinner(f"{selected_allergen} 정보를 가져오는 중..."):
            medical_info = get_medical_info_for_allergen(selected_allergen, api_key)
            
            if "error" not in medical_info:
                st.markdown(f"### 🏥 {medical_info.get('medical_name', selected_allergen)}")
                st.info(f"**유병률**: {medical_info.get('prevalence', '정보 없음')}")
                
                with st.expander("🔬 발생 기전"):
                    st.write(medical_info.get('mechanism', ''))
                
                with st.expander("⚡ 증상"):
                    if medical_info.get('symptoms'):
                        st.write("**즉각적 증상**")
                        for s in medical_info['symptoms'].get('immediate', []):
                            st.write(f"- {s}")
                        st.write("**심각한 증상**")
                        for s in medical_info['symptoms'].get('severe', []):
                            st.write(f"- 🚨 {s}")
                
                with st.expander("💊 치료 및 관리"):
                    if medical_info.get('treatment'):
                        st.error(f"**응급처치**: {medical_info['treatment'].get('emergency', '')}")
                        st.warning(f"**약물**: {medical_info['treatment'].get('medication', '')}")
                        st.success(f"**장기관리**: {medical_info['treatment'].get('long_term', '')}")
                
                with st.expander("🏫 학교 관리 지침"):
                    if medical_info.get('school_management'):
                        st.write("**예방 조치**")
                        for p in medical_info['school_management'].get('prevention', []):
                            st.write(f"✓ {p}")
            else:
                st.error(medical_info['error'])
    
    st.divider()
    
    # 빠른 참조
    st.subheader("⚡ 응급 대응 가이드")
    st.error("""
    **아나필락시스 증상**
    - 호흡곤란, 목 조임
    - 전신 두드러기
    - 혈압 급락
    - 의식 저하
    
    **즉시 조치**
    1. 119 신고
    2. 에피펜 투여
    3. 평평하게 눕히기
    4. 하지 거상
    """)

# 메인 콘텐츠
tab1, tab2, tab3, tab4 = st.tabs(["📸 이미지 분석", "📝 텍스트 분석", "📊 Excel 분석", "📄 PDF 분석"])

# 탭 1: 이미지 분석
with tab1:
    st.subheader("급식 사진 AI 분석")
    st.info("급식 사진이나 식단표 이미지를 업로드하면 AI가 알레르기 위험을 상세히 분석합니다.")
    
    uploaded_images = st.file_uploader(
        "이미지 파일 업로드",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="images"
    )
    
    if uploaded_images:
        # 이미지 미리보기
        cols = st.columns(min(len(uploaded_images), 3))
        for i, image in enumerate(uploaded_images):
            with cols[i % 3]:
                st.image(image, caption=f"이미지 {i+1}", use_container_width=True)
        
        if st.button("🤖 AI 분석 시작", key="analyze_images", type="primary"):
            analysis_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, image in enumerate(uploaded_images):
                status_text.text(f"이미지 {i+1}/{len(uploaded_images)} 분석 중...")
                progress_bar.progress((i + 1) / len(uploaded_images))
                
                with st.spinner(f"AI가 이미지 {i+1}을 분석하고 있습니다..."):
                    result = analyze_food_image_with_ai(image.read(), api_key)
                    
                    if "error" not in result:
                        analysis_results.append(result)
                        
                        # 개별 이미지 분석 결과 표시
                        with st.expander(f"📋 이미지 {i+1} 분석 결과", expanded=True):
                            if "menu_items" in result:
                                for item in result["menu_items"]:
                                    st.markdown(f"### 🍽️ {item['name']}")
                                    st.write(f"**재료**: {', '.join(item['ingredients'])}")
                                    
                                    if item.get('allergens'):
                                        st.warning(f"⚠️ 발견된 알레르기: {len(item['allergens'])}개")
                                        for allergen in item['allergens']:
                                            risk_emoji = "🔴" if allergen['risk_level'] == "고도" else "🟡" if allergen['risk_level'] == "중등도" else "🟢"
                                            st.write(f"{risk_emoji} **{allergen['allergen']}** (출처: {allergen['source']})")
                            
                            if "overall_assessment" in result:
                                assessment = result["overall_assessment"]
                                if assessment.get("high_risk_items"):
                                    st.error(f"🚨 고위험 항목: {', '.join(assessment['high_risk_items'])}")
                    else:
                        st.error(f"이미지 {i+1} 분석 실패: {result.get('error', '알 수 없는 오류')}")
            
            progress_bar.progress(1.0)
            status_text.text("분석 완료!")
            
            # AI 종합 보고서 생성
            if analysis_results:
                st.success(f"✅ {len(analysis_results)}개 이미지 분석 완료!")
                
                with st.spinner("AI가 종합 보고서를 작성 중입니다..."):
                    report = generate_ai_report(analysis_results, "이미지 파일", api_key)
                
                st.markdown("### 📊 AI 종합 분석 보고서")
                st.text_area("", report, height=600, key="image_report")
                
                # 보고서 다운로드
                st.download_button(
                    "📥 보고서 다운로드",
                    data=report.encode('utf-8'),
                    file_name=f"AI_알레르기분석보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_image_report"
                )

# 탭 2: 텍스트 분석
with tab2:
    st.subheader("급식 메뉴 텍스트 AI 분석")
    st.info("급식 메뉴나 식단을 입력하면 AI가 알레르기 위험을 분석합니다.")
    
    text_input = st.text_area(
        "급식 메뉴 입력",
        placeholder="예: 쌀밥, 된장찌개(두부, 파, 마늘), 제육볶음(돼지고기, 고추장), 김치, 우유",
        height=150,
        key="text_input"
    )
    
    if st.button("🤖 AI 분석 시작", key="analyze_text", type="primary"):
        if text_input.strip():
            with st.spinner("AI가 메뉴를 분석하고 있습니다..."):
                result = analyze_text_with_ai(text_input, api_key)
            
            if "error" not in result:
                st.success("✅ 분석 완료!")
                
                # 분석 결과 표시
                if "identified_menus" in result:
                    st.markdown("### 🍽️ 메뉴별 분석 결과")
                    
                    for menu in result["identified_menus"]:
                        with st.expander(f"📌 {menu['menu_name']}", expanded=True):
                            st.write(f"**추정 재료**: {', '.join(menu['likely_ingredients'])}")
                            
                            if menu.get("allergen_analysis"):
                                for allergen in menu["allergen_analysis"]:
                                    confidence_emoji = "✅" if allergen['confidence'] == "확실함" else "⚠️" if allergen['confidence'] == "가능성높음" else "❓"
                                    risk_color = "red" if allergen['risk_level'] == "고도" else "orange" if allergen['risk_level'] == "중등도" else "green"
                                    
                                    st.markdown(f"{confidence_emoji} :{risk_color}[**{allergen['allergen']}**] - {allergen['confidence']}")
                                    st.write(f"   출처: {allergen['source_ingredient']} | 위험도: {allergen['risk_level']}")
                                    if allergen.get('notes'):
                                        st.info(f"   💡 {allergen['notes']}")
                
                # 요약 정보
                if "summary" in result:
                    summary = result["summary"]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("발견된 알레르기", len(summary.get('total_allergens_found', [])))
                    with col2:
                        st.metric("안전도 점수", f"{summary.get('menu_safety_score', 0)}/10")
                    with col3:
                        st.metric("고위험 알레르기", len(summary.get('high_confidence_allergens', [])))
                    
                    if summary.get('special_warnings'):
                        st.error("⚠️ **특별 주의사항**")
                        for warning in summary['special_warnings']:
                            st.write(f"- {warning}")
                
                # AI 종합 보고서 생성
                with st.spinner("AI가 상세 보고서를 작성 중입니다..."):
                    report = generate_ai_report([result], "텍스트 입력", api_key)
                
                st.markdown("### 📊 AI 종합 분석 보고서")
                st.text_area("", report, height=600, key="text_report_area")
                
                # 보고서 다운로드
                st.download_button(
                    "📥 보고서 다운로드",
                    data=report.encode('utf-8'),
                    file_name=f"AI_알레르기분석보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_text_report"
                )
            else:
                st.error(f"분석 실패: {result.get('error', '알 수 없는 오류')}")
        else:
            st.warning("분석할 텍스트를 입력해주세요.")

# 탭 3: Excel 분석
with tab3:
    st.subheader("Excel 식단표 AI 분석")
    st.info("Excel 형식의 급식 식단표를 업로드하면 AI가 전체 내용을 분석합니다.")
    
    uploaded_excel = st.file_uploader(
        "Excel 파일 업로드",
        type=["xlsx", "xls"],
        key="excel"
    )
    
    if uploaded_excel:
        try:
            df = pd.read_excel(uploaded_excel)
            st.write("📊 Excel 데이터 미리보기:")
            st.dataframe(df.head(10))
            
            if st.button("🤖 AI 분석 시작", key="analyze_excel", type="primary"):
                # 모든 텍스트 데이터 추출
                all_text = ""
                for col in df.columns:
                    if df[col].dtype == 'object':
                        all_text += f"\n[{col}]\n"
                        all_text += "\n".join(df[col].dropna().astype(str).values) + "\n"
                
                with st.spinner("AI가 Excel 데이터를 분석하고 있습니다..."):
                    result = analyze_text_with_ai(all_text, api_key)
                
                if "error" not in result:
                    st.success("✅ 분석 완료!")
                    
                    # 주요 분석 결과를 DataFrame으로 표시
                    if "identified_menus" in result:
                        allergen_data = []
                        for menu in result["identified_menus"]:
                            for allergen in menu.get("allergen_analysis", []):
                                allergen_data.append({
                                    "메뉴": menu["menu_name"],
                                    "알레르기": allergen["allergen"],
                                    "확실도": allergen["confidence"],
                                    "위험도": allergen["risk_level"],
                                    "출처": allergen["source_ingredient"],
                                    "비고": allergen.get("notes", "")
                                })
                        
                        if allergen_data:
                            allergen_df = pd.DataFrame(allergen_data)
                            st.markdown("### 🔍 알레르기 분석 결과")
                            st.dataframe(
                                allergen_df.style.applymap(
                                    lambda x: 'background-color: #ffcccc' if x == "고도" else 
                                             'background-color: #fff3cd' if x == "중등도" else 
                                             'background-color: #d4edda' if x == "경도" else '',
                                    subset=['위험도']
                                ),
                                use_container_width=True
                            )
                    
                    # AI 종합 보고서 생성
                    with st.spinner("AI가 종합 보고서를 작성 중입니다..."):
                        report = generate_ai_report([result], "Excel 파일", api_key)
                    
                    st.markdown("### 📊 AI 종합 분석 보고서")
                    st.text_area("", report, height=600, key="excel_report_area")
                    
                    # 보고서 다운로드
                    st.download_button(
                        "📥 보고서 다운로드",
                        data=report.encode('utf-8'),
                        file_name=f"AI_알레르기분석보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="download_excel_report"
                    )
                else:
                    st.error(f"분석 실패: {result.get('error', '알 수 없는 오류')}")
                    
        except Exception as e:
            st.error(f"Excel 파일 읽기 오류: {str(e)}")

# 탭 4: PDF 분석
with tab4:
    st.subheader("PDF 식단표 AI 분석")
    st.info("PDF 형식의 급식 식단표를 업로드하면 AI가 내용을 추출하여 분석합니다.")
    
    uploaded_pdf = st.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        key="pdf"
    )
    
    if uploaded_pdf:
        if st.button("🤖 AI 분석 시작", key="analyze_pdf", type="primary"):
            with st.spinner("PDF에서 텍스트를 추출하고 있습니다..."):
                pdf_text = extract_pdf_text(uploaded_pdf)
            
            if "오류" not in pdf_text:
                st.write("📄 추출된 텍스트 미리보기:")
                st.text_area("", pdf_text[:1000] + "...", height=150, key="pdf_preview")
                
                with st.spinner("AI가 내용을 분석하고 있습니다..."):
                    result = analyze_text_with_ai(pdf_text, api_key)
                
                if "error" not in result:
                    st.success("✅ 분석 완료!")
                    
                    # 분석 결과 표시
                    if "summary" in result:
                        summary = result["summary"]
                        if summary.get('total_allergens_found'):
                            st.warning(f"⚠️ 발견된 알레르기: {', '.join(summary['total_allergens_found'])}")
                        
                        if summary.get('special_warnings'):
                            for warning in summary['special_warnings']:
                                st.error(f"🚨 {warning}")
                    
                    # AI 종합 보고서 생성
                    with st.spinner("AI가 종합 보고서를 작성 중입니다..."):
                        report = generate_ai_report([result], "PDF 파일", api_key)
                    
                    st.markdown("### 📊 AI 종합 분석 보고서")
                    st.text_area("", report, height=600, key="pdf_report_area")
                    
                    # 보고서 다운로드
                    st.download_button(
                        "📥 보고서 다운로드",
                        data=report.encode('utf-8'),
                        file_name=f"AI_알레르기분석보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="download_pdf_report"
                    )
                else:
                    st.error(f"분석 실패: {result.get('error', '알 수 없는 오류')}")
            else:
                st.error(pdf_text)

# 하단 정보
st.markdown("---")
with st.expander("📋 시스템 사용 안내"):
    st.markdown("""
    ### 🤖 AI 분석 시스템 특징
    - **고급 이미지 인식**: 음식 사진에서 재료와 조리법을 추정
    - **컨텍스트 이해**: 한국 급식의 특성을 고려한 분석
    - **숨겨진 알레르기 탐지**: 소스, 양념 등에 숨어있는 알레르기 성분 발견
    - **종합 보고서**: 실무에 바로 활용 가능한 상세 보고서 자동 생성
    
    ### 🎯 사용 방법
    1. 분석하고자 하는 자료 형태에 맞는 탭 선택
    2. 파일 업로드 또는 텍스트 입력
    3. AI 분석 시작 버튼 클릭
    4. 생성된 보고서 확인 및 다운로드
    
    ### ⚠️ 중요 안전 수칙
    - AI 분석 결과는 **참고용**입니다
    - 심각한 알레르기가 있는 학생은 **전문의 상담** 필수
    - 의심스러운 경우 **섭취 금지** 원칙
    - 정기적인 **알레르기 검사** 권장
    
    ### 📞 비상 연락처
    - **119**: 응급상황 (아나필락시스, 호흡곤란)
    - **1339**: 응급의료정보센터
    - **학교 보건실**: 즉각적인 1차 대응
    """)

# 추가 스타일
st.markdown("""
<style>
    .stButton > button[kind="primary"] {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="metric-container"] {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)
