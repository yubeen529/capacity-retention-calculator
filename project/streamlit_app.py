import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="배터리 용량 예측기", page_icon="🔋", layout="wide")
st.title("배터리 용량 예측기")
st.markdown("**인공지능 모델을 사용한 배터리 용량 예측 시스템**")
st.markdown("---")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('battery_model.pkl')
        return model
    except FileNotFoundError:
        st.error("battery_model.pkl 파일을 찾을 수 없습니다!")
        st.error("먼저 모델을 훈련하고 저장해주세요.")
        st.stop()
    except Exception as e:
        st.error(f"모델 로드 중 오류가 발생했습니다: {str(e)}")
        st.stop()

model = load_model()
st.success("모델이 성공적으로 로드되었습니다!")

st.sidebar.header("입력 파라미터")
st.sidebar.markdown("배터리 특성 값을 입력하세요:")

porosity = st.sidebar.number_input(
    "🔸 Porosity (다공성)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.365,
    step=0.001,
    format="%.3f",
    help="배터리 전극의 다공성 (0.0 ~ 1.0)"
)

diffusivity = st.sidebar.number_input(
    "🔸 Effective diffusivity (유효확산도)", 
    min_value=1e-9, 
    max_value=1e-6, 
    value=9.5e-8,
    step=1e-9,
    format="%.2e",
    help="배터리 내부의 Li이온 유효확산도 (c㎡/s)"
)

predict_button = st.sidebar.button("예측하기", type="primary", use_container_width=True)
col1, col2 = st.columns([2, 1])

with col1:
    if predict_button:
        try:
            input_data = pd.DataFrame({
                'Porosity': [porosity],
                'diffusivity': [diffusivity]
            })
            
            prediction = model.predict(input_data)
            prediction_value = prediction[0]
            
            st.subheader("예측 결과")
            st.markdown(f"""
            <div style="padding: 20px; background-color: #e8f5e8; border-radius: 10px; border-left: 5px solid #4CAF50;">
                <h2 style="color: #2e7d32; margin: 0;">예상 배터리 용량</h2>
                <h1 style="color: #1b5e20; margin: 10px 0;">{prediction_value:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("입력값 확인")
            input_col1, input_col2 = st.columns(2)
            
            with input_col1:
                st.metric(
                    label="Porosity", 
                    value=f"{porosity:.3f}",
                    help="입력된 다공성 값"
                )
            
            with input_col2:
                st.metric(
                    label="Diffusivity", 
                    value=f"{diffusivity:.2e}",
                    help="입력된 확산도 값"
                )
                
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
    
    else:
        st.info("왼쪽 사이드바에서 값을 입력하고 **예측하기** 버튼을 클릭하세요.")
        st.subheader("입력 예시")
        
        example_data = pd.DataFrame({
            '구분': ['최소값', '평균값', '최대값'],
            'Porosity': [0.313, 0.365, 0.428],
            'Diffusivity': ['4.22e-08', '9.49e-08', '1.75e-07'],
            '예상 범위': ['91% ~ 95%', '95% ~ 96%', '96% ~ 99%']
        })
        
        st.table(example_data)

with col2:
    st.subheader("모델 설명")
    
    st.metric(
        label="양극 활물질", 
        value="NCM811"
    )

    st.metric(
        label="바인더", 
        value="PVDF"
    )
    
    st.metric(
        label="도전재", 
        value="Super P-Li"
    )
    
    st.metric(
        label="슬러리 조성 (활:도:바)", 
        value="94:3:3"
    )

st.markdown("---")

with st.expander("사용법 안내"):
    st.markdown("""
    ### 사용 방법
    1. **왼쪽 사이드바**에서 **Porosity**와 **Effective diffusivity** 값을 입력하세요
    2. **예측하기** 버튼을 클릭하면 배터리 용량을 예측합니다
    3. 결과는 백분율(%)로 표시됩니다
    
    ### 입력 범위 가이드    
    - **Porosity**: 0.313 ~ 0.428 (일반적인 범위)
    - **Effective diffusivity**: 4.22e-08 ~ 1.75e-07 c㎡/s (일반적인 범위)
    
    ### 주의사항
    - 입력 범위를 벗어난 값은 예측 정확도가 떨어질 수 있습니다
    - 이 모델은 특정 조건에서 훈련된 모델입니다
    """)

with st.expander("모델 정보"):
    st.markdown("""
    ### 사용된 알고리즘
    - **XGBoost Regressor** (Gradient Boosting)
    - 트리 기반 앙상블 모델
    
    ### 특성 중요도
    - **Porosity**: 54.66% (더 중요)
    - **Effective diffusivity**: 45.34%
    
    ### 훈련 데이터
    - 총 215개 샘플
    - 훈련: 172개, 테스트: 43개
    - 결측값: 없음
    """)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>배터리 용량 예측 시스템 | XGBoost ML Model | Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)
