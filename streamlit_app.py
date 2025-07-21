import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="유사 이미지 검색", layout="centered")
st.title("유사 이미지 검색 (YOLO + CLIP + FAISS + PGVector)")

# --- 안내 문구 ---
st.markdown("""
이 서비스는 **7가지 객체 클래스**에 대해서만 탐지 및 검색이 가능합니다:

📌 `person`, `car`, `cell phone`, `laptop`, `book`, `handbag`, `sports ball`

❗ 그 외 객체(예: 고양이, 강아지 등)는 검색 결과가 없을 수 있습니다.
""")


# --- 공통 출력 함수 ---
def show_results(data, is_text=False, query=None):
    if not is_text:
        st.subheader("쿼리 이미지 (객체 탐지 포함)")
        st.image("data:image/jpeg;base64," + data["query_image_base64"])
    else:
        st.subheader("텍스트 쿼리")
        st.markdown(f"Query: `{query}`")

    st.subheader("유사 이미지 결과")
    for i, r in enumerate(data["results"]):
        st.markdown(f"{i+1}. Label: `{r['label']}` | 유사도: `{r['similarity']:.4f}`")
        st.image("data:image/jpeg;base64," + r["image"]["image_base64"], use_container_width=True)


# --- 검색 모드 선택 ---
search_mode = st.radio("검색 방식 선택", ["이미지 업로드", "텍스트 입력"])

# === 이미지 업로드 모드 ===
if search_mode == "이미지 업로드":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="업로드한 이미지", use_container_width=True)
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

        with st.spinner("유사 이미지 검색 중..."):
            response = requests.post("http://localhost:5000/api/search", files=files)

        if response.ok:
            show_results(response.json())
        else:
            st.error(f"검색 실패: {response.status_code} - {response.text}")

# === 텍스트 입력 모드 ===
elif search_mode == "텍스트 입력":
    query_text = st.text_input("텍스트를 입력하세요 (예: handbag)", "")

    if query_text:
        with st.spinner("텍스트 기반 이미지 검색 중..."):
            response = requests.post("http://localhost:5000/api/search_text", json={"text": query_text})

        if response.ok:
            show_results(response.json(), is_text=True, query=query_text)
        else:
            st.error(f"검색 실패: {response.status_code} - {response.text}")
