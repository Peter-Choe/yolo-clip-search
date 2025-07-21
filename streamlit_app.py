
import streamlit as st
import requests
from PIL import Image
import io
import base64

st.title("유사 이미지 검색 (YOLO + CLIP + FAISS + PGVector)")

search_mode = st.radio("검색 방식 선택", ["이미지 업로드", "텍스트 입력"])

if search_mode == "이미지 업로드":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # 이미지 미리보기
        preview_img = Image.open(uploaded_file)
        st.image(preview_img, caption="업로드한 이미지", use_column_width=True)

        # API 호출
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

        with st.spinner("유사 이미지 검색 중..."):
            response = requests.post("http://localhost:5000/api/search", files=files)

        if response.status_code != 200:
            st.error(f"API 오류: {response.status_code} - {response.text}")
        else:
            data = response.json()
            st.subheader("탐지된 쿼리 이미지 (Bounding Box 포함)")
            st.image("data:image/jpeg;base64," + data["query_image_base64"])

            st.subheader("유사 이미지 결과")
            for i, r in enumerate(data["results"]):
                st.markdown(f"**{i+1}. Label:** {r['label']} | **Cosine Similarity:** {r['similarity']:.4f}")
                st.image("data:image/jpeg;base64," + r["image"]["image_base64"])

elif search_mode == "텍스트 입력":
    query_text = st.text_input("텍스트를 입력하세요 (예: 'female tennis player')")

    if query_text:
        with st.spinner("텍스트 기반 유사 이미지 검색 중..."):
            response = requests.post(
                "http://localhost:5000/api/search_text",  
                json={"text": query_text}
            )

        if response.status_code != 200:
            st.error(f"API 오류: {response.status_code} - {response.text}")
        else:
            data = response.json()
            st.subheader("텍스트 쿼리")
            st.markdown(f"**Query:** {query_text}")

            st.subheader("유사 이미지 결과")
            for i, r in enumerate(data["results"]):
                st.markdown(f"**{i+1}. Label:** {r['label']} | **Cosine Similarity:** {r['similarity']:.4f}")
                st.image("data:image/jpeg;base64," + r["image"]["image_base64"])
