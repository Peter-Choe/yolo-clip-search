import streamlit as st
import requests
from PIL import Image
import io
import base64

import streamlit as st
import requests
from PIL import Image
import io

st.title("이미지 기반 유사 이미지 검색 (CLIP + FAISS)")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Streamlit preview를 위해 PIL 이미지로 한번 열기 (이 시점에서 .read() 하지 않음)
    preview_img = Image.open(uploaded_file)
    st.image(preview_img, caption="업로드한 이미지", use_column_width=True)

    # API 호출을 위한 버퍼 생성
    uploaded_file.seek(0)  # 포인터를 처음으로 되돌려야 서버에서 읽을 수 있음
    files = {
        "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
    }

    with st.spinner("유사 이미지 검색 중..."):
        response = requests.post(
            "http://localhost:5000/api/search",  # FastAPI 서버 주소
            files=files,
        )

    if response.status_code != 200:
        st.error(f"API 오류 발생: {response.status_code} - {response.text}")
    else:
        data = response.json()

        st.subheader("탐지된 쿼리 이미지 (Bounding Box 포함)")
        st.image("data:image/jpeg;base64," + data["query_image_base64"])

        st.subheader("유사 이미지 결과")
        for i, r in enumerate(data["results"]):
            st.markdown(f"**{i+1}. Label:** {r['label']} | **Cosine Similarity:** {r['similarity']:.4f}")
            st.image("data:image/jpeg;base64," + r["image"]["image_base64"])
