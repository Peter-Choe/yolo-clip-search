from PIL import Image
import io

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    바이트 데이터를 PIL 이미지로 변환
    - 입력: 이미지 바이트 (예: UploadFile.read() 결과)
    - 출력: RGB 포맷의 PIL 이미지 객체
    """
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_to_base64(pil_img):
    """
    PIL 이미지를 Base64 문자열로 인코딩
    - 입력: PIL 이미지
    - 출력: "data:image/jpeg;base64,..." 형태의 문자열 없이, 인코딩된 base64 string
    """
    import io, base64
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")  # JPEG 포맷으로 메모리 버퍼에 저장
    return base64.b64encode(buf.getvalue()).decode("utf-8")


from PIL import ImageDraw, ImageFont

def draw_bbox_on_image(
    image,  # PIL.Image
    bbox,   # (x1, y1, x2, y2)
    label: str = None,
    color: str = "red",
    width: int = 3,
):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if label:
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except:
            font = ImageFont.load_default()

        # textsize → getbbox 또는 getsize로 교체
        try:
            bbox_text = font.getbbox(label)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        except AttributeError:
            text_width, text_height = font.getsize(label)

        # 텍스트 박스 배경 그리기
        text_bg = [x1, y1 - text_height - 4, x1 + text_width + 6, y1]
        draw.rectangle(text_bg, fill=color)

        # 텍스트 쓰기
        draw.text((x1 + 3, y1 - text_height - 2), label, fill="white", font=font)

    return image


def crop_with_padding(image, bbox, pad=15):
    """
    이미지와 bbox (x1, y1, x2, y2)로부터 padding 포함 crop을 반환
    """
    w, h = image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image.crop((x1, y1, x2, y2))
