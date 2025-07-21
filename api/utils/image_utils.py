from PIL import Image
import io

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_to_base64(pil_img):
    import io, base64
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
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
