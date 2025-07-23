import os
import cv2
import numpy as np
from PIL import Image
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    stream=sys.stdout
)

def extract_bottle_caps_from_directory(
    input_dir="images/quadro",
    output_dir="images/quadro_extraido",
    size=(100, 100),
    dp=1.2,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=30,
    maxRadius=60,
    extra_radius=15
):
    os.makedirs(output_dir, exist_ok=True)
    total_caps = 0

    for image_name in os.listdir(input_dir):
        if not image_name.lower().endswith((".jpeg", ".jpg", ".png")):
            continue

        image_path = os.path.join(input_dir, image_name)
        image_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])

        if os.path.exists(image_output_dir) and len(os.listdir(image_output_dir)) > 0:
            logging.info(f"[SKIP] {image_name} já foi processada. Pulando...")
            continue

        os.makedirs(image_output_dir, exist_ok=True)

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            logging.info(f"Analisando imagem: {image_name}, {len(circles)} tampinha(s) detectada(s).")

            for i, (x, y, r) in enumerate(circles, start=1):
                r = r + extra_radius

                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                result = cv2.bitwise_and(image, mask)

                x1, y1 = max(0, x - r), max(0, y - r)
                x2, y2 = min(image.shape[1], x + r), min(image.shape[0], y + r)
                cropped = result[y1:y2, x1:x2]

                pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA))
                pil_img.putdata([
                    (0, 0, 0, 0) if (r == 0 and g == 0 and b == 0) else (r, g, b, a)
                    for (r, g, b, a) in pil_img.getdata()
                ])

                if size:
                    pil_img = pil_img.resize(size, Image.LANCZOS)

                output_path = os.path.join(image_output_dir, f"cap_{i}.png")
                pil_img.save(output_path, "PNG")

            total_caps += len(circles)
            logging.info(f"Imagem {image_name} - {len(circles)} tampinha(s) extraída(s).")
        else:
            logging.info(f"Imagem {image_name} - Nenhuma tampinha detectada.")

    logging.info(f"✅ Tampinhas detectadas no total: {total_caps}")
    return total_caps
