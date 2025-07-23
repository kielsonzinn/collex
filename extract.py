import os
import cv2
import numpy as np
from PIL import Image

def extract_bottle_caps_from_directory(
    input_dir="images/quadro",
    output_dir="images/quadro_extraido",
    size=(100, 100),
    dp=1.2,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=30,
    maxRadius=60
):
    os.makedirs(output_dir, exist_ok=True)
    total_caps = 0

    for image_name in os.listdir(input_dir):
        if not image_name.lower().endswith((".jpeg", ".jpg", ".png")):
            continue

        image_path = os.path.join(input_dir, image_name)
        image_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])

        if os.path.exists(image_output_dir) and len(os.listdir(image_output_dir)) > 0:
            print(f"[SKIP] {image_name} já foi processada. Pulando...")
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
            print(f"Analisando imagem: {image_name}, {len(circles)} tampinhas detectadas.")

            for i, (x, y, r) in enumerate(circles, start=1):
                # Criar máscara circular
                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
                result = cv2.bitwise_and(image, mask)
                cropped = result[y - r:y + r, x - r:x + r]

                # Converter para PIL e remover fundo preto
                pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA))
                datas = pil_img.getdata()
                new_data = []
                for item in datas:
                    if item[0] == 0 and item[1] == 0 and item[2] == 0:
                        new_data.append((0, 0, 0, 0))
                    else:
                        new_data.append(item)
                pil_img.putdata(new_data)

                # Redimensionar
                if size:
                    pil_img = pil_img.resize(size, Image.LANCZOS)

                # Salvar como PNG com fundo transparente
                output_path = os.path.join(image_output_dir, f"cap_{i}.png")
                pil_img.save(output_path, "PNG")

            total_caps += len(circles)
            print(f"Imagem {image_name} - {len(circles)} tampinhas extraídas.")
        else:
            print(f"Imagem {image_name} - Nenhuma tampinha detectada.")

    print(f"\nTampinhas detectadas no total: {total_caps}")
    return total_caps


if __name__ == "__main__":
    extract_bottle_caps_from_directory()
