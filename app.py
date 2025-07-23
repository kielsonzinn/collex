from flask import Flask, render_template, request
import base64
import cv2
import numpy as np
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
from extract import extract_bottle_caps_from_directory
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    stream=sys.stdout
)

print("🔄 Processando imagens do diretório antes de iniciar o servidor...")
extract_bottle_caps_from_directory("images/quadro", "images/quadro_extraido")
print("✅ Processamento concluído. Iniciando servidor Flask...")

app = Flask(__name__)

def image_to_base64(file):
    return base64.b64encode(file.read()).decode("utf-8")

def cv2_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def pil_to_cv2(pil_img):
    img = np.array(pil_img)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def base64_to_cv2(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def rotate_image(img, angle):
    """Roda a imagem em torno do centro, mantendo o mesmo tamanho."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def extract_bottle_caps_from_upload(file_storage, size=(100, 100), save_dir=None):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        print("[ERRO] Não foi possível decodificar a imagem enviada.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(img.shape[:2]) // 3,
        param1=100,
        param2=50,
        minRadius=80,
        maxRadius=200
    )

    caps = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"[INFO] {len(circles)} tampinhas detectadas no upload.")

        for i, (x, y, r) in enumerate(circles, start=1):
            mask = np.zeros_like(img, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
            result = cv2.bitwise_and(img, mask)

            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(img.shape[1], x + r), min(img.shape[0], y + r)
            cropped = result[y1:y2, x1:x2]

            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA))
            pil_img.putdata([
                (0, 0, 0, 0) if (r == 0 and g == 0 and b == 0) else (r, g, b, a)
                for (r, g, b, a) in pil_img.getdata()
            ])

            if size:
                pil_img = pil_img.resize(size, Image.LANCZOS)

            caps.append(pil_img)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, f"cap_{i}.png")
                pil_img.save(output_path, "PNG")
                print(f"[SALVO] {output_path}")
    else:
        print("[INFO] Nenhuma tampinha detectada no upload.")

    return caps

def prepare_image(image_base64, size=(100, 100)):
    img = base64_to_cv2(image_base64)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    return gray

def compare_images(image1, image2):
    return ssim(image1, image2)

def find_similar_caps(cap_image, caps_dir, top_n=5, threshold=0.75, rotate_step=180):
    """
    Compara tampinhas com rotações diferentes.
    Retorna [(score, caminho), ...]
    """
    scores = []
    h, w = cap_image.shape[:2]
    rotated_versions = [rotate_image(cap_image, angle) for angle in range(0, 360, rotate_step)]

    for root, _, files in os.walk(caps_dir):
        for file in files:
            if file.lower().endswith(".png"):
                cap_path = os.path.join(root, file)
                cap_gray = cv2.cvtColor(cv2.imread(cap_path), cv2.COLOR_BGR2GRAY)
                cap_gray = cv2.resize(cap_gray, (w, h))

                best_score = 0
                for rotated in rotated_versions:
                    score = compare_images(rotated, cap_gray)
                    if score > best_score:
                        best_score = score
                        if best_score >= 1.0:
                            break

                if best_score >= threshold:
                    scores.append((best_score, cap_path))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_n]

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        uploaded_file = request.files["image"]
        caps = extract_bottle_caps_from_upload(uploaded_file)
        images = [cv2_to_base64(pil_to_cv2(cap)) for cap in caps]
        return render_template("select.html", images=images)
    return render_template("upload.html")

@app.route("/select", methods=["POST"])
def select_images():
    selecteds = request.form.getlist("selected")
    quadro_extraido_dir = "images/quadro_extraido"
    retorno = {}

    for image_base64 in selecteds:
        cap_gray = prepare_image(image_base64)
        threshold = 1.00
        while threshold > 0.00:
            print(f"Analisando considerando semelhança de {threshold}")
            similar_caps = find_similar_caps(cap_gray, quadro_extraido_dir, threshold=threshold)
            if similar_caps:
                related_imgs_data = []
                for score, cap_path in similar_caps:
                    img = cv2.imread(cap_path)
                    related_imgs_data.append({
                        "img": cv2_to_base64(img),
                        "score": round(score * 100, 2)  # Percentual com 2 casas decimais
                    })
                retorno[image_base64] = related_imgs_data
                break
            threshold = round(threshold - 0.05, 2)

    return render_template("final.html", related=retorno)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
