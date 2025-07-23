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

# ✅ Converte arquivo enviado para base64
def image_to_base64(file):
    return base64.b64encode(file.read()).decode("utf-8")

# ✅ Converte imagem OpenCV para base64
def cv2_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

# ✅ Converte PIL para OpenCV
def pil_to_cv2(pil_img):
    img = np.array(pil_img)
    if img.shape[2] == 4:  # RGBA -> BGRA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:  # RGB -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# ✅ Converte base64 para OpenCV
def base64_to_cv2(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# ✅ Extrai tampinhas da imagem enviada
def extract_bottle_caps_from_upload(file_storage, size=(100, 100)):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=500,
        param1=100,
        param2=50,
        minRadius=80,
        maxRadius=200
    )

    caps = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            mask = np.zeros_like(img, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

            result = cv2.bitwise_and(img, mask)
            cropped = result[y - r:y + r, x - r:x + r]

            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGBA))

            datas = pil_img.getdata()
            new_data = []
            for item in datas:
                if item[0] == 0 and item[1] == 0 and item[2] == 0:
                    new_data.append((0, 0, 0, 0))
                else:
                    new_data.append(item)
            pil_img.putdata(new_data)

            if size:
                pil_img = pil_img.resize(size, Image.LANCZOS)

            caps.append(pil_img)

    return caps

# ✅ Prepara imagem (agora recebe base64)
def prepare_image(image_base64, size=(100, 100)):
    img = base64_to_cv2(image_base64)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    return gray

# ✅ Compara duas imagens
def compare_images(image1, image2):
    return ssim(image1, image2)

# ✅ Busca tampinhas similares no diretório
def find_similar_caps(cap_image, caps_dir, top_n=5, threshold=0.75):
    scores = []
    for root, _, files in os.walk(caps_dir):
        for file in files:
            if file.lower().endswith(".png"):
                cap_path = os.path.join(root, file)
                cap_gray = cv2.cvtColor(cv2.imread(cap_path), cv2.COLOR_BGR2GRAY)
                cap_gray = cv2.resize(cap_gray, cap_image.shape[::-1])
                score = compare_images(cap_image, cap_gray)
                scores.append((score, cap_path))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [cap for score, cap in scores[:top_n] if score >= threshold]

# ✅ Endpoint de upload
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
                print(f"Achou...")
                # ✅ Converter todos os caminhos encontrados para base64
                related_imgs_base64 = []
                for cap_path in similar_caps:
                    img = cv2.imread(cap_path)
                    related_imgs_base64.append(cv2_to_base64(img))

                retorno[image_base64] = related_imgs_base64
                break
            threshold = round(threshold - 0.05, 2)

    return render_template("final.html", related=retorno)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
