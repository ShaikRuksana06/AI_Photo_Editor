from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_FOLDER = 'models'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load colorization model
prototxt = os.path.join(MODEL_FOLDER, 'colorization_deploy_v2.prototxt')
caffemodel = os.path.join(MODEL_FOLDER, 'colorization_release_v2.caffemodel')
pts_npy = os.path.join(MODEL_FOLDER, 'pts_in_hull.npy')

net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
pts = np.load(pts_npy)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.transpose().reshape(2, 313, 1, 1)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# 🔑 DeepAI API Key
DEEP_AI_API_KEY = '.........'  # Replace with your real API key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/edit', methods=['POST'])
def edit():
    file = request.files['image']
    operation = request.form['operation']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    output_path = os.path.join(UPLOAD_FOLDER, 'edited_' + filename)

    if operation == 'remove_scratches':
        # Try DeepAI scratch removal
        try:
            response = requests.post(
                "https://api.deepai.org/api/image-scrubber",
                files={'image': open(filepath, 'rb')},
                headers={'api-key': DEEP_AI_API_KEY}
            )
            data = response.json()
            print("DeepAI Response:", data)

            if 'output_url' in data:
                result_url = data['output_url']
                result_data = requests.get(result_url).content
                with open(output_path, 'wb') as f:
                    f.write(result_data)
            else:
                print("⚠️ DeepAI failed, falling back to OpenCV.")
                # Fallback: OpenCV scratch removal
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.medianBlur(gray, 3)
                laplacian = cv2.Laplacian(blur, cv2.CV_64F)
                laplacian = cv2.convertScaleAbs(laplacian)
                _, mask = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
                cv2.imwrite(output_path, result)
        except Exception as e:
            print("Exception during DeepAI call:", e)
            # Fallback in case of exception
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 3)
            laplacian = cv2.Laplacian(blur, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            _, mask = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            cv2.imwrite(output_path, result)

    elif operation == 'black_white':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(output_path, result)

    elif operation == 'colorize':
        bw = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        normalized = bw.astype("float32") / 255.0
        lab = cv2.cvtColor(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        net_input = cv2.dnn.blobFromImage(L, 1.0, (224, 224))
        net.setInput(net_input)
        ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_dec_us = cv2.resize(ab_dec, (bw.shape[1], bw.shape[0]))
        lab_out = np.concatenate((L[:, :, np.newaxis], ab_dec_us), axis=2)
        lab_out = lab_out.astype("uint8")
        result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path, result)

    elif operation == 'rotate':
        result = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(output_path, result)

    elif operation == 'grayscale':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(output_path, result)

    else:
        return "❌ Invalid operation", 400

    return send_from_directory(UPLOAD_FOLDER, 'edited_' + filename)

if __name__ == '__main__':
    print("🚀 Starting Flask Photo Editor...")
    app.run(debug=True)
