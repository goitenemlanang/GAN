from flask import Flask, render_template, request
from torch import nn
import numpy as np
from PIL import Image
import re
from io import BytesIO
from flask_cors import CORS
import base64
from random import randint
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import torch
from torchvision import transforms
import pyodbc
import bs4
from bs4 import BeautifulSoup


# from tensorflow.keras.models import load_model
class Generator(nn.Module):
    def __init__(self, im_channels):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(im_channels + 1, 64, 5, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, im_channels, 3, stride=1, padding=1),
        )

    def forward(self, X):
        out = self.net(X)
        return out


def delete_all_pixels_except_white(image):
    """Deletes all pixels in an image except white pixels.

    Args:
      image: A numpy array representing the image.

    Returns:
      A numpy array representing the image with all non-white pixels removed.
    """

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image with white pixels as 255 and
    # non-white pixels as 0.
    thresh = cv2.threshold(gray, 248, 255, cv2.THRESH_BINARY)[1]

    # Invert the image so that white pixels are 0 and non-white pixels are 255.
    inverted = 255 - thresh

    # Set all non-white pixels to 0.
    image[inverted > 0] = 0

    return image


model = Generator(im_channels=3)
model.load_state_dict(torch.load("G:\GAN_IMAGEE\generator.pt"))

img_name = "img.jpg"

app = Flask(__name__)
CORS(app)


@app.route("/")
def register_page():
    return render_template("index.html")


@app.route("/brush")
def index():
    return render_template("main.html")


@app.route("/output", methods=["GET"])
def get_output():
    return render_template("output.html", img_name=img_name)


@app.route("/model", methods=["POST"])
def get_image():
    transform = transforms.ToTensor()
    name = "image.jpg"
    image_b64 = request.values["imageBase64"]
    image_data = base64.b64decode(re.sub("^data:image/.+;base64,", "", image_b64))

    image_PIL = Image.open(BytesIO(image_data))
    image_np = np.array(image_PIL)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_np = cv2.resize(image_np, (32, 32))
    cv2.imwrite("G:\GAN_IMAGEE\imgs\in" + name, image_np)

    im = cv2.imread("G:\GAN_IMAGEE\imgs\in" + name)
    image = delete_all_pixels_except_white(im)  # mas k image
    image = transform(image)  # mask image
    image_np = transform(image_np)  # mask image

    batch_masked = image_np.unsqueeze(0).clone() * (1 - image.unsqueeze(0))
    batch_masked = torch.cat((batch_masked, image.unsqueeze(0)[:, :1]), dim=1)
    pred = model(batch_masked).clamp(0, 1).detach()
    pred_i = transforms.ToPILImage()(pred[0])
    pred_i.save("G:\GAN_IMAGEE\imgs\pred" + name)

    in_image = cv2.imread("G:\GAN_IMAGEE\imgs\in" + name)
    pred_image = cv2.imread("G:\GAN_IMAGEE\imgs\pred" + name)
    lower_white = np.array([240, 240, 240], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(in_image, lower_white, upper_white)

    # Replace the white pixels in the first image with corresponding pixels from the second image
    in_image[mask > 0] = pred_image[mask > 0]

    # Save the first image with the pixels filled
    cv2.imwrite("G:\GAN_IMAGEE\imgs\out" + name, in_image)

    size = (486, 286)
    out = cv2.imread("G:\GAN_IMAGEE\imgs\out" + name)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = cv2.resize(out, size, interpolation=cv2.INTER_CUBIC)

    in_i = cv2.imread("G:\GAN_IMAGEE\imgs\in" + name)
    in_i = cv2.cvtColor(in_i, cv2.COLOR_BGR2RGB)
    in_i = cv2.resize(in_i, size, interpolation=cv2.INTER_CUBIC)

    global img_name
    img_name = "img" + str(randint(11, 99)) + ".jpg"
    plt.imsave("G:\GAN_IMAGEE\static\images\i" + img_name, in_i)
    plt.imsave("G:\GAN_IMAGEE\static\images\o" + img_name, out)
    return "success"


# app.run(debug=True)

conn = pyodbc.connect(
    driver="{ODBC Driver 17 for SQL Server}",
    host="GỌITÊNEMLÀNẮNG\SQLLANH",
    database="SQLUSER",
    UID="sa",
    PWD="123",
    trusted_connection="yes",
)


@app.route("/login", methods=["post"])
def login():
    user_name = request.form.get("username")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE Username = '" + user_name + "'")
    rows = cursor.fetchall()
    with open("./templates/index.html", encoding="utf-8") as fp:
        soup1 = bs4.BeautifulSoup(fp.read(), "html.parser")

    if len(rows) == 0:
        errors = soup1.find(class_="modal")
        errors["class"].remove("hide")
        errorss = soup1.find(class_="error")
        errorss["style"] = "display:block"
    else:
        # errors = soup1.find(class_ = "modal")
        # errors["class"].add("hide")

        imgtag = soup1.find(id="canvas")
        # imgtag["style"] = f"background-image: url({rows[0][2]}); background-size: cover; background-position: center center; width: 100%; height: 100%;background-repeat: no-repeat;"
        imgtag[
            "style"
        ] = f"background-image: url({rows[0][2]}); background-size: 100% 100%; background-repeat: no-repeat; width: 100%; height: 100%; border: 2px dashed black; border-radius: 2px;"
        # avatar
        imgava = soup1.find(id="avatar")
        imgava["src"] = rows[0][3]
        imgava["style"] = "margin-right:5px;"

        loginn = soup1.find(class_="card__login")
        nameUser = soup1.find(id="name__avatar")
        nameUser.replace_with(rows[0][1])

        loginn["style"] = "display:none"
        logoutt = soup1.find(class_="card__logout")
        logoutt["style"] = "display:flex"
    return soup1.prettify()


@app.route("/logout", methods=["post"])
def logout():
    with open("./templates/index.html", encoding="utf-8") as fp:
        soup1 = bs4.BeautifulSoup(fp.read(), "html.parser")
    loginn = soup1.find(class_="card__login")
    loginn["style"] = "display:block"

    logoutt = soup1.find(class_="card__logout")
    logoutt["style"] = "display:none"
    return render_template("index.html")


app.run(debug=True)
