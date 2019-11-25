from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from django.http import JsonResponse , FileResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from django.core.files.base import ContentFile
from io import BytesIO
import cv2
import json
import os
import numpy as np
from keras.models import load_model
import face_recognition
from PIL import Image, ImageDraw

emotion_dict = {'anger': 0, 'sadness': 5, 'neutral': 4, 'disgust': 1, 'fear': 6, 'surprise': 2, 'happy': 3}


def pred(face2):
    face2 = cv2.resize(face2, (48, 48))
    face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    face2 = np.reshape(face2, [1, face2.shape[0], face2.shape[1], 1])
    model = load_model("./model/model_v6_23.hdf5")
    predicted_class = np.argmax(model.predict(face2))
    label_map = dict((v, k) for k, v in emotion_dict.items())
    print(predicted_class)
    predicted_label = os.getcwd() + '/Page/graphics/' + label_map[predicted_class] + '.png'
    print(predicted_label)
    return predicted_label


def index(request):
    return render(request, 'index.html')

@csrf_exempt
def test(request):

    a = json.loads(request.body)

    print(os.getcwd())

    data = a['img'][23:]
    print(a['name'])
    img = Image.open(BytesIO(base64.b64decode(data)))
    image_path = os.getcwd() + '/Page/images/' + str(a['name'])
    image_path_rec = os.getcwd() + '/Page/images_rec/' + str(a['name'])
    img.save(image_path + '.jpeg', 'JPEG')
    face = cv2.imread(image_path + '.jpeg')
    pl_face = face_recognition.load_image_file(image_path + '.jpeg')
    face_locations = face_recognition.face_locations(face)
    face_encodings = face_recognition.face_encodings(face, face_locations)
    pil_image = Image.fromarray(pl_face)

    # Опеределыяю размер сдвига
    shift_x = np.shape(face)[1] / 7
    shift_y = np.shape(face)[0] / 7

    draw = ImageDraw.Draw(pil_image)
    # для рисования эмодзи
    # font = ImageFont.truetype('Symbola.ttf', 64, encoding='unic')
    lbl = pred(face)
    i = 0

    if (np.shape(face_encodings)[0] == 0):
        print("Face wasn't recognise")
        exit(1)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # i += 1
        # text_width, text_height = draw.textsize(name)
        # cv2.rectangle(face, (left, bottom), (right, top), (0, 255, 255), 2)
        print('left - ', left, ', right - ', right, ', top - ', top, ', bottom - ', bottom)
        pred_face = face[max(top - np.int32(shift_y), 0):min(bottom + np.int32(shift_y), face.shape[0]),
                    max(left - np.int32(shift_x), 0):min(right + np.int32(shift_x), face.shape[1])]
        lbl = pred(pred_face)
        em = Image.open(lbl)
        em = em.resize((64, 64))
        # em = Image.composite(em, Image.new('RGB', em.size, 'white'), em).show()
        draw.rectangle(((left, top), (right, bottom)), outline=(250, 0, 0), width=5)
        draw.rectangle(((left, top - 75), (left + 75, top)), outline=(250, 0, 0), width=5)
        pil_image.paste(em, (left + 6, top - 70), em)
        # draw.text((left + 2, top - 63), lbl, fill=(255, 255, 255, 255), font=font)
        # cv2.imshow(str(i), pred_face)

    del draw

    pil_image.save(image_path_rec + '.jpeg', 'JPEG')

    return FileResponse(open(image_path_rec + '.jpeg', 'rb'))