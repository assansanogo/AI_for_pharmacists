import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

import numpy as np
import pandas as pd
import cv2

def reads_text_on_image(new_file):
    for i,val in enumerate(new_file):
        yield [val['description']]+ [el["x"] for el in val["boundingPoly"]["vertices"]] + [el["y"] for el in val["boundingPoly"]["vertices"]]

def process(my_im):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()
    L = []

    success, content = cv2.imencode('.png', my_im)\

    image = types.Image(content=content.tobytes())

    # Performs label detection on the image file
    response = client.text_detection(image=image)
    texts = response.text_annotations

    for i in range(len(texts)):
        vertices = [texts[i].description] + [vertex.x for vertex in texts[i].bounding_poly.vertices] + [vertex.y for vertex in texts[i].bounding_poly.vertices]
        L.append(vertices)

    df = pd.DataFrame(L)

    resultat = None
    try:
        df.columns = ["reading", "x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"]
        resultat = df
        resultat.to_csv("./reading.csv", index=None, sep=',')
    except ValueError:
        pass

    return resultat


def dataframe_csv():
    df = pd.read_csv("./reading.csv")
    print(df)

if __name__=='__main__':
    main()
