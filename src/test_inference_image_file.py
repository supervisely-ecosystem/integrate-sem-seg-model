import requests
import json


SERVER_URL = 'http://127.0.0.1:8000/inference_images'
IMAGE_PATH = 'demo_data/image_01.jpg'
SETTINGS = json.dumps({
    'settings': {'confidence_threshold': 0.25}
})


def run():
    r = requests.post(SERVER_URL, files=(
        # settings can be omitted
        ('settings', (None, SETTINGS, 'text/plain')),
        ('files', open(IMAGE_PATH, 'rb')),
        # for multiple files add more tuples
        # ('files', open(IMAGE_PATH, 'rb')),
    ))
    print(r.status_code)
    print(r.json())

if __name__ == '__main__':
    run()
