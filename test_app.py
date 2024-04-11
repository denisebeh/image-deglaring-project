import requests
import cv2
import numpy as np

# Define the base URL for your API
base_url = 'http://0.0.0.0:4000'

def test_ping():
    # Send GET request to the /ping API endpoint
    response = requests.get(f'{base_url}/ping')
    
    # Verify response status code
    assert response.status_code == 200, 'Expected status code 200'
    
    # Verify the JSON response 
    data = response.json()
    assert data == {'message': 'pong'}, 'Unexpected response data'
    
    print("Service is alive and running!")


def test_infer():
    # Pass an image to the call using the ‘files’ parameter in a POST request as a binary
    # Key = ‘image’ and value= <binary value of the image file> 
    img_path = '/Users/denisebeh/Downloads/SD1_Output/val/out/GlareImage/000058.png'
    img = cv2.imread(img_path)

    # Encode image
    image_binary = cv2.imencode('.png', img)
    image_binary = image_binary[1]

    # Post image for inference
    files = {'image': (image_binary)}
    response = requests.post(f'{base_url}/infer', files=files)

    # Verify response status code
    assert response.status_code == 200, 'Expected status code 200'
    
    # Verify the inference result
    data = response.json()

    assert len(data['image']) > 0, 'No image returned'

    data = np.asarray(bytearray(data['image']), dtype="uint8")
    img_data = cv2.imdecode(data, -1)
    
    assert len(img_data[0]) > 0
    assert len(img_data) > 0

    print("Inference result successfully received!")

if __name__ == '__main__':
    # Run the tests
    test_ping()
    test_infer()