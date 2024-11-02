import argparse
from ultralytics import YOLO
import cv2

def load_model(): #load YOLO model, version 8
    model = YOLO('yolov8n.pt')
    return model

def load_image(image_path):
    #load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    return image

def extract_boxes(box, image):
    x1, y1, x2, y2 = map(int, box.xyxy[0]) #coordinates for boxes
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) #draw rectangles on specified coordinates
    cv2.putText(image, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #labels the boxes

def output_boxes(image):
    output_path = "boxedImage.jpg"
    cv2.imwrite(output_path, image)
    print(f"Image with bounding boxes saved as {output_path}")

def count(image_path, model):
    
    #runs the model on the image
    image = load_image(image_path)
    results = model(image)
    
    num_people = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:
                num_people += 1
                extract_boxes(box, image)
    
    output_boxes(image)
    return num_people

def main():

    #sets up terminal interface
    parser = argparse.ArgumentParser(description="Counts people in an image.")
    parser.add_argument("image_path", type=str, help="Path to the image for model to process.") #adds argument
    args = parser.parse_args()

    model = load_model()
    try:
        num_people = count(args.image_path, model)
        print(f"Number of people in image: {num_people}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

#terminal run command: python3 main.py images/sample_image.jpg


#subtle things are more important e.g. people are willing to overlook subtle issues
#read facet paper and download dataset
#get prototype working
#try to add argument for skin colour
#try to fine tune model
#try to put in custom model
#change model vs. change data
#try to see whether the data has a certain factor that impacts the model and causes issues