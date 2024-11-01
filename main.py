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

def count(image_path, model):
    
    #runs the model on the image
    image = load_image(image_path)
    results = model(image)
    people_detections = results.pandas().xyxy[0]

    #filter for object we want (people)
    num_people = len(people_detections)
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


#subtle things are more important e.g. people are willing to overlook subtle issues
#read facet paper and download dataset
#get prototype working
#try to add argument for skin colour
#try to fine tune model
#try to put in custom model
#change model vs. change data
#try to see whether the data has a certain factor that impacts the model and causes issues