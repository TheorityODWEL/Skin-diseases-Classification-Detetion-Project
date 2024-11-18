# Skin-diseases-Classification-Detetion-Project

1. All the notebooks were runned using the Kaggle Notebooks
2. All the datasets were uploaded to project by using Roboflow, therefore there is no uploded data
## Classification

1. Open Notebooks in kaggle, BaseModel and EnhancedModel (You can open them seperatly they are not releted to eachother)
2. After opening Editor, choose as accelelator the GPU P100
3. Go to Roboflow website and get your API key, and past it in CHANG-YO-YOUR-API-KEY place
4. Download data using **!pip install roboflow**
5. Run The cells

Also You can accses them by links 
Enhanced Model https://www.kaggle.com/code/millkan/notebook9adfff1c47
Base Model https://www.kaggle.com/code/millkan/notebookf97052f66e
## Detection

1. You need to clone git clone the Yolo Models (you can use ofiical yolo github page)
2. Run the cells

   ### OR



### 1. Install Required Libraries
Ensure you have the ultralytics library installed. Use the following command:
**pip install ultralytics**
### 2. Load the Model
Use the ultralytics library to load your best.pt file:
**from ultralytics import YOLO**

### Load the trained model
model = YOLO('Yolo_Best_Params/best.pt')  # Replace with the path to your best.pt file
### 3. Run Inference
#### Inference on an Image
**results = model('path/to/image.jpg', save=True)**
Add save=True to save the annotated output image in the runs/predict directory.
#### Inference on a Video
**results = model('path/to/video.mp4', save=True)**
Processes the video and saves the output.
#### Inference on a Webcam
**results = model(0)**
Use 0 for the default webcam or specify another webcam ID.
### 4. Customize Prediction Settings
You can adjust parameters for better results:
Confidence Threshold (conf): Filters out low-confidence detections.
IoU Threshold (iou): Manages overlapping bounding boxes.
Example:
**results = model('path/to/image.jpg', conf=0.5, iou=0.4)**
### 5. Visualize Results
#### Display Predictions

**results.show()**
Access Prediction Details
You can access bounding boxes, confidence scores, and class labels programmatically:
**for result in results:
    print(result.boxes)  # Bounding boxes
    print(result.probs)  # Class probabilities (if applicable)**
### 6. Save Outputs
#### To save annotated outputs:
By default, results are saved in the runs/predict directory.
To specify a custom directory:
results = model('path/to/image.jpg', save=True, save_dir='custom/output/directory')
### 7. Evaluate the Model (Optional)
Evaluate the model on a validation dataset using the following command:
**metrics = model.val(data='path/to/data.yaml')**
**print(metrics)**
Replace data.yaml with your dataset configuration file.
### 8. Notes
Replace path/to/best.pt with the actual path to your trained model file.
Ensure input files (images/videos) are accessible and in supported formats.
For further customization or deployment, consult the Ultralytics YOLOv8 Documentation.

