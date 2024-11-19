import argparse
from ultralytics import YOLO
import cv2
import torch
import os

def load_model(weights_path):
    """Load the YOLOv8 model with trained weights."""
    model = YOLO(weights_path)
    return model

def predict(model, image_path, output_dir):
    """Run inference on an image and save the results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image {image_path} not found.")
        return

    # Perform prediction
    results = model(image)
    print("Inference complete.")

    # Save the results
    result_image_path = os.path.join(output_dir, f"predicted_{os.path.basename(image_path)}")
    results.save(output_dir)  # Automatically saves images
    print(f"Result saved to {result_image_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")
    parser.add_argument('--weights', type=str, required=True, help="Path to the trained YOLOv8 weights file (e.g., best.pt).")
    parser.add_argument('--image', type=str, required=True, help="Path to the image for prediction.")
    parser.add_argument('--output', type=str, default='outputs', help="Directory to save the prediction results.")
    args = parser.parse_args()

    # Ensure a valid device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = load_model(args.weights)
    model.to(device)

    # Run prediction
    predict(model, args.image, args.output)

if __name__ == "__main__":
    main()
