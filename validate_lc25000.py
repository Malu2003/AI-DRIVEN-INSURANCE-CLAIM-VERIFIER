"""
Validate the fine-tuned LC25000 model.
"""
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os

def load_model(checkpoint_path):
    # Build model
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier.in_features, 5)  # 5 classes for LC25000
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    classes = checkpoint.get('classes', ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc'])
    best_f1 = checkpoint.get('best_f1', 0.0)
    
    return model, classes, best_f1

def validate_sample(model, image_path, transform, classes):
    model.eval()
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_class = classes[output.argmax(dim=1).item()]
        confidence = probabilities.max().item()
    
    return pred_class, confidence

def main():
    # Load model
    model, classes, best_f1 = load_model('checkpoints/lc25000/best.pth.tar')
    print(f"Loaded model with classes: {classes}")
    print(f"Best F1 score during training: {best_f1:.4f}")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    
    # Test on a few samples
    test_dir = "data/LC25000/test"
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
            
        print(f"\nTesting {class_name}:")
        samples = os.listdir(class_dir)[:3]  # Test first 3 images
        for sample in samples:
            image_path = os.path.join(class_dir, sample)
            pred_class, confidence = validate_sample(model, image_path, transform, classes)
            print(f"Image: {sample}")
            print(f"Predicted: {pred_class} with {confidence*100:.2f}% confidence")
            print(f"Ground Truth: {class_name}")
            print("-" * 50)

if __name__ == "__main__":
    main()