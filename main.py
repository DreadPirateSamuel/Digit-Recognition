import pygame
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import asyncio
import platform

# Neural network model (matched to train.py)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):  # Only digits 0-9
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load pre-trained model
def load_model():
    model = SimpleCNN(num_classes=10)
    try:
        model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("Model not found. Please train the model first using train.py.")
        exit()
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        exit()
    model.eval()
    return model

# Preprocess image
def preprocess_image(surface):
    data = pygame.image.tostring(surface, 'RGB')
    img = Image.frombytes('RGB', surface.get_size(), data)
    img = img.convert('L')
    img_array = np.array(img)
    # Adaptive thresholding to preserve shape
    img_array = np.where(img_array > 100, 255, 0).astype(np.uint8)
    # Center digit using bounding box
    non_zero = np.where(img_array > 10)
    if len(non_zero[0]) == 0:
        img = Image.fromarray(img_array)
    else:
        y_min, y_max = non_zero[0].min(), non_zero[0].max()
        x_min, x_max = non_zero[1].min(), non_zero[1].max()
        w, h = x_max - x_min + 1, y_max - y_min + 1
        max_side = max(w, h)
        new_size = int(max_side * 1.5)
        new_img = np.zeros((new_size, new_size), dtype=np.uint8)
        x_offset = (new_size - w) // 2
        y_offset = (new_size - h) // 2
        new_img[y_offset:y_offset+h, x_offset:x_offset+w] = img_array[y_min:y_max+1, x_min:x_max+1]
        img = Image.fromarray(new_img).resize((28, 28), Image.Resampling.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

# Segment canvas with adjusted centers
def segment_canvas(canvas, num_segments):
    segments = []
    canvas_width = 56  # Canvas is 56x56
    segment_width = 36  # Adjusted to capture full digits
    step = canvas_width // num_segments
    centers = [step // 2 + i * step for i in range(num_segments)]  # Center points
    for i, center in enumerate(centers):
        segment = pygame.Surface((segment_width, segment_width))
        segment.fill((0, 0, 0))
        src_x = max(0, center - segment_width // 2)
        src_x = min(src_x, canvas_width - segment_width)
        segment.blit(canvas, (0, 0), (src_x, 0, segment_width, segment_width))
        segment = pygame.transform.scale(segment, (28, 28))
        pixels = pygame.surfarray.array3d(segment)
        mean_pixel = np.mean(pixels)
        max_pixel = np.max(pixels)
        non_zero_count = np.sum(pixels > 10)
        print(f"Segment {i} mean pixel: {mean_pixel:.2f}, max pixel: {max_pixel:.2f}, non-zero pixels: {non_zero_count}, x-range: {(src_x*10)}-{(src_x+segment_width)*10}")
        if mean_pixel < 5 or max_pixel < 50 or non_zero_count < 50:  # Stricter threshold
            print(f"Segment {i} skipped (too faint: mean={mean_pixel:.2f}, max={max_pixel:.2f}, non-zero={non_zero_count})")
            continue
        segments.append(segment)
    return segments

# Convert prediction index to digit
def index_to_char(idx):
    if 0 <= idx <= 9:
        return str(idx)
    return "?"

# Drawing interface
async def main():
    pygame.init()
    screen = pygame.display.set_mode((560, 560))
    canvas = pygame.Surface((56, 56))
    pygame.display.set_caption("Draw a Digit")
    canvas.fill((0, 0, 0))
    drawing = False
    model = load_model()
    font = pygame.font.SysFont('arial', 20)
    prediction_text = ""
    mode = "single"
    modes = ["single", "double", "triple"]
    mode_index = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                if mode == "single":
                    resized_canvas = pygame.transform.scale(canvas, (28, 28))
                    img_tensor = preprocess_image(resized_canvas)
                    with torch.no_grad():
                        output = model(img_tensor)
                        probs = torch.softmax(output, dim=1)[0]
                        pred = output.argmax(dim=1).item()
                        top2_probs, top2_indices = torch.topk(probs, 2)
                        print(f"Single mode predicted: {index_to_char(pred)} (confidence: {top2_probs[0]:.2f}, "
                              f"second: {index_to_char(top2_indices[1].item())} with {top2_probs[1]:.2f})")
                        prediction_text = f"Predicted: {index_to_char(pred)}"
                else:
                    num_segments = 2 if mode == "double" else 3
                    segments = segment_canvas(canvas, num_segments)
                    predictions = []
                    for i, segment in enumerate(segments):
                        img_tensor = preprocess_image(segment)
                        with torch.no_grad():
                            output = model(img_tensor)
                            probs = torch.softmax(output, dim=1)[0]
                            pred = output.argmax(dim=1).item()
                            top2_probs, top2_indices = torch.topk(probs, 2)
                            print(f"Segment {i} predicted: {index_to_char(pred)} (confidence: {top2_probs[0]:.2f}, "
                                  f"second: {index_to_char(top2_indices[1].item())} with {top2_probs[1]:.2f})")
                            predictions.append(str(pred))
                    if predictions:
                        prediction_text = f"Predicted: {''.join(predictions)}"
                    else:
                        prediction_text = "Predicted: None (Draw more digits)"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    canvas.fill((0, 0, 0))
                    prediction_text = ""
                if event.key == pygame.K_q:
                    pygame.quit()
                    return
                if event.key == pygame.K_m:
                    mode_index = (mode_index + 1) % len(modes)
                    mode = modes[mode_index]
                    canvas.fill((0, 0, 0))
                    prediction_text = f"Mode: {mode}"

        if drawing:
            pos = pygame.mouse.get_pos()
            x, y = pos[0] // 10, pos[1] // 10
            if 0 <= x < 56 and 0 <= y < 56:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if 0 <= x + dx < 56 and 0 <= y + dy < 56:
                            canvas.set_at((x + dx, y + dy), (255, 255, 255))

        # Scale canvas to screen
        scaled_canvas = pygame.transform.scale(canvas, (560, 560))
        screen.blit(scaled_canvas, (0, 0))

        # Draw segment boundaries
        if mode != "single":
            num_segments = 2 if mode == "double" else 3
            for i in range(1, num_segments):
                x = i * (560 // num_segments)
                pygame.draw.line(screen, (100, 100, 100), (x, 0), (x, 560), 1)

        # Display prediction and mode
        if prediction_text:
            text_surface = font.render(prediction_text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))
        mode_surface = font.render(f"Mode: {mode}", True, (255, 255, 255))
        screen.blit(mode_surface, (10, 540))

        pygame.display.flip()
        await asyncio.sleep(1.0 / 60)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
