from pathlib import Path
from PIL import Image
import json
import lightning as L
import torch
import numpy as np
import matplotlib.pyplot as plt

# Prüfen, ob ein GPU (cuda) verfügbar ist
# print(f"Is GPU (Cuda) available: {torch.cuda.is_available()}")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(torch.cuda.get_device_name(device=device))

MODEL_DIRECTORY_PATH = Path('C://Users/lace/AI-Challenge/Data2')

# Sollte der Pfad nicht existieren, wird hier eine Fehlermeldung ausgegeben
if not MODEL_DIRECTORY_PATH.exists():
    raise Exception(f"Fehler🛑 Der Pfad {MODEL_DIRECTORY_PATH} existiert nicht")
else:
    print(f"Alles gut👍 der Pfad {MODEL_DIRECTORY_PATH} wurde gefunden")

def get_image_paths(path: Path) -> list:
    return list(sorted(path.glob('**/*.jpg')))

def read_data_from_json(image_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        angle = data.get("angle")
        speed = data.get("speed")
        return angle, speed
    pass


# alle Bildpfade laden
image_paths = get_image_paths(MODEL_DIRECTORY_PATH)
assert len(image_paths) > 0, "Keine Bilder gefunden"
print(f"Es wurden {len(image_paths)} Bilder gefunden")

# der erste Pfad auswählen und ausgeben (wenn du einen anderen willst, ändere den Index)
some_image_path = image_paths[42]
print(some_image_path)

# Bild laden und anzeigen
img = Image.open(some_image_path)
print(f"Größe des Bildes (BxH): {img.size}")

# Dazugehörige Daten für Lenkung und Geschwindigkeit ausgeben
angle, speed = read_data_from_json(some_image_path)
print(f"Geschwindigkeit {speed:.2f}, Lenkung: {angle:.2f}")

from multiprocessing.pool import ThreadPool
with ThreadPool(20) as pool:
  vehicle_data = np.array(pool.map(read_data_from_json, image_paths))
all_angles = vehicle_data[:,0]
all_speeds = vehicle_data[:,1]

print(f"Kennzahlen zu Lenkung: min={all_angles.min():.2f}, max: {all_angles.max():.2f}, Durchschnitt: {all_angles.mean():.2f}")
print(f"Kennzahlen zu Geschwindigkeit: min={all_speeds.min():.2f}, max: {all_speeds.max():.2f}, Durchschnitt: {all_speeds.mean():.2f}")

# Diagramm zeichnen
import matplotlib.style as mplstyle
mplstyle.use('fast')
fig, (ax1, ax2) = plt.subplots(2,figsize=(10, 8))
ax1.plot(all_angles)
ax1.set_title("Angle")
ax1.minorticks_on()
ax2.plot(all_speeds)
ax2.set_title("Speed")
ax2.minorticks_on()

def read_data_from_json(image_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        angle = data.get("angle")
        speed = data.get("speed")
        return angle, speed
    pass

from torchvision.transforms import v2

transformer = v2.Compose(
    [
        v2.ColorJitter(
            brightness=0.5,
            saturation=0.2,
            hue=0.1,
        ),
    ]
)

from torchvision.transforms import v2

transformer = v2.Compose(
    [
        v2.ColorJitter(
            brightness=0.5,
            saturation=0.2,
            hue=0.1,
        ),
    ]
)

fig, axs = plt.subplots(1, 4, figsize=(3*4, 3))
for i, ax in enumerate(axs):
    # Zuerst das Originalbild
    if i == 0:
        ax.imshow(img)
        ax.set_title("Original")
        ax.axis('off')
        continue
    # Danach 3 zusätzlich augmentierte Bilder
    else:
        augmented_img = transformer(img)
        ax.imshow(augmented_img)
        ax.set_title(f"Augmented {i}")
        ax.axis('off')

from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        transform=None, # Optional: Ein Transform für die Augmentation
    ):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        # Wir werden das Bild etwas verkleinern, um Rechenzeit zu sparen (es sind immer noch genügend Informationen im Bild vorhanden)
        # Original ist das Bild BxH 320x240 Pixel gross, wir verkleinern es auf 16x120 Pixel
        image = image.resize((160, 120))
        # Augmentation (wenn vorhanden)
        if self.transform:
            image = self.transform(image)
        image_tensor = pil_to_tensor(image)
        image_tensor = image_tensor / 255.0  # Skaling
        angle, _ = read_data_from_json(path)
        angle_tensor = torch.tensor(angle, dtype=torch.float32).unsqueeze(0)
        return image_tensor, angle_tensor
from torch.utils.data import DataLoader
import random

dataset = ImageDataset(image_paths)

# Wir werden 80% der Daten für das Training verwenden und 20% für das Validieren
random.shuffle(image_paths)
split_idx = int(len(image_paths) * 0.8)
train_image_paths = image_paths[:split_idx]
val_image_paths = image_paths[split_idx:]

# Wichtig: Augmentation nur für das Trainingsset aktivieren!
train_dataset = ImageDataset(train_image_paths, transform=transformer)
val_dataset = ImageDataset(val_image_paths, transform=None)

# Überprüfen wir mal die Grösse dieser Sets
print(len(train_dataset), len(val_dataset))

# Inhalt anschauen
image, angle = train_dataset[0]
print(image.shape, image.dtype, angle, angle.dtype)

# Erstellen der DataLoaders
BATCH_SIZE = 128
training_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
validation_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# Ein Batch aus dem DataLoader laden und dessen Shape anzeigen
images, labels = next(iter(training_loader))
print(f"Input shape: {images.shape}")
print(f"Output shape {labels.shape}")

from lightning import LightningModule, Trainer

class DenseModel(LightningModule):
    # NEU Die Lernrate geben wir als Parameter mit, damit wir sie später einfach anpassen können
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr
        self.logged_metrics = {
            "train_loss": [],
            "val_loss": [],
        }
        
        self.dropout = 0.1
        # convolution layers initialisieren
        self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = torch.nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        # hilf/optimierungslayer initialisieren
        self.batch1 = torch.nn.BatchNorm2d(24)
        self.batch2 = torch.nn.BatchNorm2d(32)
        self.batch3 = torch.nn.BatchNorm2d(64)
        self.batch4 = torch.nn.BatchNorm2d(64)
        self.batch5 = torch.nn.BatchNorm2d(64)
        self.batch_l1 = torch.nn.BatchNorm1d(100)
        self.batch_l2 = torch.nn.BatchNorm1d(50)

        self.drop = torch.nn.Dropout(self.dropout)
        self.relu = torch.nn.ReLU()

        # nachgeschaltetes klassisches Netzwerk initialisieren
        self.flatten = torch.nn.Flatten(1, -1)
        self.linear1 = torch.nn.Linear(6656, 100)
        self.linear2 = torch.nn.Linear(100, 50)
        self.output = torch.nn.Linear(50, 1)

    def forward(self, x):
        # convolution layer 1
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.drop(x)

        # convolution layer 2
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.drop(x)

        # convolution layer 3
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu(x)
        x = self.drop(x)

        # convolution layer 4
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu(x)
        x = self.drop(x)

        # convolution layer 5
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu(x)
        x = self.drop(x)

        # nachggeschalteter teil
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.batch_l1(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.linear2(x)
        x = self.batch_l2(x)
        x = self.relu(x)
        x = self.drop(x)

        # Regression, daher nur ein Output Neuron und ohne Aktivierungsfunktion am Schluss
        angle = self.output(x)
        return angle

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        # Regression, also wollen wir den Mean Squared Error
        loss = torch.nn.functional.mse_loss(y_pred, y)
        self.logged_metrics["train_loss"].append(loss.item())
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        # Regression, also wollen wir den Mean Squared Error
        loss = torch.nn.functional.mse_loss(y_pred, y)
        self.logged_metrics["val_loss"].append(loss.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# Wir können uns ansehen, wie die Architektur des Modells aussieht.
# Hier sieht man die einzelnen Schichten, welche Ausgangssignale (Output Shape) sie erzeugen, sowie die Anzahl künstlicher Neuronen (Param #), welche sie enthalten.

from lightning.pytorch.callbacks import ModelSummary
ModelSummary().on_fit_start(Trainer(), DenseModel())

# Anzahl Epochen definieren
EPOCHS = 20

# Trainer initialisieren.
trainer = Trainer(max_epochs=EPOCHS)

# Modell initialisieren
LEARNING_RATE = 0.01
model = DenseModel(lr=LEARNING_RATE)

# Nun können wir uns die Trainings- und Validierungsfehler anschauen
train_loss = model.logged_metrics["train_loss"]
val_loss = model.logged_metrics["val_loss"]
plt.plot(train_loss, label="train_loss")
plt.plot(
    [int(i * (len(train_loss) / len(val_loss))) for i in range(len(val_loss))],
    val_loss,
    label="val_loss",
)
# Die X-Achse mit Epochs beschriften. Jedoch max 10 Beschriftungen
plt.xticks(
    ticks=[i * (len(train_loss) // 10) for i in range(10)],
    labels=[f"{i*(EPOCHS//10)}" for i in range(10)],
);
plt.xlabel("Epoch")
plt.ylabel("Loss (Fehler)")
plt.legend()

# Logarithmische Skala für die Y-Achse, da die Werte sehr gross sind
# plt.yscale("log")

# Unsere Vorhersage-Funktion

def predict(img_tensor):
    model.eval()
    with torch.no_grad():
        y_pred = model(img_tensor.unsqueeze(0))
    # Rückgabe ist das das Label, sowie die Wahrscheinlichkeiten für jedes Label
    return y_pred.item()

# Testen wir unsere Funktion
image_tensor, _ = dataset[0]
angle_pred = predict(image_tensor)
print(f"Predicted angle: {angle_pred}")

# Wir schauen uns ein paar zufällige Bilder an und vergleichen die tatsächlichen Werte mit den vorhergesagten Werten
import random

# Wir wählen zufällige Bilder aus
random_indices = random.sample(range(len(val_dataset)), 6)
true_y = [angle.item() for _, angle in [val_dataset[i] for i in random_indices]]

# Wir berechnen die vorhergesagten Werte
pred_y = [predict(image_tensor) for image_tensor, _ in [val_dataset[i] for i in random_indices]]

# Wir zeigen die Bilder und die vorhergesagten Werte an
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for i, ax in enumerate(axs.flat):
    image, _ = val_dataset[random_indices[i]]
    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(f"True: {true_y[i]:.1f} / Pred: {pred_y[i]:.1f} / Fehler {true_y[i] - pred_y[i]:.1f}")
    ax.axis("off")

# Wir lassen uns das Validierungs-Set vorhersagen und extrahieren die Werte

true_y = [angle.item() for _, angle in val_dataset]
pred_y = [predict(image_tensor) for image_tensor, _ in val_dataset]

# Wir können den gesamten Fehler berechnen
total_error = sum([abs(true - pred) for true, pred in zip(true_y, pred_y)])/len(true_y)
print(f"Der durchschnittliche Fehler auf dem Validierung-Set ist {total_error:.1f}")

# Visuelle Inspektion der Vorhersagen (max. 100 Werte)
# Da das Validierungs-Set zufällig ist, sortieren wir diese der Grösse nach
# Wenn das Model gut ist, sollten die Linien grösstenteils übereinander liegen

n = min(100, len(true_y))
true_y_plot = true_y[:n]
pred_y_plot = pred_y[:n]
sorted_idx = np.argsort(true_y_plot)
true_y_plot = np.array(true_y_plot)[sorted_idx]
pred_y_plot = np.array(pred_y_plot)[sorted_idx]

plt.plot(true_y_plot, label="True")
plt.plot(pred_y_plot, label="Predicted")
plt.legend()
plt.ylabel("Angle")
plt.title("Visuelle Inspektion")

# Der Name unseres Modells
MODEL_NAME = "model_03c_Fahrmodell_Training.onnx"

# Zur Sicherheit prüfen, ob das Model bereits existiert
# (Lösche es, wenn du es überschreiben willst. Oder benenne es um)
assert not Path(MODEL_DIRECTORY_PATH / MODEL_NAME).exists(), f"Das Model {MODEL_NAME} existiert bereits"

# Model speichern
input_example = train_dataset[0][0].unsqueeze(0)
model.to_onnx(MODEL_DIRECTORY_PATH / MODEL_NAME, input_sample=input_example)



brightness 0.5
0.3
saturation 0.3
hue 0.07


maybe flipp the iamges?
 -> also flip the angle -> maybe offset it

 60k Bilder is a lot

 Eher mehr bilder und dann nicht alle answerten, trianingsdaten machen extrem viel au ss

Only use lower 50% for pathfinding (not for stop signs)

amplify steering -> when it is  to sensitive

