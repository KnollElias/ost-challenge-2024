import matplotlib.pyplot as plt
from typing import List, Optional
from PIL import Image

def plot_images(images: List[Image.Image], max: int = 9, image_titles: Optional[List[str]] = None,
                figure_title: Optional[str] = None) -> plt.Figure:
    images_to_plot = images[:max]

    num_images = len(images_to_plot)
    rows = (num_images + 2) // 3
    cols = min(3, num_images)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    # Plot images
    for i, img in enumerate(images_to_plot):
        axes[i].imshow(img)
        axes[i].axis("off")
        if image_titles and i < len(image_titles):
            axes[i].set_title(image_titles[i])  # sewt title if availabl

    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    if figure_title:
        fig.suptitle(figure_title)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # adj layout to maek spavce for the title
    return fig

# Test

# Arrange
images = [Image.new('RGB', (200, 200), c) for c in ('Blue', 'Green', 'Red', 'Yellow', 'Black', 'Purple', 'Orange', 'Pink', 'Brown', 'Grey')]

# Test
# assert return is Figure
assert type(plot_images(images)) == plt.Figure, "Die Funktion sollte eine Figure zurückgeben"
assert len(plot_images(images).axes) == 9, "Die Funktion sollte per default maximal 9 Bilder anzeigen"
# assert number of images is correct
assert len(plot_images(images, max=6).axes) == 6, "Die Funktion sollte maximal 6 Bilder anzeigen mit max=6"
# assert number of images is correct
assert len(plot_images(images[:3]).axes) == 3, "Die Funktion sollte maximal 3 Bilder anzeigen wenn nur 3 mitgegeben werden (und max höher ist)"
# assert title is set
assert plot_images(images[:5], image_titles=[f"Bild {n+1}" for n in range(5)]).axes[0].get_title() == 'Bild 1', "Die Funktion sollte die Bilder-Überschrift korrekt setzen"
# assert title is set
assert plot_images(images[:5], image_titles=[f"Bild {n+1}" for n in range(5)], figure_title="Beispiel Grafik")._suptitle.get_text() == "Beispiel Grafik", "Die Funktion sollte den Haupt-Titel korrekt setzen"

# plt.close('all')

# Dev test
# images = [Image.new("RGB", (200, 200), color) for color in ["blue", "green", "red", "yellow", "black", "purple", "orange", "pink", "brown"]]
# fig = plot_images(images, max=6, image_titles=[f"Bildly {i+1}" for i in range(6)], figure_title="Wie seht das denne us?")
# plt.show()
