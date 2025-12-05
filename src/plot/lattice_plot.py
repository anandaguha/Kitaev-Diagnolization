import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from matplotlib.colors import ListedColormap

def plot_kitaev_patterns_hex(nx, ny):
    patterns = kitaevList(nx, ny)

    # Define two colors: 0 = yellow, 1 = blue
    cmap = ListedColormap(["yellow", "blue"])

    fig, axes = plt.subplots(2, 7, figsize=(22, 8))  # 14 phases
    axes = axes.ravel()

    # Hex geometry
    radius = 0.5
    dx = 3/2 * radius       # horizontal spacing
    dy = np.sqrt(3) * radius  # vertical spacing

    for phase, pattern in enumerate(patterns):
        arr = np.array(pattern).reshape(ny, nx)

        ax = axes[phase]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Phase {phase+1}")

        for row in range(ny):
            for col in range(nx):
                # offset every other row
                x = col * dx
                if row % 2 == 1:
                    x += dx / 2
                y = row * dy 

                color = cmap(arr[row, col])
                hexagon = RegularPolygon(
                    (x, y), numVertices=6, radius=radius,
                    orientation=np.radians(30),
                    facecolor=color, edgecolor="k"
                )
                ax.add_patch(hexagon)

        ax.set_xlim(-1, nx*dx + 1)
        ax.set_ylim(-1, ny*dy*0.5 + 1)

    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
from matplotlib.colors import ListedColormap

def plot_kitaev_patterns_hex_1(nx, ny):
    patterns = kitaevList(nx, ny)

    # Custom 2-color map: 0 = yellow, 1 = blue
    cmap = ListedColormap(["yellow", "blue"])

    fig, axes = plt.subplots(2, 7, figsize=(22, 8))  # 14 phases
    axes = axes.ravel()

    # Hexagon geometry
    dx = 1.0          # horizontal spacing
    dy = np.sqrt(3)/2 # vertical spacing (for hex tiling)

    for phase, pattern in enumerate(patterns):
        arr = np.array(pattern).reshape(ny, nx)

        ax = axes[phase]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Phase {phase+1}")

        # Plot hexagons
        for row in range(ny):
            for col in range(nx):
                x = col + 0.5 * (row % 2)  # stagger every other row
                y = row * dy
                color = cmap(arr[row, col])
                hexagon = RegularPolygon(
                    (x, y), numVertices=6, radius=0.5, orientation=np.radians(30),
                    facecolor=color, edgecolor="k"
                )
                ax.add_patch(hexagon)

        ax.set_xlim(-1, nx+1)
        ax.set_ylim(-1, ny*dy+1)

    plt.tight_layout()
    plt.show()
