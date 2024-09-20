import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_network_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 12)

    # Draw input layer
    rect = patches.FancyBboxPatch((0.5, 8), 1.5, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightblue')
    ax.add_patch(rect)
    ax.text(1.25, 8.5, 'Input\n28x28x1', ha='center', va='center', fontsize=12)

    # Draw Conv Layer 1
    rect = patches.FancyBboxPatch((3, 8), 2, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightgreen')
    ax.add_patch(rect)
    ax.text(4, 8.5, 'Conv1\n28x28x32', ha='center', va='center', fontsize=12)

    # Draw Pooling Layer 1
    rect = patches.FancyBboxPatch((6, 8), 1.5, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightcoral')
    ax.add_patch(rect)
    ax.text(6.75, 8.5, 'Pool1\n14x14x32', ha='center', va='center', fontsize=12)

    # Draw Conv Layer 2
    rect = patches.FancyBboxPatch((8, 8), 2, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightgreen')
    ax.add_patch(rect)
    ax.text(9, 8.5, 'Conv2\n14x14x64', ha='center', va='center', fontsize=12)

    # Draw Pooling Layer 2
    rect = patches.FancyBboxPatch((11, 8), 1.5, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightcoral')
    ax.add_patch(rect)
    ax.text(11.75, 8.5, 'Pool2\n7x7x64', ha='center', va='center', fontsize=12)

    # Draw Flatten Layer
    rect = patches.FancyBboxPatch((2.5, 5), 2, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightgrey')
    ax.add_patch(rect)
    ax.text(3.5, 5.5, 'Flatten\n3136', ha='center', va='center', fontsize=12)

    # Draw Fully Connected Layer 1
    rect = patches.FancyBboxPatch((6.5, 5), 2, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightyellow')
    ax.add_patch(rect)
    ax.text(7.5, 5.5, 'FC1\n128', ha='center', va='center', fontsize=12)

    # Draw Fully Connected Layer 2
    rect = patches.FancyBboxPatch((10.5, 5), 2, 1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightyellow')
    ax.add_patch(rect)
    ax.text(11.5, 5.5, 'FC2\n10', ha='center', va='center', fontsize=12)

    # Draw arrows
    def draw_arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(facecolor='black', shrink=0.05))

    draw_arrow((2, 8), (3, 8))
    draw_arrow((5, 8), (6, 8))
    draw_arrow((7.5, 8), (8, 8))
    draw_arrow((10, 8), (11, 8))
    draw_arrow((3.5, 5), (6.5, 5))
    draw_arrow((7.5, 5), (10.5, 5))

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Remove axes
    plt.axis('off')

    plt.show()




if __name__ == '__main__':
    draw_network_diagram();