import numpy as np
import matplotlib.pyplot as plt

def plot_robot(ax, frames, show=False, origin=False):
    """
    Plots a robot based on a list of homogeneous transformation matrices.

    Parameters:
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis to plot on.
    frames : list
        List of homogeneous transformation matrices (4x4).
    """
    if not all(frame.shape == (4, 4) for frame in frames):
        raise ValueError("All frames must be 4x4 homogeneous transformation matrices.")

    # Plot each frame
    for i, frame in enumerate(frames):
        plot_frame_a(ax, frame, str(i), show=False, show_origin=origin)

    # Plot the transition between frames
    for i in range(len(frames) - 1):
        plot_transicao_ab(ax, frames[i], frames[i + 1], show=False)
    
    _square_axes(ax)

    if show:
        plt.show()


def plot_frame_a(ax, FA, nome='A', show=False, show_origin=True):
    """
    Plots a 3D frame {A} based on its homogeneous transformation matrix.

    Parameters:
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis to plot on.
    FA : numpy.ndarray
        Homogeneous transformation matrix (4x4).
    nome : str
        Name of the frame to be displayed on the plot (default: 'A').
    """
    if FA.shape != (4, 4):
        raise ValueError("FA must be a 4x4 homogeneous transformation matrix.")

    # Origin of the frame
    origin = FA[:3, 3]
    
    # Axes directions
    x_axis = FA[:3, 0]  # x-axis direction
    y_axis = FA[:3, 1]  # y-axis direction
    z_axis = FA[:3, 2]  # z-axis direction

    # Plot the x-axis
    ax.plot(
        [origin[0], origin[0] + x_axis[0]],
        [origin[1], origin[1] + x_axis[1]],
        [origin[2], origin[2] + x_axis[2]],
        'b', linewidth=2
    )
    ax.text(
        origin[0] + x_axis[0],
        origin[1] + x_axis[1],
        origin[2] + x_axis[2],
        f"x_{{{nome}}}",
        color='b'
    )

    # Plot the y-axis
    ax.plot(
        [origin[0], origin[0] + y_axis[0]],
        [origin[1], origin[1] + y_axis[1]],
        [origin[2], origin[2] + y_axis[2]],
        'r', linewidth=2
    )
    ax.text(
        origin[0] + y_axis[0],
        origin[1] + y_axis[1],
        origin[2] + y_axis[2],
        f"y_{{{nome}}}",
        color='r'
    )

    # Plot the z-axis
    ax.plot(
        [origin[0], origin[0] + z_axis[0]],
        [origin[1], origin[1] + z_axis[1]],
        [origin[2], origin[2] + z_axis[2]],
        'g', linewidth=2
    )
    ax.text(
        origin[0] + z_axis[0],
        origin[1] + z_axis[1],
        origin[2] + z_axis[2],
        f"z_{{{nome}}}",
        color='g'
    )

    # Plot the origin
    if show_origin:    
        ax.scatter(origin[0], origin[1], origin[2], color='k', s=50)
        ax.text(origin[0], origin[1], origin[2], f"{{{nome}}}", color='k')
    if show:
        _square_axes(ax)
        plt.show()

def plot_transicao_ab(ax, FA, FB, show=False):
    """
    Plots the transition from frame {A} to frame {B}, connecting their origins.

    Parameters:
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis to plot on.
    FA : numpy.ndarray
        Homogeneous transformation matrix of frame {A} (4x4).
    FB : numpy.ndarray
        Homogeneous transformation matrix of frame {B} (4x4).
    """
    if FA.shape != (4, 4) or FB.shape != (4, 4):
        raise ValueError("Both FA and FB must be 4x4 homogeneous transformation matrices.")

    # Origins of frames {A} and {B}
    origin_a = FA[:3, 3]
    origin_b = FB[:3, 3]

    # Plot the line connecting the origins of frames {A} and {B}
    ax.plot(
        [origin_a[0], origin_b[0]],
        [origin_a[1], origin_b[1]],
        [origin_a[2], origin_b[2]],
        color=[240/256, 180/256, 100/256],
        linewidth=4
    )

    if show:
        _square_axes(ax)
        plt.show()

def _square_axes(ax):
    """
    Sets the aspect ratio of the 3D plot to be equal.

    Parameters:
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis to set the aspect ratio.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

if __name__ == '__main__':
    A = np.eye(4)
    B = np.eye(4) + np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_frame_a(ax, A, 'A')
    plot_frame_a(ax, B, 'B')
    plot_transicao_ab(ax, A, B, show=True)
    
