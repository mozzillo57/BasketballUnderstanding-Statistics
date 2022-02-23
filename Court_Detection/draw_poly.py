import cv2
import numpy as np

def draw_poly_box(frame, pts, color=[0, 255, 0]):
    """Draw polylines bounding box.

    Parameters
    ----------
    frame : OpenCV Mat
        A given frame with an object
    pts : numpy array
        consists of bounding box information with size (n points, 2)
    color : list
        color of the bounding box, the default is green

    Returns
    -------
    new_frame : OpenCV Mat
        A frame with given bounding box.
    """
    new_frame = frame.copy()
    temp_pts = np.array(pts, np.int32)
    temp_pts = temp_pts.reshape((-1, 1, 2))
    cv2.polylines(new_frame, [temp_pts], True, color, thickness=2)

    return new_frame