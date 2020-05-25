import cv2
import numpy as np
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d


def list_available_cameras():
    """
    Searches for available webcams and returns a list of the ones that are found.

    Returns
    -------
    list
        Indices of available webcams that can be used for VideoCapture().
    """
    camera_index = 0
    camera_list = []
    while True:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened() is False:
            break
        else:
            camera_list.append(camera_index)
        cap.release()
        cv2.destroyAllWindows()
        camera_index += 1
    return camera_list


def initiate_camera(camera_index):
    """
    Starts a VideoCapture object for streaming or recording.

    Parameters
    ----------
    camera_index : int
        Index of the camera to initiate.

    Returns
    -------
    VideoCapture
        cv2.VideoCapture object
    """
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if camera.isOpened() is False:
        print('Camera unable to be opened.')
        # TODO Change this to a message box
    width = 1280
    height = 960
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def apply_mask(frame, lower, upper, kernel):
    """
    Masks video frames by the selected color range. Frame must be in native OpenCV color space.

    Parameters
    ----------
    frame : (N, M) array
        Array representing a single video frame in BGR color space.
    lower : (N) array of length 3
        Array representing the lower HSV values for masking.
    upper : (N) array of length 3
        Array representing the upper HSV values for masking.
    kernel : cv2 StructuringElement
        Smoothing kernel that is used for closing the mask.

    Returns
    -------
    (N, M) array
        Video frame that is masked by the currently selected colors.

    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower, upper)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_reps(y, threshold, open_size, close_size):
    """
    From the Y profile of a barbell's path, determine the concentric phase of each rep.

    The algorithm is as follows:
        1. Compute the gradient (dy/dt) of the Y motion
        2. Binarize the gradient signal by a minimum threshold value to eliminate noise.
        3. Perform 1D opening by open_size using a minimum then maximum filter in series.
        4. Perform 1D closing by close_size using a maximum then minimum filter in series.

    The result is a step function that is true for every time point that the concentric (+Y) phase of the rep
    is being performed.

    Parameters
    ----------
    y : (N) array
        Y component of the motion of the barbell path.
    threshold : float
        Miniumum acceptable value of the gradient (dY/dt) to indicate a rep.
        Increasing this can help eliminate noise, but may cause a small delay after a rep begins to when it is
        counted, therefore underestimating the time to complete a rep.
    open_size : int
        Minimum threshold of length of time that it takes to complete a rep (in frames).
        Increase this if there are false positive spikes in the rep step signal that are small in width.
    close_size : int
        Minimum length of time that could be between reps.
        Increase this if there are false breaks between reps that should be continuous.

    Returns
    -------
    (N) array
        Step signal representing when reps are performed. (1 indicates concentric phase of rep, 0 indicates no rep).
    """
    ygrad = np.gradient(y)
    rep_signal = np.where(ygrad > threshold, 1, 0)

    # Opening to remove spikes
    rep_signal = maximum_filter1d(minimum_filter1d(rep_signal, open_size), open_size)

    # Closing to connect movements (as in the step up from the jerk)
    rep_signal = minimum_filter1d(maximum_filter1d(rep_signal, close_size), close_size)

    return rep_signal
