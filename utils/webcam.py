import cv2


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
