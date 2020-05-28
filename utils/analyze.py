import time
import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.ndimage import label
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d


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


def analyze_set(t, x, y, r, diameter):
    """
    Calculates statistics of a set from a video recording and barbell tracking arrays.

    Parameters
    ----------
    t : (N) array
        Time in seconds from the beginning of the recording that each frame is taken.
    x : (N) array
        X location of the barbell in pixels.
    y : (N) array
        Y location of the barbell in pixels.
    r : (N) array
        Radius of the barbell marker in pixels measured at each time step.
    diameter : float
        Nominal diameter of the marker in mm, taken from the GIU.

    Returns
    -------
    List
        Index 0: DataFrame
            A new DataFrame with updated information for more advanced analytics.
        Index 1: Float
            Pixel calibration factor used to convert pixels to meters.
    """

    # Need to zero out bullshit numbers like 3.434236345E-120 or some shit
    t[0] = 0
    t[np.abs(t) < 0.0000001] = 0
    x[np.abs(x) < 0.0000001] = 0
    y[np.abs(y) < 0.0000001] = 0
    r[np.abs(r) < 0.0000001] = 0

    # Need some info from the UI
    nominal_radius = diameter / 2 / 1000  # meters
    # Calculate meter/pixel calibration from beginning of log when barbell isn't moving
    # Take it from frame 5 - 20 to allow time for video to "start up" (if that's even a thing?)
    calibration = nominal_radius / np.median(r[5:20])

    # Smooth motion to remove outliers
    xsmooth = medfilt(x, 7)
    ysmooth = medfilt(y, 7)

    # Calibrate x and y movement
    xcal = (xsmooth - np.min(xsmooth)) * calibration  # meters
    ycal = (ysmooth - np.min(ysmooth)) * calibration  # meters

    # Calculate displacement and velocity
    displacement = np.sqrt(np.diff(xcal, prepend=xcal[0]) ** 2 + np.diff(ycal, prepend=ycal[0]) ** 2)  # m
    velocity = np.zeros(shape=t.shape, dtype=np.float32)  # m/s
    velocity[1:] = displacement[1:] / np.diff(t)

    # Find the reps and label them
    reps_binary = find_reps(ycal, threshold=0.01, open_size=5, close_size=9)

    set_analyzed = pd.DataFrame()
    set_analyzed['Time'] = t
    set_analyzed['X_pix'] = x
    set_analyzed['Y_pix'] = y
    set_analyzed['R_pix'] = r
    set_analyzed['X_m'] = xcal
    set_analyzed['Y_m'] = ycal
    set_analyzed['Displacement'] = displacement
    set_analyzed['Velocity'] = velocity
    set_analyzed['Reps'] = reps_binary

    return set_analyzed, calibration


def analyze_reps(set_data, set_stats, movement):
    """
    Given an analyzed set log, calculate metrics for each rep that is found for updating the table and plots.

    Parameters
    ----------
    set_data : DataFrame
        Data collected and analyzed from logging the set. Expected columns are Time, Velocity, X_m, Y_m, and Reps.
    set_stats : Dictionary
        Metadata for the set. The only expected keys is weight, but number_of_reps is added and returned.
    movement : string
        Name of exercise that must correspond to a key in the lifts dictionary in lifts.json.

    Returns
    -------
    list of dictionaries
        Index 0: set_stats dictionary updated with number of reps
        Index 1: rep_stats dictionary with metrics measured for each rep
    """
    reps_labeled, n_reps = label(set_data['Reps'].values)
    set_stats['number_of_reps'] = n_reps
    velocity = set_data['Velocity'].values
    xcal = set_data['X_m'].values
    ycal = set_data['Y_m'].values
    rep_stats = {}
    for rep in range(1, n_reps + 1):
        idx = tuple([reps_labeled == rep])
        rep_stats[f"rep{rep}"] = {}
        rep_stats[f"rep{rep}"]['rep_id'] = f"{set_stats['set_id']}_{rep}"
        rep_stats[f"rep{rep}"]['exercise'] = movement
        rep_stats[f"rep{rep}"]['average_velocity'] = np.average(velocity[idx])
        rep_stats[f"rep{rep}"]['peak_velocity'] = np.max(velocity[idx])
        rep_stats[f"rep{rep}"]['peak_power'] = set_stats['weight'] * 9.80665 * rep_stats[f"rep{rep}"]['peak_velocity']
        rep_stats[f"rep{rep}"]['height_when_peaked'] = ycal[idx][np.argmax(velocity[idx])]
        rep_stats[f"rep{rep}"]['x_rom'] = np.max(xcal[idx]) - np.min(xcal[idx])
        rep_stats[f"rep{rep}"]['y_rom'] = np.max(ycal[idx]) - np.min(ycal[idx])
        rep_stats[f"rep{rep}"]['concentric_time'] = set_data['Time'].values[idx][-1] - set_data['Time'].values[idx][0]

    return set_stats, rep_stats


def post_process_video(video_file, n_frames, set_data):
    """
    Opens a video file, traces the bar path, then saves it back to disk with the correct framerate.

    Parameters
    ----------
    video_file : string
        Path to the video file.
    n_frames : int
        Number of frames of the video.
    set_data : DataFrame
        Full DataFrame obtained from analyzing the set containing (at minimum) the t, x, y coordinates of the bar.
    """

    # Get smoothed motion to remove outliers and make path nicer
    xsmooth = medfilt(set_data['X_pix'].values, 7)
    ysmooth = medfilt(set_data['Y_pix'].values, 7)

    # Open video stream
    cap = cv2.VideoCapture(video_file)
    if cap.isOpened() is False:
        print('Camera unable to be opened.')
        # TODO Change this to a message box
    time.sleep(1)
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Estimate correct fps and save to that
    fps = int(n_frames / set_data['Time'].values[-1])
    video_out = cv2.VideoWriter(video_file.replace('.mp4', '_traced.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        if current_frame > 6:
            try:
                for cf in range(5, current_frame-1):
                    xy1 = (int(xsmooth[cf-1]), int(ysmooth[cf-1]))
                    xy2 = (int(xsmooth[cf]), int(ysmooth[cf]))
                    cv2.line(frame, xy1, xy2, (255, 255, 255), 1)
                    # TODO: Plot max velocities in a different color or as a different size
                    # TODO Fix indexing to avoid dotted line look
            except IndexError:
                pass
        video_out.write(frame)
        current_frame += 1
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
