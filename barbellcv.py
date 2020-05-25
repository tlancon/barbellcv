# Standard library imports
import os
import sys
import time
import json
from collections import deque
# External library imports
import cv2
import qdarkstyle
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, uic
from scipy.signal import medfilt
from scipy.ndimage import label
# Custom imports
import utilities

pg.setConfigOption('background', '#19232D')
pg.setConfigOptions(antialias=True)

qtCreatorFile = os.path.abspath('resources/barbellcv_log.ui')
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class KiloCountLogApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # ALL data is saved to the data directory for now - this needs to exist
        if os.path.isdir('./data/') is False:
            os.mkdir('./data/')
        if os.path.isdir(f"./data/{time.strftime('%y%m%d')}") is False:
            os.mkdir(f"./data/{time.strftime('%y%m%d')}")
        self.data_dir = os.path.dirname(f"./data/{time.strftime('%y%m%d')}/")

        # Connect signals
        self.buttonPreview.clicked.connect(self.preview_camera)
        self.buttonSelectColor.clicked.connect(self.select_colors)
        self.spinLbs.editingFinished.connect(self.lbs_changed)
        self.spinKgs.editingFinished.connect(self.kgs_changed)
        self.buttonLogSet.clicked.connect(self.log_set)

        # Set up camera options
        # Find available cameras
        self.camera_list = utilities.list_available_cameras()
        for c in self.camera_list:
            self.comboCamera.addItem(str(c))
        # Set up rotation
        for r in ['0', '90', '180', '270']:
            self.comboRotation.addItem(r + u"\u00b0")

        # Load parameters from previous session if present
        if os.path.isfile('./resources/settings.json') is True:
            self.load_settings()

        # Load lifts to dropdown
        dlf = open('./resources/lifts.json', 'r')
        self.lifts = json.load(dlf)
        for lift in self.lifts:
            self.comboExercise.addItem(self.lifts[lift]['name'])
        dlf.close()

        # Set up table for display
        table_headers = ['Avg Vel (m/s)', 'Pk Vel (m/s)', 'Pk Power (W)', 'Y at Pk (m)',
                         'X ROM (m)', 'Y ROM (m)', 'Conc. Time (s)']
        self.tableSetStats.setRowCount(len(table_headers))
        self.tableSetStats.setVerticalHeaderLabels(table_headers)
        self.tableSetStats.verticalHeader().setDefaultSectionSize(40)
        self.tableSetStats.setColumnCount(6)

        # Set up plots for display
        # Create empty plot for y and velocity
        self.plotTimeline.clear()
        self.t1 = self.plotTimeline.plotItem
        self.t1.setLabel('bottom', 'Time (s)', **{'color': '#FFFFFF'})
        self.t1.setLabel('left', 'Y (m)', **{'color': '#E4572E'})
        self.t1.setLabel('right', 'Velocity (m/s)', **{'color': '#17BEEB'})
        # Link X axis but keep y separate for y, velocity
        self.t2 = pg.ViewBox()
        self.t1.showAxis('right')
        self.t1.scene().addItem(self.t2)
        self.t1.getAxis('right').linkToView(self.t2)
        self.t2.setXLink(self.t1)
        self.update_timeline_view()
        self.t1.vb.sigResized.connect(self.update_timeline_view)
        # Create empty plot for barbell motion path
        self.plotMotion.clear()
        self.xy = self.plotMotion.plotItem
        self.xy.setLabel('bottom', 'X (m)', **{'color': '#76B041'})
        self.xy.setLabel('left', 'Y (m)', **{'color': '#76B041'})
        self.xy.setAspectLocked(lock=True, ratio=1)

        # Logic controls for button clicks
        self.selecting = False  # Whether color selection window is open
        self.selecting_active = False  # Whether mouse is currently selecting color
        self.tracking = False  # Whether barbell tracking is ongoing for a set
        self.cropping = False  # Whether the lifer is currently cropping

        # Globals that need sharing throughout the app
        self.output_dir = os.path.abspath(os.path.dirname('./data/'))
        self.mask_colors = deque()
        self.table_colors = ['#76B041', '#E4572E']  # [Good rep, bad rep]
        self.smoothing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Methods for initializing UI

    def load_settings(self):
        """
        Loads settings back from previous session.
        """
        settings_file = open('./resources/settings.json')
        settings = json.load(settings_file)
        settings_file.close()

        try:
            if settings['camera'] in self.camera_list:
                self.comboCamera.setCurrentIndex(settings['camera'])
            self.comboRotation.setCurrentIndex(settings['rotation'])
            self.spinMinHue.setValue(settings['colors']['min_hue'])
            self.spinMaxHue.setValue(settings['colors']['max_hue'])
            self.spinMinSaturation.setValue(settings['colors']['min_saturation'])
            self.spinMaxSaturation.setValue(settings['colors']['max_saturation'])
            self.spinMinValue.setValue(settings['colors']['min_value'])
            self.spinMaxValue.setValue(settings['colors']['max_value'])
            self.spinDiameter.setValue(settings['diameter'])
        except KeyError:
            print('Some settings not found in settings.json. Initializing with defaults instead.')

    def save_settings(self):
        """
        Saves settings for a future session.
        """
        settings = {'camera': self.comboCamera.currentIndex(), 'rotation': self.comboRotation.currentIndex(),
                    'colors': {
                        'min_hue': self.spinMinHue.value(),
                        'max_hue': self.spinMaxHue.value(),
                        'min_saturation': self.spinMinSaturation.value(),
                        'max_saturation': self.spinMaxSaturation.value(),
                        'min_value': self.spinMinValue.value(),
                        'max_value': self.spinMaxValue.value(),
                    }, 'diameter': self.spinDiameter.value()}
        settings_file = open('./resources/settings.json', 'w')
        json.dump(settings, settings_file, indent=4)
        settings_file.close()

    # Methods for adapting UI

    def lbs_changed(self):
        """
        Adapts kgs spinbox to a change in lbs.
        """
        kgs = round(self.spinLbs.value() * 0.453592, 0)
        self.spinKgs.setValue(kgs)

    def kgs_changed(self):
        """
        Adapts lbs spinbox to a change in kgs.
        """
        lbs = round(self.spinKgs.value() * 2.20462, 0)
        self.spinLbs.setValue(lbs)

    def preview_camera(self):
        """
        Launches a stream of the webcam to allow the lifter to see what framing, rotation, and cropping look like.
        """
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Previewing the camera. Press the Enter key to exit.')
        self.buttonPreview.setText('Press Enter\nto finish.')
        self.comboCamera.setEnabled(False)
        self.buttonSelectColor.setEnabled(False)
        self.buttonLogSet.setEnabled(False)
        # When implemented: self.buttonAnalyzeSet.setEnabled(False)
        cap = utilities.initiate_camera(self.comboCamera.currentIndex())
        while True:
            _, frame = cap.read()
            frame = np.rot90(frame, self.comboRotation.currentIndex())
            cv2.imshow('Camera Preview', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('\r'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.buttonPreview.setText('Preview')
        self.comboCamera.setEnabled(True)
        self.buttonSelectColor.setEnabled(True)
        self.buttonLogSet.setEnabled(True)
        self.statusbar.clearMessage()
        # When implemented: self.buttonAnalyzeSet.setEnabled(True)

    def select_colors(self):
        """
        Launches a stream of the webcam to allow the lifter to select the color of the barbell marker.
        """
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Select colors of the marker using the left mouse button. Right click to clear. '
                                   'Press the Enter key to confirm.')
        self.buttonSelectColor.setText('Press Enter\nto finish.')
        self.buttonPreview.setEnabled(False)
        self.buttonLogSet.setEnabled(False)
        # When implemented: self.buttonAnalyzeSet.setEnabled(False)
        self.selecting = True
        n_90_rotations = self.comboRotation.currentIndex()
        cap = utilities.initiate_camera(self.comboCamera.currentIndex())
        while True:
            _, frame = cap.read()
            frame = np.rot90(frame, n_90_rotations)
            cv2.imshow('Select Colors', frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.setMouseCallback('Select Colors', self.handle_color_selection, hsv)
            if len(self.mask_colors) != 0:
                self.spinMinHue.setValue(min(c[0] for c in self.mask_colors))
                self.spinMaxHue.setValue(max(c[0] for c in self.mask_colors))
                self.spinMinSaturation.setValue(min(c[1] for c in self.mask_colors))
                self.spinMaxSaturation.setValue(max(c[1] for c in self.mask_colors))
                self.spinMinValue.setValue(min(c[2] for c in self.mask_colors))
                self.spinMaxValue.setValue(max(c[2] for c in self.mask_colors))
                lower = np.array([self.spinMinHue.value(), self.spinMinSaturation.value(), self.spinMinValue.value()])
                upper = np.array([self.spinMaxHue.value(), self.spinMaxSaturation.value(), self.spinMaxValue.value()])
                masked = cv2.bitwise_and(frame, frame,
                                         mask=utilities.apply_mask(frame, lower, upper, self.smoothing_kernel))
                cv2.imshow('Masked', masked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('\r'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.selecting = False
        self.buttonPreview.setEnabled(True)
        self.buttonLogSet.setEnabled(True)
        # When implemented: self.buttonAnalyzeSet.setEnabled(True)
        self.buttonSelectColor.setText('Select Color')
        self.statusbar.clearMessage()

    def reset_colors(self):
        """
        Resets the selection colors back to their full ranges.
        """
        self.spinMinHue.setValue(0)
        self.spinMaxHue.setValue(180)
        self.spinMinSaturation.setValue(0)
        self.spinMaxSaturation.setValue(255)
        self.spinMinValue.setValue(0)
        self.spinMaxValue.setValue(255)

    def update_table(self, metadata):
        """
        Clear the table and update it with stats from the current set.

        Parameters
        ----------
        metadata : Dictionary
            Dictionary containing metadata from the current set, including all of the measures to be viewed in the
            table.
        """
        self.tableSetStats.setColumnCount(len(metadata.keys()))
        for r, rep in enumerate(metadata.keys()):
            self.tableSetStats.setItem(0, r, QtWidgets.QTableWidgetItem(f"{metadata[rep]['average_velocity']:.2f}"))
            self.tableSetStats.setItem(1, r, QtWidgets.QTableWidgetItem(f"{metadata[rep]['peak_velocity']:.2f}"))
            self.tableSetStats.setItem(2, r, QtWidgets.QTableWidgetItem(f"{metadata[rep]['peak_power']:.2f}"))
            self.tableSetStats.setItem(3, r, QtWidgets.QTableWidgetItem(f"{metadata[rep]['height_when_peaked']:.2f}"))
            self.tableSetStats.setItem(4, r, QtWidgets.QTableWidgetItem(f"{metadata[rep]['x_rom']:.2f}"))
            self.tableSetStats.setItem(5, r, QtWidgets.QTableWidgetItem(f"{metadata[rep]['y_rom']:.2f}"))
            self.tableSetStats.setItem(6, r, QtWidgets.QTableWidgetItem(f"{metadata[rep]['time_to_complete']:.2f}"))
            if metadata[rep]['peak_velocity'] >= 1.2:
                col_color = QtGui.QColor(self.table_colors[0])
            else:
                col_color = QtGui.QColor(self.table_colors[1])
            for i in range(self.tableSetStats.rowCount()):
                self.tableSetStats.item(i, r).setBackground(col_color)

    def log_set(self):
        """
        Records a video of a set for further analysis.

        Opens a stream to the webcam and rotates each frame appropriately, then displays the stream.

        Note that this saves to 30 fps regardless of the actual frame rate since cv2.VideoWriter requires a frame rate
        when it's instantiated.
        """
        # Adapt the UI
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Recording your set. Press the Enter key when you are finished.')
        self.buttonLogSet.setText('Press Enter\nto finish.')
        self.buttonPreview.setEnabled(False)
        self.buttonSelectColor.setEnabled(False)
        self.buttonAnalyzeSet.setEnabled(False)
        self.tracking = True
        # Prepare set metadata
        video_file, log_file, meta_file = self.build_filepaths()
        set_metadata = {}
        set_metadata['raw_video_file'] = video_file
        set_metadata['log_file'] = log_file
        set_metadata['exercise'] = self.comboExercise.currentText()
        set_metadata['weight'] = self.spinKgs.value()
        set_metadata['calibration_colors'] = list(self.mask_colors)
        set_metadata['nominal_diameter'] = self.spinDiameter.value()
        # Initialize
        n_90_rotations = self.comboRotation.currentIndex()
        n_frames = 0
        path_time = np.array([], dtype=np.float32)
        path_x = np.array([], dtype=np.float32)
        path_y = np.array([], dtype=np.float32)
        path_radii = np.array([], dtype=np.float32)
        # Camera setup
        cap = utilities.initiate_camera(self.comboCamera.currentIndex())
        time.sleep(2)
        if n_90_rotations in [0, 2]:
            width = int(cap.get(3))
            height = int(cap.get(4))
        else:
            width = int(cap.get(4))
            height = int(cap.get(3))
        video_out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        start_time = time.time()
        while True:
            _, frame = cap.read()
            frame = cv2.UMat(np.rot90(frame, n_90_rotations))
            lower = np.array([self.spinMinHue.value(), self.spinMinSaturation.value(), self.spinMinValue.value()])
            upper = np.array([self.spinMaxHue.value(), self.spinMaxSaturation.value(), self.spinMaxValue.value()])
            masked = utilities.apply_mask(frame, lower, upper, self.smoothing_kernel)
            contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Only track points if a contour is found
            if len(contours) != 0:
                largest = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(largest)
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 255), -1)
                path_time = np.append(path_time, (time.time() - start_time))
                path_x = np.append(path_x, x)
                path_y = np.append(path_y, y)
                path_radii = np.append(path_radii, radius)
            cv2.imshow('Tracking barbell...', frame)
            video_out.write(frame)
            n_frames += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('\r'):
                self.tracking = False
            if self.tracking is False:
                break
        # Release hold on camera and write video
        cap.release()
        video_out.release()
        cv2.destroyAllWindows()
        self.buttonLogSet.setText('Log Set')
        self.buttonPreview.setEnabled(True)
        self.buttonSelectColor.setEnabled(True)
        self.buttonAnalyzeSet.setEnabled(True)
        # Do the actual analysis
        # First, correct Y for video height since Y increases going DOWN
        path_y = height - path_y
        set_data = self.analyze_set(path_time, path_x, path_y, path_radii)
        set_data.to_csv(log_file)
        # Convert the video to the correct framerate and trace the bar path
        self.post_process_video(video_file, n_frames, set_data)
        # Compute stats for each rep
        reps_labeled, n_reps = label(set_data['Reps'].values)
        set_metadata['number_of_reps'] = n_reps
        velocity = set_data['Velocity'].values
        xcal = set_data['X_m'].values
        ycal = set_data['Y_m'].values
        rep_stats = {}
        for rep in range(1, n_reps + 1):
            indices = [reps_labeled == rep]
            rep_stats[f"rep{rep}"] = {}
            rep_stats[f"rep{rep}"]['average_velocity'] = np.average(velocity[indices])
            rep_stats[f"rep{rep}"]['peak_velocity'] = np.max(velocity[indices])
            rep_stats[f"rep{rep}"]['peak_power'] = self.spinKgs.value() * 9.80665 * rep_stats[f"rep{rep}"]['peak_velocity']
            rep_stats[f"rep{rep}"]['height_when_peaked'] = ycal[indices][np.argmax(velocity[indices])]
            rep_stats[f"rep{rep}"]['x_rom'] = np.max(xcal[indices]) - np.min(xcal[indices])
            rep_stats[f"rep{rep}"]['y_rom'] = np.max(ycal[indices]) - np.min(ycal[indices])
            rep_stats[f"rep{rep}"]['time_to_complete'] = set_data['Time'].values[indices][-1] - set_data['Time'].values[indices][0]
        set_metadata['rep_stats'] = rep_stats
        # Update the table and plots
        self.update_table(set_metadata['rep_stats'])
        self.update_plots(set_data)

        # Write the metadata
        # From https://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module
        # Since json can't encode floats??
        # I really don't want to loop through the dict, check type, and convert floats to strings but...
        # json.encoder.FLOAT_REPR = lambda o: format(0, '.2f')
        # metadata = open(meta_file, 'w')
        # json.dump(set_metadata, metadata, indent=4)
        # metadata.close()

        # Adjust UI back
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Analysis complete!', 5000)

    def update_plots(self, data):
        """
        Adapt timeline and motion plots to new log and analysis.

        Parameters
        ----------
        data : DataFrame
            Data from the analyzed log. Must have columns for Time, X_m, Y_m, Velocity, and Reps.
        """
        self.t2.clear()
        y_pen = pg.mkPen(color='#E4572E', width=1.5)
        v_pen = pg.mkPen(color='#17BEEB', width=1.5)
        self.t1.plot(data['Time'].values, data['Y_m'].values, pen=y_pen, clear=True)
        self.t2.addItem(
            pg.PlotCurveItem(data['Time'].values, data['Velocity'].values, pen=v_pen, clear=True))

        m_pen = pg.mkPen(color='#76B041', width=1.5)
        self.xy.plot(data['X_m'].values[20:], data['Y_m'].values[20:], pen=m_pen, clear=True)

        reps_labeled, n_reps = label(data['Reps'].values)
        if n_reps != 0:
            for rep in range(1, n_reps + 1):
                indices = [reps_labeled == rep]
                t_l = data['Time'].values[indices][0]
                t_r = data['Time'].values[indices][-1]
                pk_vel = np.max(data['Velocity'].values[indices])
                if pk_vel >= 1.2:
                    rep_color = '#76B041'
                else:
                    rep_color = '#E4572E'
                lri_brush = pg.mkBrush(color=rep_color)
                lri_pen = pg.mkPen(color=rep_color)
                lri = pg.LinearRegionItem((t_l, t_r), brush=lri_brush, pen=lri_pen, movable=False)
                lri.setOpacity(0.3)
                ti = pg.TextItem(text=f"{rep}", color='#FFC914', anchor=(0.5, 0.5))
                ti.setPos((t_r + t_l) / 2, 0.9)
                self.t1.addItem(lri)
                self.t1.addItem(ti)

    def update_timeline_view(self):
        self.t2.setGeometry(self.t1.vb.sceneBoundingRect())
        self.t2.linkedViewChanged(self.t1.vb, self.t2.XAxis)

    def closeEvent(self, event):
        """
        Offer lifter opportunity to cancel closing.
        If lifter does want to close, save current settings to a json file.
        """
        quit_message = 'Are you sure you want to quit?'
        reply = QtWidgets.QMessageBox.question(self, 'Message', quit_message,
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.save_settings()
            event.accept()
        else:
            event.ignore()

    # Methods for utility

    def handle_color_selection(self, event, x, y, flags, frame):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mask_colors.append(frame[y, x].tolist())
            self.selecting_active = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting_active is True:
                self.mask_colors.append(frame[y, x].tolist())
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting_active = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyWindow('Masked')
            self.mask_colors = deque()
            self.reset_colors()

    def build_filepaths(self):
        """
        Builds a filename based on:
            Timestamp with format YYYYMMDD-HHMMSS
            Exercise name
            Weight in kilograms

        Returns
        -------
        list
            Paths to use for saving a video (index 0), a log (index 2), and the resulting metadata.
        """
        timestamp = time.strftime('%y%m%d-%H%M%S')
        exercise = self.comboExercise.currentText().lower().replace(' ', '')
        kilos = f"{int(round(self.spinKgs.value(), 0))}kg"
        video_path = os.path.join(self.data_dir, f"{timestamp}_{exercise}_{kilos}.mp4")
        log_path = os.path.join(self.data_dir, f"{timestamp}_{exercise}_{kilos}.csv")
        metadata_path = os.path.join(self.data_dir, f"{timestamp}_{exercise}_{kilos}.json")
        return [video_path, log_path, metadata_path]

    def analyze_set(self, t, x, y, r):
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

        Returns
        -------
        DataFrame
            A new DataFrame with updated information for more advanced analytics.
        """
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Analyzing set...')

        # Need to zero out bullshit numbers like 3.434236345E-120 or some shit
        t[0] = 0
        t[np.abs(t) < 0.0000001] = 0
        x[np.abs(x) < 0.0000001] = 0
        y[np.abs(y) < 0.0000001] = 0
        r[np.abs(r) < 0.0000001] = 0

        # Need some info from the UI
        nominal_radius = self.spinDiameter.value() / 2 / 1000  # meters
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
        reps_binary = utilities.find_reps(ycal, threshold=0.01, open_size=5, close_size=9)

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

        self.statusbar.clearMessage()

        return set_analyzed

    def post_process_video(self, video_file, n_frames, set_data):
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
        self.statusbar.clearMessage()
        self.statusbar.showMessage('Tracing and converting video...')

        # Get smoothed motion to remove outliers and make path nicer
        xsmooth = medfilt(set_data['X_pix'].values, 7)
        ysmooth = medfilt(set_data['Y_pix'].values, 7)
        # Make lists of smoothed points for each set
        # rep_points = {}
        # for r in range(1, np.max(set_data['Reps'].values) + 1):
        #     rep_points[f"{r}"] = [xsmooth[set_data['Reps'].values==r], ysmooth[set_data['Reps'].values==r]]

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

        self.statusbar.clearMessage()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = KiloCountLogApp()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())