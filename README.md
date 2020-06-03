# |||---barbell-cv---|||

Get started with velocity-based training with nothing but a laptop, a webcam,
and a high contrast marker on your barbell.

![Power snatch logged in barbellcv.](docs/front_squat_failed.png)

## Quick Start

1. Download this repository as a .zip file, then unzip it where you like
2. Go to that directory using CMD (Windows) or the terminal (Mac/Linux)
3. Make sure the correct dependencies are installed using pip:
    ```
    python -m pip install -r requirements.txt
   ```
4. Run the program:
    ```
    python main.py
    ```
5. Preview your webcam using the "Preview" button, and rotate it if needed using the adjacent dropdown.
Press Enter to escape the preview.
6. Select the color of your barbell marker interactively.
    - Press the "Select Color" button.
    - Drag your mouse over the marker in the popup (make sure to move the marker around and select it
    under varying light conditions and angles).
    - Press Enter when the marker is tracked satisfactorily.
7. Select the exercise you want to do.
    - Add exercises that you find missing to the /resources/lifts.json file.
8. Input the weight for the set in lbs or kgs.
9. Press "Log Set" and wait for the webcam preview to show before lifting.
10. After lifting, press the Enter key to complete the set.
11. The results for the set are shown.

    ![Output from example set.](docs/failure_criteria.png)

12. Optionally reclassify phases of lifts for each rep.
    - For example, three reps will likely be found for a single clean and jerk. Individually label the
    concentric phases of the clean, then the front squat, then the jerk separately using the dropdown
    in row 1 of the table.
    - Also mark false positive reps as FALSE, failed reps as FAILED, or incorrectly detected reps for
    which the entire ROM was not found as PARTIAL.
    - Below is an example of a set of front squat triples where the unracking of the bar was detected
    as rep 1. The lifter was able to reclassify this false positive rep as FALSE, erasing that incorrect
    "rep" from the set.
     
    ![Front squat set where the unracking was detected as a false rep.](docs/front_squat_false_rep.png)
    
    ![Front squat set where the unracking was corrected.](docs/front_squat_corrected.png)

## Instructions
*More in-depth instructions coming soon after testing or available on request.*

## Limitations
For the video analysis to work correctly on Windows 10 you may need to install the basic
[K-Lite Codec Pack](https://codecguide.com/download_kl.htm).

*As I test this setup more I will learn more about its accuracy and what the limitations are, then update
this section. Suffice it to say that all results should be taken with a grain of salt until verified.*

*If you install this program and use it for your training, I'd love to hear your feedback. For any
bugs or suggestions please open an issue here.*
