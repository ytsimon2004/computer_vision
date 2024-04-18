Computer vision CV2 GUI
======== 
by Yu-Ting Wei (ytsimon2004@gmail.com)


# Installation

- Create environment for the required dependencies

```bash
conda create -n comvis python~=3.10.0
conda activate comvis
pip install -r requirements.txt
```

- Buildup the src path

```bash
  conda install conda-build
  conda develop src
```

# GUI usage

## CV2Player

~~~
python play.py [-F <VIDEO FILE>] [-O <OUTPUT>]
~~~
![example_gui_view.png](figures%2Fexample_gui_view.png)

- CV2-based video player (use [OpenAI Sora](https://openai.com/sora) as an example video)
- See also in [player_GUI.py](./src/comvis/gui/player_GUI.py)
- `-F` specify a input video file path, `-O` save as processed video as another .mp4 or .avi (optional) 

### Keyboard Control:
- space = play/pause
- left = go back 10 frames
- right = go forward 10 frames
- left square bracket `[` = go to the beginning frame
- right square bracket `]` = go to the end frame
- plus `+` = playing speed x 2
- minus `-` = playing speed / 2
- backspace: delete a command word
- escape: clear all the commands
- enter: run the command
- Other keys: type commands

### Mouse Control
- Left bottom pointing toward the time bar to select the frame of interest

### command
- :h = print help
- :q = quit the GUI
  
-----------------------

## ImageProcPlayer
~~~
python image_proc_GUI.py [-F <VIDEO FILE>] [--json <PARS FILE>] [-O <OUTPUT>]
~~~
![example_process.png](figures%2Fexample_process.png)
- CV2-based video player for seeing image process effect
- See also in [image_proc_GUI.py](./src/comvis/gui/image_proc_GUI.py)
- `--json` specify a json filepath for storage the parameters for all the image processing function,
  default is `None`, then generate a default file under the same directory as video file.
  See also in [process_pars.py](src%2Fcomvis%2Futils%2Fprocess_pars.py)

### keyboard Control:
- inherit all the usage in `CV2Player`
  
### Mouse Control
- inherit all the usage in `CV2Player`
- Right button for selecting the ROI for image processing, otherwise do for the whole view

### command
- :h = print help
- :d = delete the selected ROI
- :q = quit the GUI
- :gray = switch the image to grayscale
- :blur = GaussianBlur the image 
- :bilateral = Bilateral filter the image
- :sharpen = Sharpen the image by applying filter2D
- :sobelX | :sobelY | :sobelXY = sobel edge detection
- :canny = canny edge detection
- :circle = HoughCircles circular object detection
- :red = Detect red object and enhance the brightness

--------------

## ObjTrackerPlayer
~~~
python object_tracker_GUI [-F <VIDEO FILE>]  [-O <OUTPUT>] [-T | --tracker <CV2 TRACKER>]
~~~

![example_grab_obj.jpg](figures%2Fexample_grab_obj.jpg)![example_obj_detect.gif](figures%2Fexample_obj_detect.gif)

- CV2-based video player to see the object tracking result
- See also [object_tracker_GUI.py](src%2Fcomvis%2Fgui%2Fobject_tracker_GUI.py)
- `-T` or `--tracker` specify which tracker. currently support kcf, csrt, mil in cv2

### keyboard Control:
- inherit all the usage in `CV2Player`

### Mouse Control
- Left button for selecting a ROI for object tracking

--------------

### ProcessParameters
- Dictionary for storing the parameters for all the cv2 function for image processing 
- default as:
~~~
  DEFAULT_PROC_PARS: ProcessParameters = {
    'gaussian_blur': GaussianBlurPars(ksize=5, sigma=60),
    'bilateral': BilateralPars(d=30, sigma_color=75, sigma_space=75),
    'filter2d': Filter2DPars(
        kernel=np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
    ),
    # detect
    'sobelX': SobelPars(ddepth=None, dx=1, dy=0, ksize=3, scale=1, delta=0),
    'sobelY': SobelPars(ddepth=None, dx=0, dy=1, ksize=3, scale=1, delta=0),
    'sobelXY': SobelPars(ddepth=None, dx=1, dy=1, ksize=3, scale=1, delta=0),
    'canny': CannyPars(lower_threshold=30, upper_threshold=150),
    'hough_circles': HoughCirclesPars(method=cv2.HOUGH_GRADIENT, dp=1, param1=100, param2=30, minRadius=10,
                                      maxRadius=30),
    'color_grab': ColorGrabPars(lower_color=np.array([35, 0, 0]),
                                upper_color=np.array([100, 60, 60]),
                                to_color=np.array([255, 0, 0]))
}

  ~~~

----
