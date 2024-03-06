Computer vision
======== 
by Yu-Ting Wei (ytsimon2004@gmail.com)

- Materials for *Computer Vision Course*
- Check source code in `src/comvis/*`
- Implementation showcases in `notebook/`


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
- CV2-based video player 
- See also in [player_GUI.py](./src/comvis/gui/player_GUI.py)
- ~~~
  python play.py -F <VIDEO FILE>
  ~~~
### Keyboard Control:
- space = play/pause
- left = go back 10 frames
- right = go forward 10 frames
- left square bracket `[` = go to the beginning frame
- right square bracket `]` = go to the end frame
- plus `+` = playing speed x 2
- minus `-` = playing speed / 2
- Other keys: type commands
- backspace: delete a command word
- escape: clear all the commands
- enter: run the command

### Mouse Control
- Left bottom pointing toward the time bar to select the frame of interest

### command
- :h = print help
- :q = quit the GUI
  
-----------------------

## ImageProcPlayer
- CV2-based video player for seeing image process effect
- See also in [image_proc_GUI.py](./src/comvis/gui/image_proc_GUI.py)
- ~~~
  python image_proc_GUI.py -F <VIDEO FILE> --json <PARS> -O <OUTPUT>
  ~~~
- `--json` specify a json filepath for storage the parameters for all the image processing function,
    default is `None`, then generate a default file under the same directory as video file. See also in `ProcessParameters`

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
- :sharpen = Sharpen the image by applying filter2D
- :sobelX | :sobelY | :sobelXY = sobel edge detection
- :canny = canny edge detection
- :circle = HoughCircles circular object detection

### ProcessParameters
- Dictionary for storing the parameters for all the cv2 function for image processing 
- default as:
~~~
  DEFAULT_PROC_PARS: ProcessParameters = {
    'GaussianBlur': GaussianBlurPars(ksize=5, sigma=60),
    'Canny': CannyPars(lower_threshold=30, upper_threshold=150),
    'Filter2D': Filter2DPars(
        kernel=np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
    ),
    #
    'SobelX': SobelPars(ddepth=None, dx=1, dy=0, ksize=3, scale=1, delta=0),
    'SobelY': SobelPars(ddepth=None, dx=0, dy=1, ksize=3, scale=1, delta=0),
    'SobelXY': SobelPars(ddepth=None, dx=1, dy=1, ksize=3, scale=1, delta=0),
    #
    'HoughCircles': HoughCirclesPars(method=cv2.HOUGH_GRADIENT, dp=1, param1=100, param2=30, minRadius=10, maxRadius=30)

}
  ~~~

----