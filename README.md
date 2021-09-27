# UnityUIToolkit

## Preamble
### Motivation
This toolkit was developed for the dissertation, 'A model-based design tool for 3D GUI layout design that accommodates user attributes'
submitted in 2021 for the MPhil in Machine Learning & Machine Intelligence program at the University of Cambridge.

### Proposed user workflow
Begin by adjusting the weights of the objective functions based on the factors which will be most impactful for the target user group.
Once the UI is generated, you may further fine-tune the weights to continually adjust the UI layout.
Once complete, you may record the optimal locations and colors of the UI panels (outputted from python_server.py), as well as the weights used.
These can also be used in the preference learning application to adjust the UI layout based on preference learning.

## Toolkit Definitions
**Color Harmony**: Range of colors with similar hues on the HSV scale. <br />
**Colorfulness**: Measure based on the amount of coloration in the users' environment. <br />
**Edgeness**: Measure based on the amount of 'busyness' in the users' environment. <br />
**Fitts' Law**: Average movement time to each UI panel as a function of index of difficulty. <br />
**Consumed Endurance**: Severity of upper-arm fatigue from prolonged arm use. <br />
**Muscle Activation**: Muscle activation of the upper arms. <br />
**RULA**: Amount of 'risk' associated with the current arm posture. <br />
**Cognitive Load**: Measure of the users’ workload or cognitive usage. <br />

## Running the Unity UI Toolkit

### Step 1: Take a photo of the environment
Open and build the "ImageCapture" scene with a USB-connected Hololens. Locate the Python script 'hololens_server.py' in the Assets/Scripts/Python Scripts folder and run.
Run the HololensComms app on the Hololens and press the camera button to take a photo.
The Python script should output the number of bytes received, as well as the file names of the files generated. There should be three files generated:
  1. context_img_XXXXXXXXXX.png: The image file
  2. context_img_XXXXXXXXXX.log: The image metadata file containing  Hololens camera to world matrix and projection matrix
  3. context_img_buff_XXXXXXXXXX.log: The image buffer file

### Step 2: Specify designer constraints
Open the "UITool" scene. Under the UI toolkit prefab, find the UI tool script in the inspector pane.
Input the image buffer file and meta file to taken in Step 1 to use for the user environment image (sample buffer and meta image files are provided in the
Assets/Scripts/Python Scripts/input_images folder). Specify the number of panels you would like to appear in the UI.
For each panel, you may input the application type and dimensions (in meters).
*Note: If no buffer image file is specified, then the toolkit will use the Unity environment (e.g. the office/classroom/lab virtual environment) as the input.*

<p align="center">
  <img src="https://github.com/jwlee97/UnityUIToolkit/blob/master/SampleImages/sceneview.png" />
  <img src="https://github.com/jwlee97/UnityUIToolkit/blob/master/SampleImages/menu.png" />
</p>

### Step 3: Send constraints to socket
When all constraints have been specified, open and run python_server.py located in the Assets/Scripts/Python Scripts folder. Run the Unity scene.

### Step 4: Specify objective function weights
Run the Unity simulation. In the game mode window, specify the weights of each objective function using the sliders.
Ticking 'Enable occlusion' will allow the panels to overlap, and ticking 'Enable color harmony' will allow the toolkit to color harmonize the panels in the UI layout.

<p align="center">
  <img src="https://github.com/jwlee97/UnityUIToolkit/blob/master/SampleImages/constraints.PNG" />
</p>

### Step 5: Submit Constraints
Once complete, press 'Submit Constraints'. The toolkit will take several seconds to optimize the layout and transmit the data over the socket.
The python script will output the optimal locations for each panel during the process. 

<p align="center">
  <img src="https://github.com/jwlee97/UnityUIToolkit/blob/master/SampleImages/python_script.png" width="600"/>
</p>

### Step 6: Show Optimal UI
Once the Python server has sent the optimal locations over the socket, press 'Show Optimal UI' and choose a cognitive load > 0.0 using the slider on the Unity toolkit menu.
You should be able to see the UI layout in Unity in various environments by changing the value for cognitive load.

<p align="center">
  <img src="https://github.com/jwlee97/UnityUIToolkit/blob/master/SampleImages/classroom.jpg" width="600"/>
</p>

## Optional Steps

### Change Layout
You may also re-adjust the objective function weights to generate another UI layout. After re-adjusting the slider values, press the 'Submit Constraints' button
and wait for the Python script to finish optimizing and transmitting values back to the Unity toolkit. Once this is finished, press the 'Show Optimal UI' button again.

### Display 2D Layout on Image
If desired, you may also see the UI layout projected onto the initial user environment image by running UIoptimizer.py in the Assets\Scripts\Python Scripts folder.
Change the image file, panel sizes, and objective function weights in main() to adjust the UI layout. The script will output the layout image as \output_images\out.png.

<p align="center">
  <img src="https://github.com/jwlee97/UnityUIToolkit/blob/master/SampleImages/office_UI.PNG" width="600"/>
</p>

### Preference Learning
To launch the preference learning application in a separate window, press the 'run preference learning' button. The 'preference' image in the first iteration will be the output of the weighted sum optimization using the weights specified in the Unity toolkit. 
At each iteration, choose the preferred UI layout - this will become the 'preference' image for the next iteration, and the application will suggest a new layout as the 'suggestion'. When finished, press 'quit'. The script will display the optimized UI layout when finished in Unity, and output the optimal locations of each panel in world coordinates.

<p align="center">
  <img src="https://github.com/jwlee97/UnityUIToolkit/blob/master/SampleImages/preflearning_iteration.png" width="600"/>
</p>
