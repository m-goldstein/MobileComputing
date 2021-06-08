#!/usr/bin/env python
# coding: utf-8

# # ECE/CS 434 | MP4: IMU PDR
# <br />
# <nav>
#     <span class="alert alert-block alert-warning">Due at 11:59PM April 13th 2021 on Gradescope</span> |
#     <a href="https://www.gradescope.com/courses/223105">Gradescope</a> | 
#     <a href="https://courses.grainger.illinois.edu/cs434/sp2021/">Course Website</a> | 
#     <a href="http://piazza.com/illinois/spring2021/csece434">Piazza</a>
# </nav><br> 
# 
# **Name(s):** _ , _<br>
# **NetID(s):** _ , _
# 
# <hr />  

# ## Objective
# In this MP, you will:
# - Implement a step counting algorithm using accelerometer data.
# - Apply signal processing and linear algebra functions such as low/high pass filtering, peak detection, fft, etc. to the step counting algorithm.
# - Calculate phone orientation using a single static accelerometer reading. 
# - Track phone orientation through a sequence of gyroscope data by performing integration.

# ---
# ## Problem Overview
# In pedestrian dead-reckoning applications, two pieces of information need to be tracked: how far a user walked, and the direction of the walk. In the first part of this MP, you will write a step counter using accelerometer data as input. In the second part, you will derive the initial orientation of the phone using a single accelerometer reading and calculate the final orientation using a sequence of gyroscope data.

# ---
# ## Imports & Setup
# 
# ### Installing requirements correctly
# 
# First. we will make sure that the correct versions of required modules are installed. This ensures that your local Python environment is consistent with the one running on the Gradescope autograder. Just convert the following cell to code and run:
# 
# <div class="alert alert-block alert-info"><b>Note:</b> It's preferred that your local environment matches the autograder to prevent possible inconsistencies. However, if you're running into annoying Python version issues but haven't had any issues getting consistent results on the autograder, there is no need to stress over it. Just skip for now and come back when you do encounter inconsistencies:) Ditto below.
# </div>
# 
# <div class="alert alert-block alert-info"><b>WARNING:</b> ENSURE THE FOLLOWING CELL IS MARKDOWN OR DELETED BEFORE SUBMITTING. THE AUTOGRADER WILL FAIL 
# </div>

# if __name__ == '__main__':
#     import sys
#     !{sys.executable} -m pip install -r requirements.txt

# ### Your imports
# Write your import statements below. If Gradescope reports an error and you believe it is due to an unsupported import, check with the TA to see if it could be added.

# In[2]:


import numpy as np
import pandas as pd

# This function is used to format test results. You don't need to touch it.
def display_table(data):
    from IPython.display import HTML, display

    html = "<table>"
    for row in data:
        html += "<tr>"
        for field in row:
            html += "<td><h4>{}</h4><td>".format(field)
        html += "</tr>"
    html += "</table>"
    display(HTML(html))


# ### Sanity-check
# 
# Running the following code block verifies that the correct module versions are indeed being used. 
# 
# Try restarting the Python kernel (or Jupyter) if there is a mismatch even after intalling the correct version. This might happen because Python's `import` statement does not reload already-loaded modules even if they are updated.

# In[3]:


if __name__ == '__main__':
    from IPython.display import display, HTML

    def printc(text, color):
        display(HTML("<text style='color:{};weight:700;'>{}</text>".format(color, text)))

    _requirements = [r.split("==") for r in open(
        "requirements.txt", "r").read().split("\n")]

    import sys
    for (module, expected_version) in _requirements:
        try:
            if sys.modules[module].__version__ != expected_version:
                printc("[✕] {} version should to be {}, but {} is installed.".format(
                    module, expected_version, sys.modules[module].__version__), "#f44336")
            else:
                printc("[✓] {} version {} is correct.".format(
                    module, expected_version), "#4caf50")
        except:
            printc("[–] {} is not imported, skipping version check.".format(
                module), "#03a9f4")


# ---
# ## Part 1. Step Counter
# We have provided you with smartphone accelerometer data collected under three circumstances
# <ol type="A">
#   <li>walking with phone in pant pocket</li>
#   <li>walking with phone held in the hand statically as if the user is looking at it while walking</li>
#   <li>walking with phone in hand and the hand swinging</li>
# </ol>
# For each file, there are three columns, representing the accelerometer readings in three local axes(unit: $m / s^{2}$). The accelerometer is sampled at 100Hz.
# 
# Implement your algorithm in the `count_steps(walk_accl_file)` function below. Do NOT change the function signature. You are, however, free to define and use helper functions. You are expected to use common signal processing and linear algebra functions (e.g., high/low pass filtering, convolution, cross correllation, peak detection, fft etc.) 

walk_accl_files = ['data/holdstatic_20steps.csv', 'data/inpocket_26steps.csv',
                    'data/inpocket_36steps.csv', 'data/swing_32steps.csv', 'data/swing_38steps.csv']
# In[ ]:
import scipy
import scipy.signal
import numpy
import numpy.linalg as LA
import matplotlib.pyplot as plt

def lpf_response(cs=200/60,fs=100,order=4):
    nyquist = (1/2)*fs
    thresh = cs/nyquist
    return scipy.signal.butter(order,thresh,btype='low',analog=False)

# cs : cutoff; 200 steps per minute; this is better than record marathon runners
def lpf(vals,cs=200/60,fs=100,order=4):
    num,dem=lpf_response(cs=cs,fs=fs,order=order)
    return scipy.signal.lfilter(num,dem,vals)

# This function takes 1 argument:
#     walk_accl_file  (string) - name of data file for accelerometer data
# It returns an integer, the number of steps

def count_steps(walk_accl_file):
    # Your implementation starts here:
    data = pd.read_csv(walk_accl_file,header=None)
    x_dim,y_dim,z_dim = np.asarray(data[0]),np.asarray(data[1]),np.asarray(data[2])
    mag = np.asarray([np.sqrt((x_dim[i]**2+y_dim[i]**2+z_dim[i]**2)) for i in range(len(x_dim))])
    mag_adjusted = np.asarray([e-mag.mean() for e in mag])
    baseline = np.std(mag_adjusted)
    filtered_mags  = lpf(mag_adjusted,order=4) # i think 4th order? 
    [peak_vals,peak_locs] = scipy.signal.find_peaks(filtered_mags,prominence=baseline)
    return len(peak_vals)


# ### Run & Test
# Use the cell below to run and test `count_steps(walk_accl_file)`. 

# In[ ]:


def estimate_steps_score(calculated, expected):
    delta = abs(calculated - expected)
    return 1 if(delta <= 2) else max((1 - abs(delta - 2) / expected), 0)


if __name__ == '__main__':
    walk_accl_files = ['data/holdstatic_20steps.csv', 'data/inpocket_26steps.csv',
                       'data/inpocket_36steps.csv', 'data/swing_32steps.csv', 'data/swing_38steps.csv']
    groundtruth = [20, 26, 36, 32, 38]
    output = [['Dataset', 'Expected Output', 'Your Output', 'Grade']]
    for i in range(len(groundtruth)):
        calculated = count_steps(walk_accl_files[i])
        score = estimate_steps_score(calculated, groundtruth[i])
        output.append([walk_accl_files[i], groundtruth[i],
                      calculated, "{:2.2f} / 5.00".format(score * 5)])
    output.append(['<i>👻 Hidden test 1 👻</i>','<i>???</i>', '<i>???</i>', '<i>???</i> / 15.00'])
    output.append(['<i>...</i>', '<i>...</i>', '<i>...</i>', '<i>...</i>'])
    output.append(['<i>👻 Hidden test 5 👻</i>','<i>???</i>', '<i>???</i>', '<i>???</i> / 15.00'])
    display_table(output)


# ---
# ## Part 2. Orientation Tracking
# 
# ### Part 2.1 Initial Orientation Calculation
# Assume the phone is static at the initial moment. We will provide you with the accelerometer reading at that moment (unit: $m / s^{2}$). Your goal is to identify the initial phone orientation from this reading. We will not provide compass data here since all the data are collected indoor and compass won’t give an accurate north indoor. Instead, assume at the initial moment, the projection of the phone’s local Y axis onto the horizontal plane is pointing towards the global Y axis. This will also give a fixed phone initial orientation.
# 
# **We expect you to output the global direction in which the phone’s local X axis is pointing at.**
# 
# <div class="alert alert-block alert-info"><b>Hint:</b> Find the global Y axis’s direction in the local frame and let this direction be a 3 × 1 vector $v_{1}$. Let the gravity in
# the local frame be another 3 × 1 vector $v_{2}$. Then essentially you need to solve the following equation: <br> $
# R\left[v_{1} v_{2}\right]=\left[\begin{array}{ll}
# 0 & 0 \\
# 1 & 0 \\
# 0 & 1
# \end{array}\right]$ </div>

# ### Part 2.2 3D Orientation Tracking
# In this part, you need to take the initial orientation calculated in part 1, and perform gyro integration for each timestamp onward. We will provide you with a trace of gyroscope data, in CSV format. There are three columns in the file, representing the gyroscope readings in three **local** axes (unit: $rad / s$). The gyroscope is sampled at 100Hz. Your task is to track the phone’s 3D orientation and **output the end direction in which the phone’s local X axis is pointing at in the global frame**.
# 
# One way of solving this problem can be:
# <ol type="A">
#     <li> Assume the gyroscope’s sample interval is $\Delta t$. </li>
#     <li> Get the phone's instant rotation axis and rotation angle in the local frame $(\vec{l}, \Delta \theta)$ for each time stamp $t_{i},$ where $\vec{l}=\left(\omega_{x}, \omega_{v}, \omega_{z}\right)$ and $\Delta \theta=\sqrt{\left(\omega_{x}^{2}+\omega_{v}^{2}+\omega_{z}^{2}\right)} \cdot \Delta t$ </li>
#     <li> Project the instant rotation axis $\vec{l}$ into the global frame using the phone's $3 \mathrm{D}$ orientation matrix $R_{i}$ at time $t_{i}$. </li>
#     <li> Convert the instant rotation axis and angle in global frame into the form of rotation matrix $\Delta R_{i}$. </li>
#     <li> Find the total 3D rotation matrix for time $t_{i+1}: R_{i+1}=\Delta R_{i} \cdot R_{i}$ </li>
# </ol>
# 
# --- 
# **Implement both algorithms in `track_orientation(orientation_accl_file, gyro_file)` below.** This is because the initial rotation matrix needed for calculating final orientation is a by-product of calculating initial orientation. Do NOT change the function signature. You are, however, free to define and use helper functions.

# In[ ]:
#SOURCES: 
## https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjK6aTDtvXvAhXxdc0KHVfqB6IQFjACegQIBxAD&url=http%3A%2F%2Fwww.cse.psu.edu%2F~mkg31%2Fteaching%2Fcse_ee597%2Fclass_material%2FIMU_basics.pptx&usg=AOvVaw3Gx_r7rAEHUR9M9mE9CFcJ
## https://arxiv.org/pdf/1704.06053.pdf
## https://www.allaboutcircuits.com/technical-articles/how-to-interpret-IMU-sensor-data-dead-reckoning-rotation-matrix-creation/
def calc_magnitude(X):
    acc = 0.0
    for e in X:
        acc += np.power(e,2.0)
    return np.sqrt(acc)

# From the wikipedia article on Rotation Matrices
def construct_R(ang,wx,wy,wz):
    C = np.cos(ang)
    S = np.sin(ang)
    wx2 = wx**2
    wy2 = wy**2
    wz2 = wz**2
    I = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    Ux = np.matrix([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])
    Ux2 = np.matrix([[wx2,wx*wy,wx*wz],[wx*wy,wy2,wy*wz],[wx*wz,wy*wz,wz2]])
    R = C*I+S*Ux+(1-C)*Ux2
    return R
fs = 100
orientation_accl_file = 'data/orientation_accl.csv'
gyro_file = 'data/gyro.csv'

"""
# This function takes 2 arguments:
#     - orientation_accl_file (string) - name of file containing a single accl reading
#     - gyro_file (string) - name of file containing a sequence of gyroscope data
# It returns two arguments: an array representing the initial global direction
# in which the phone's local X axis is pointing at, and the final.
"""
def track_orientation(orientation_accl_file, gyro_file):
    # Your implementation starts here:
    orientation_data = pd.read_csv(orientation_accl_file,header=None)
    gyro_data = pd.read_csv(gyro_file,header=None)
    gyro_x,gyro_y,gyro_z = np.asarray(gyro_data[0]),np.asarray(gyro_data[1]),np.asarray(gyro_data[2])
    acc_x,acc_y,acc_z = orientation_data[0],orientation_data[1],orientation_data[2]
    grav = np.asarray([float(acc_x),float(acc_y),float(acc_z)])
    b = np.matrix([[0,1,0],[0,0,1]]).T
    v2 = grav/LA.norm(grav)

    # Find an orthogonal vector (kind of like Gram-Schmidt)
    v1 = b[:,0].T - ((b[:,0].T@v2)*v2)
    v1 = v1/LA.norm(v1)
    A        = np.zeros((3,3))
    A[:,0:1] = v1.reshape(3,1)
    A[:,1:2] = v2.reshape(3,1)
    A[:,2:3] = np.cross(v1,v2).reshape(3,1)
    A        = A.T
    
    # find a,b,c coeffs
    R = np.zeros((3,3))
    R[:,0] = (LA.inv(A)@np.asarray(b[:,1])).T 
    R[:,1] = v1
    R[:,2] = v2
    R      = R.T
    init   = [e for e in R[:,0]]
    angular_mag = np.asarray([(1.0/fs)*np.sqrt(gyro_x[i]**2+gyro_y[i]**2+gyro_z[i]**2) for i in range(len(gyro_x))])
    rot_inst = np.asarray([(np.asarray([gyro_x[i],gyro_y[i],gyro_z[i]]).T,angular_mag[i]) for i in range(len(angular_mag))],dtype='object')
    R_i = [R]
    # use gyroscope data to find new orientation
    for i in range(0,len(rot_inst)):
        (lx,ly,lz),ang = rot_inst[i]
        p = R_i[i]@np.matrix([[lx],[ly],[lz]])
        p = p/LA.norm(p)
        wx,wy,wz = [float(e) for e in p]
        # Update R_i
        R_i.append(construct_R(ang,wx,wy,wz)@R_i[-1])
    final = [float(e) for e in R_i[-1][:,0]]
    return [
        init,
        final,
    ]  # [initial orientation], [final orientation]


# ### Run & Test
# Use the cell below to run and test Part 2.

# In[ ]:


def get_deviation(calculated, expected):
    calculated = np.array(calculated)
    expected = np.array(expected)
    with np.errstate(divide='ignore', invalid='ignore'):
        dot_prod = np.dot(calculated, expected) /             np.linalg.norm(calculated) / np.linalg.norm(expected)
        return np.degrees(np.arccos(dot_prod))


if __name__ == '__main__':
    gt_init = [0.9999, -0.0020, 0.0120]
    gt_final = [-0.0353, 0.9993, 0.0076]
    stu_init, stu_final = track_orientation(
        'data/orientation_accl.csv', 'data/gyro.csv')

    output = [['Test', 'Dataset', 'Expected Output',
               'Your Output', 'Deviation', 'Result', 'Grade']]
    init_state = 'FAILED'
    final_state = 'FAILED'
    init_grade = 0
    final_grade = 0
    init_dev = get_deviation(stu_init, gt_init)
    final_dev = get_deviation(stu_final, gt_final)
    if(init_dev < 2):
        init_state = 'PASSED'
        init_grade = 10
    if(final_dev < 2):
        final_state = 'PASSED'
        final_grade = 10
    output.append(['Initial Orientation',
                  'orientation_accl.csv, gyro.csv', gt_init, stu_init, "{:2.2f}°".format(init_dev), init_state, "{} / 10".format(init_grade)])
    output.append(['Final Orientation', 'orientation_accl.csv, gyro.csv',
                  gt_final, stu_final, "{:2.2f}°".format(final_dev), final_state, "{} / 10".format(final_grade)])
    output.append(['<i>👻 Hidden test 1 👻</i>','<i>???</i>', '<i>???</i>', '<i>???</i>', '<i>???</i>', '<i>???</i>', '<i>???</i> / 10'])
    output.append(['<i>...</i>', '<i>...</i>', '<i>...</i>', '<i>...</i>', '<i>...</i>', '<i>...</i>', '<i>...</i>'])
    output.append(['<i>👻 Hidden test 4 👻</i>','<i>???</i>', '<i>???</i>', '<i>???</i>', '<i>???</i>', '<i>???</i>', '<i>???</i> / 10'])
    display_table(output)


# ---
# ## Rubric
# 
# #### Step Counting (100 points) 
# You will be graded on the 5 sets of provided data (5 points each), as well as 5 sets of hidden data (15 points each). For each test case, the grade depends on how much the result deviates from the groudtruth. A 2-step error for the provided data is tolerated. A 4-step error for the hidden data is tolerated. For results greater than the error threshold, your score will be scaled proportionally.
# 
# ####  Orientation Tracking (100 points) 
# You will be graded on the provided data as well as 4 addition sets of data. They are each worth 20 points. A 2-degree error is tolerated. For results greater than the error threshold, no points will be rewarded since we provided a detailed algorithm to follow. The test data also include the simple case where the phone’s initial local frame is aligned with the global frame, and phone will only rotate along Z axis onwards. (In case you find the MP too difficult, only doing 1D integration on Z axis should at least give you some points.)

# ---
# ## Submission Guideline
# This Jupyter notebook is the only file you need to submit on Gradescope. If you are working in a pair, make sure your partner is correctly added on Gradescope and that both of your names are filled in at the top of this file.
# 
# **Make sure any code you added to this notebook, except for import statements, is either in a function or guarded by `__main__`(which won't be run by the autograder). Gradescope will give you immediate feedback using the provided test cases. It is your responsibility to check the output before the deadline to ensure your submission runs with the autograder.**
