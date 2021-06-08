#!/usr/bin/env python
# coding: utf-8

# # ECE/CS 434 | MP3: AoA
# <br />
# <nav>
#     <span class="alert alert-block alert-warning">Due March 28th 11:59PM 2021 on Gradescope</span> |
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
# - Implement algorithms to find angle of arrivals of voices using recordings from microphone arrays.
# - Perform triangulation over multiple AoAs to deduce user locations.
# - Optimize voice localization algorithms using tools from probability theory, or signal processing.

# ---
# ## Imports & Setup
# The following `code` cell, when run, imports the libraries you might need for this MP. Feel free to delete or import other commonly used libraries. Double check with the TA if you are unsure if a library is supported.

# In[2]:


import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize
"""if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use("seaborn") # This sets the matplotlib color scheme to something more soothing
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')

# This function is used to format test results. You don't need to touch it.
def display_table(data):
    from IPython.display import HTML, display
    html = "<table>"
    for row in data:
        html += "<tr>"
        for field in row:
            html += "<td><h4>%s</h4><td>"%(field)
        html += "</tr>"
    html += "</table>"
    display(HTML(html))


# ---
# ## Problem Description
# 
# Providing voice assistants with location information of the user can be helpful in resolving ambiguity in user commands. In this project, you will create a speaker localization algorithm using recordings from multiple voice assistant microphone arrays.
# 
# <figure>
# <img src="images/scenario.png" alt="AoA Scenario" style="width: 500px;"/>
# <figcaption>Figure 1: Application Scenario</figcaption>
# </figure>
# 
# Consider the following scenario: there are eight voice assistants around the user. We will provide you with the location of these eight devices $L_{0}, L_{1}, \ldots, L_{7}$, their microphone array configuration, and the recordings from each of these devices $D_{0}, D_{1}, \ldots, D_{7}$. Your algorithm should take $D_{0}, D_{1}, \ldots D_{7}$ and $L_{0}, L_{1}, \ldots L_{7}$ as input and output the location of the user $L_{x}$.
# 
# You can tackle this problem by doing AoA on all eight devices and then use triangulation to find the user
# location.

# ---
# ## Data Specification
# 
# Figure 3 shows the microphone array configuration. Each microphone array has 6 microphones indicated by green dots. They form a hexagon with mic #1 facing +x, mic #0 60 degrees counter-clockwise from mic #1, and so on. The diameter of the microphone array is $0.09218\text{ m}$(the distance between mic #0 and mic #3).  The sampling rate is $16000\text{ Hz}$.
# 
# Four sets of data can be found in `dataset#/`:
# ```
# ├── dataset0
# │   ├── 0.csv
# │   ├── 1.csv
# │   ├── ...
# │   ├── 7.csv
# │   └── config.csv
# ├── dataset1
# │   ├── ...
# ├── dataset2
# │   ├── ...
# └── dataset3
#     ├── 0.csv
#     ├── 1.csv
#     ├── ...
#     ├── 7.csv
#     └── config.csv
#     
# ```
# In each directory, `0.csv` through `7.csv` contain data collected at each of the 8 microphone arrays. They each have 6 columns, corresponding to recorded samples from individual microphones on the mic array, with column number matching mic number. `config.csv` contains the microphone array coordinates. There are 8 comma-separated rows, corresponding to the (x, y) coodinates of the 8 microphone arrays. This is visualized in Figure 2 below. Note that the coordinates are in metres.

# In[3]:


if __name__ == '__main__':
    array_locs = np.genfromtxt ('dataset0/config.csv', delimiter=",")
    user_1_location = np.array((3.0, 1.0))

    from matplotlib.patches import RegularPolygon, Circle
    fig, ax = plt.subplots(2, 1, figsize=(10,16))
    ax[0].set_title("Figure 2: A visual of the setting for user 1")
    ax[0].grid(b=True, which="major", axis="both")
    ax[0].set_xlim((-0.5, 6.5))
    ax[0].set_xticks(np.arange(0, 7))
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylim((-0.5, 5))
    ax[0].set_yticks(np.arange(0, 5))
    ax[0].set_ylabel("y (m)")
    for (loc_num, (loc_x, loc_y)) in enumerate(array_locs, start=0):
        ax[0].add_patch(RegularPolygon(
            xy=(loc_x,loc_y), 
            numVertices=6, 
            radius=0.2, 
            orientation=np.pi/6
        ))
        ax[0].text(
            x=loc_x, 
            y=loc_y, 
            s=loc_num,
            color="white", 
            horizontalalignment="center",
            verticalalignment="center",
        )
    ax[0].add_patch(Circle(xy=user_1_location,radius=0.2, color="#DB7093"))
    ax[0].text(user_1_location[0], user_1_location[1], "user 1", color="white", ha="center", va="center")
    ax[1].set_title("Figure 3: Microphone Array Configuration")
    ax[1].grid(b=True, which="major", axis="both")
    ax[1].set_xlim((-1.5,1.5))
    ax[1].set_xticks([0])
    ax[1].set_ylim((-1.0,1.3))
    ax[1].set_yticks([0])
    ax[1].add_patch(RegularPolygon((0,0), 6, 1, np.pi/6))
    for mic_i in np.arange(6):
        mic_pos = np.e**(-1j * 2 * np.pi / 6 *  mic_i)             * np.e**(1j * 2 * np.pi / 6)
        ax[1].add_patch(Circle(
            xy=(mic_pos.real, mic_pos.imag),
            radius=0.1, 
            color="#4c7d4c"
        ))
        ax[1].text(
            x=mic_pos.real, 
            y=mic_pos.imag, 
            s=mic_i,
            color="white", 
            horizontalalignment="center",
            verticalalignment="center",
        )
    ax[1].annotate(
        "",
        xy=(0.42, -0.75),
        xytext=(-0.42, 0.75),
        arrowprops=dict(arrowstyle="|-|", color="white", lw=2)
    )
    ax[1].text(0.15, 0, "0.09218 m", color="white", ha="center")
    plt.show()
"""

# ---
# ## Your Implementation
# Implement your localization algorithm in the function `aoa_localization(mic_data_folder, FS, MIC_OFFSETS)`. Do **NOT** change its function signature. You are, however, free to define and use helper functions. 
# 
# You are encouraged to inspect, analyze and optimize your implementation's intermediate results using plots and outputs. You may use the provided scratch notebook (`scratch.ipynb`) for this purpose, and then implement the relevant algorithm in the `aoa_localization` function (which will be used for grading). Your implementation for `aoa_localization` function should **NOT** output any plots or data. It should only return the user's calculated location.

# In[ ]:

def get_coords(fp,ap_id,MIC_OFFSETS,use_offsets=True):
    data = {}
    csvdata = pd.read_csv(fp)
    data = list(tuple((e[0], e[1])) for e in csvdata.values)
    data.insert(0, tuple((float(csvdata.columns[0]),float(csvdata.columns[1]))))
    if use_offsets==True:
        for i in range(len(data)):
            e = data[i]
            try:
                data[i] = tuple((e[0]+MIC_OFFSETS[ap_id][0], e[1]+MIC_OFFSETS[ap_id][1]))
            except:
                print('bad index: ap_id=%d\ti=%d'%(ap_id,i))
    return data
        
# assuming speed of sound is 331 m/s
def get_wavelength(freq=(2.4*(10**9))):
    return float((331*(10**1))/freq)

def to_rad(deg):
    return (float(deg)*(math.pi/180.0))

def to_deg(rad):
    return (float(180.0/math.pi)*float(rad))


def calc_array_steering_vector(num_mic,theta,in_deg=False):
    a = []
    d = (get_wavelength() * theta)/(2*math.pi)
    a1 = np.exp( ((-1j)*(2*math.pi)*d)/(get_wavelength()))
    for i in range(num_mic):
        if (i == 0):
            a.append(a1)
        else:
            if in_deg == False:
                trig = math.cos(theta)
                a2 = np.exp( ((-1j)*(2*math.pi*i*get_wavelength()))*trig)
            else:
                trig = math.cos(to_rad(theta))
                a2 = np.exp( ((-1j)*(2*math.pi*i*get_wavelength()))*trig)
            a.append(a1*a2)
    return a

def calc_signal(s, a,n=[0]):
    result = np.matmul(s,a) 
    return result
"""def calc_array_steering_vector(num_mic,theta,d,in_deg=False):
    a = []
    a1 = np.exp( ((-1j)*(2*math.pi)*d)/(get_wavelength()))
    for i in range(num_mic):
        if (i == 0):
            a.append(a1)
        else:
            if in_deg == False:
                trig = math.cos(theta)
                a2 = np.exp( ((-1j)*(2*math.pi*i*get_wavelength()))*trig)
            else:
                trig = math.cos(to_rad(theta))
                a2 = np.exp( ((-1j)*(2*math.pi*i*get_wavelength()))*trig)
            a.append(a1*a2)
    return a
def calc_theta(id_1,id_2,in_deg=False):
    ans = math.asin((to_rad(60.0*id_2)-to_rad(60*id_1))/math.pi)
    if (in_deg == True):
        return to_deg(ans)
    else:
        return ans
def calc_added_distance(id_1,id_2):
    ans = (get_wavelength()/2)*math.sin(calc_theta(id_1,id_2))
    return ans
"""

def dist(x,y):
    x = tuple((float(x[0]),float(x[1])))
    y = tuple((float(y[0]),float(y[1])))
    a1 = math.pow(x[0]-y[0],2.0)
    a2 = math.pow(x[1]-y[1],2.0)
    return math.sqrt(a1+a2)

def mean_square_error(coord, mic_locs, est_distances):
    err = 0.0
    for loc,d in zip(mic_locs, est_distances):
        guess = dist(coord,loc)
        err += math.pow(guess-d,2.0)
    return err/len(est_distances)

def find_loc(mic_locs, est_distances):
    sol = minimize(mean_square_error,(0,0),args=(mic_locs,est_distances),method='L-BFGS-B',options={'ftol':1e-5,'maxiter':1e+7})
    return sol.x
def main():
    sig = {}
    for i in range(4):
        for j in range(6):
            ap = AP(i,j)
            a = calc_array_steering_vector(6, (math.pi/3)*j)
            for k in range(len(ap.data.values)):
                sig['AP%d%d%d'%(i,j,k)] = calc_signal(ap.data.values[k],a)
                print(sig['AP%d%d%d'%(i,j,k)])
    return sig
class AP:
    ap_loc = None
    mic_locs = {}
    data = {}

    MIC_OFFSETS = [(0.023,0.0399), (0.0461,0), (0.0230,-0.0399), (-0.0230,-0.0399), (-0.0461,0), (-0.0230,0.0399)]
    FS = 16000 # sampling frequency
    def __init__(self,set_id, ap_id):
        directory = 'dataset{}/'.format(set_id)
        self.data = pd.read_csv(directory+'{}.csv'.format(ap_id))
        self.data = pd.DataFrame(self.data)
        self.data.columns = ['Mic{}'.format(i) for i in range(0,6)]
        self.mic_locs = get_coords(directory+'config.csv',ap_id,self.MIC_OFFSETS,use_offsets=True)
    def calc_sampling_skew(self,id_1,id_2):
        data1 = np.asarray(self.data['Mic{}'.format(id_1)], dtype=float)
        data2 = np.asarray(self.data['Mic{}'.format(id_2)], dtype=float)
        corr_data = np.flip(np.correlate(data1,data2,'full'))
        return (corr_data.argmax()-(len(data1)-1))

MIC_OFFSETS = [(0.023,0.0399), (0.0461,0), (0.0230,-0.0399), (-0.0230,-0.0399), (-0.0461,0), (-0.0230,0.0399)]
FS = 16000 # sampling frequency
mic_data_folder = 'dataset{}'.format(0)

def aoa_localization(mic_data_folder, FS, MIC_OFFSETS):
    """AoA localization algorithm. Write your code here.

    Args:
        mic_data_folder: name of folder (without a trailing slash) containing 
                         the mic datafiles `0.csv` through `7.csv` and `config.csv`.
        FS: microphone sampling frequency - 16kHz.
        MIC_OFFSETS: a list of tuples of each microphone's location relative to the center of its mic array. 
                     This list is calculated based on the diameter(0.09218m) and geometry of the microphone array.
                     For example, MIC_OFFSETS[1] is [0.09218*0.5, 0]. If the location of microphone array #i is
                     [x_i, y_i]. Then [x_i, y_i] + MIC_OFFSETS[j] yields the absolute location of mic#j of array#i.
                     This is provided for your convenience and you may choose to ignore.

    Returns:
        The user's location in this format (in metres): (L_x, L_y)

    """

    # Your return value should be the user's location in this format (in metres): (L_x, L_y)
    return (0.0, 1.0) 


# ---
# ## Running and Testing
# Use the cell below to run and test your code, and to get an estimate of your grade.

# In[ ]:


def calculate_score(calculated, expected):
    calculated = np.array(calculated)
    expected = np.array(expected)
    distance = np.linalg.norm(calculated - expected, ord=2)
    score = max(1 - (distance-1)/3, 0)
    return min(score, 1)
"""
if __name__ == '__main__':
    test_folder_user_1 = 'user1_data'
    test_folder_user_2 = 'user2_data'
    groundtruth = [(3.0, 1.0), (4.0, 1.0), (3.0, 1.0), (4.0, 1.0)]
    MIC_OFFSETS = [(0.023,0.0399), (0.0461,0), (0.0230,-0.0399), (-0.0230,-0.0399), (-0.0461,0), (-0.0230,0.0399)]
    FS = 16000 # sampling frequency
    
    output = [['Dataset', 'Expected Output', 'Your Output', 'Grade', 'Points Awarded']]
    for i in range(4):
        directory_name = 'dataset{}'.format(i)
        student_loc = aoa_localization(directory_name, FS, MIC_OFFSETS)
        score = calculate_score(student_loc, groundtruth[i])    
        output.append([
            str(i),
            str(groundtruth[i]), 
            str(student_loc), 
            "{:2.2f}%".format(score * 100),
            "{:1.2f} / 5.0".format(score * 5),
        ])

    output.append([
        '<i>👻 Hidden test 1 👻</i>', 
        '<i>???</i>', 
        '<i>???</i>', 
        '<i>???</i>', 
        "<i>???</i> / 10.0"])
    output.append([
        '<i>...</i>', 
        '<i>...</i>', 
        '<i>...</i>', 
        '<i>...</i>', 
        "<i>...</i>"])
    output.append([
        '<i>👻 Hidden test 6 👻</i>', 
        '<i>???</i>', 
        '<i>???</i>', 
        '<i>???</i>', 
        "<i>???</i> / 10.0"])
    display_table(output)
"""

# ---
# ## Rubric
# You will be graded on the four datasets provided to you (5 points each) and six additional datasets under different settings(10 points each). Make sure you are not over-fitting to the provided data. We will use the same code from the **Running and Testing** section above to grade all 10 traces of data. You will be graded on the distance between your calculated user location and ground truth. An error of upto $1 \text{ m}$ is tolerated (and still awarded 100% of the grade). An error of $4 \text{ m}$ or above will be awarded a 0 grade. Grades for errors between $1 \text{ m}$ and $4 \text{ m}$ will be scaled proportionally.

# ---
# ## Submission Guidlines
# This Jupyter notebook (`MP3.ipynb`) is the only file you need to submit on Gradescope. As mentioned earlier, you will only be graded using your implementation of the `aoa_localization` function, which should only return the calculated **NOT** output any plots or data. If you are working in a pair, make sure your partner is correctly added on Gradescope and that both of your names are filled in at the top of this file.
# 
# **Make sure any code you added to this notebook, except for import statements, is either in a function or guarded by `__main__`(which won't be run by the autograder). Gradescope will give you immediate feedback using the provided test cases. It is your responsibility to check the output before the deadline to ensure your submission runs with the autograder.**

# In[ ]:




