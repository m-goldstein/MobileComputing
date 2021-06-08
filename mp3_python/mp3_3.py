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

# In[4]:


import numpy as np
import pandas as pd

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

"""
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

# In[5]:

"""
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

# In[9]:
from scipy import signal
from sklearn.preprocessing import RobustScaler
import scipy
import math
from scipy.signal import find_peaks
from scipy.optimize import minimize
import numpy.linalg as ln
import heapq
MIC_OFFSETS = [(0.023,0.0399), (0.0461,0), (0.0230,-0.0399), (-0.0230,-0.0399), (-0.0461,0), (-0.0230,0.0399)]
FS = 16000 # sampling frequency
def dist(x,y):
    a1 = np.power(x[0]-y[0],2.0)
    a2 = np.power(x[1]-y[1],2.0)
    return np.sqrt(a1+a2)
def to_rad(deg):
    return (np.pi/180.0)*deg
def to_deg(rad):
    return (180.0/np.pi)*rad
class AP:
    skews = {}
    mic_dists = None
    mic_thetas = None
    data = {}
    ap_locs = None
    mic_locs = {}
    MIC_OFFSETS = None
    FS = -1
    
    def get_ap_locs(self, data_folder):
        csvdata = np.asarray(pd.read_csv(data_folder+'/config.csv',header=None))
        return csvdata 
    
    def get_mic_locs(self):
        keys = [key for key in self.data.keys()]
        for key in range(len(keys)):
            base_loc = self.ap_locs[key]
            self.mic_locs[keys[key]] = []
            for i in range(len(self.MIC_OFFSETS)):
                e = tuple(((base_loc[0]+self.MIC_OFFSETS[i][0]),(base_loc[1]+self.MIC_OFFSETS[i][1])))
                self.mic_locs[keys[key]].append(e)

    def __init__(self, data_folder, FS, MIC_OFFSETS):
        self.ap_locs = self.get_ap_locs(data_folder)
        self.MIC_OFFSETS = MIC_OFFSETS
        self.FS = FS
        self.mic_thetas = []
        self.steering = []
        self.power = []
        self.mic_dists = []
        self.lags = []
        for i in range(len(self.MIC_OFFSETS)):
            self.mic_dists.append(dist(self.MIC_OFFSETS[0],self.MIC_OFFSETS[i]))
            if (self.MIC_OFFSETS[0][0]-self.MIC_OFFSETS[i][0]) == 0:
                self.mic_thetas.append(0.0)
            else:
                self.mic_thetas.append(np.arctan((self.MIC_OFFSETS[0][1]-self.MIC_OFFSETS[i][1])/(self.MIC_OFFSETS[0][0]-self.MIC_OFFSETS[i][0])))
        for i in range(8):
            csvdata = pd.read_csv(data_folder+'/{}.csv'.format(i),header=None)
            keys = [key for key in csvdata.keys()]
            ap_data = {}
            for k in keys:
                ap_data[k] = csvdata[k]
                #tmp = np.zeros((6,24000))
                #tmp[:6, :24000] = [j for j in ap_data[k]]
            self.data['AP{}'.format(i)] = ap_data#np.matrix(ap_data)
    def calc_Rxx(self):
        Rxx_ = {}
        Rss_ = {}
        eigs_ = {}
        keys = [key for key in self.data.keys()]
        for key in keys:
            Rss_[key] = []
            for i in range(0,6):
                Rss_[key].append(np.matrix(self.steering).T[0].T@np.matrix(self.data[key][i]))
        self.Rss = Rss_
        for key in keys:
            Rxx_[key] = []
            eigs_[key] = []
            for i in range(0,6):
                Rxx_[key].append(self.Rss[key][i] @ np.conjugate(self.Rss[key][i]).T)
                eigs_[key].append(ln.eigvals(Rxx_[key][i])[0])
        self.Rxx = Rxx_
        self.eigs = eigs_
    def calc_skews(self):
        skewz = {}
        keys = [key for key in self.data.keys()]
        for key in keys:
            skewz[key] = []
            for i in range(0,6):
                lag = self.calc_lag(self.data[key][0],self.data[key][i])
                t = float(lag/self.FS)
                skewz[key].append(float(343)*t)
        self.skews = (skewz)
        return skewz
    def calc_steering_vector(self):
        steering_ = [[]]*6
        keys = [key for key in self.data.keys()]
        thetas = np.asarray([theta for theta in np.arange(0,360)])
        wavelength = float(343)/self.FS
        beta = (-1j*2*np.pi)/wavelength
        for i in range(0,6):
            steering_[i] = (1/1)*np.matrix([np.exp(self.mic_dists[i]*beta)*np.exp(-1j*(i*np.pi*wavelength)*np.cos(to_rad(theta))) for theta in thetas])
        tmp = np.zeros((6,360))
        #tmp[:6,:180] = steering_[i]
        #tmp[:6,:180] = [np.matrix(j).reshape(-1,180) for j in steering_]
        self.steering = np.array([j for j in steering_]).reshape(6,360).T
        return steering_
    def calc_power(self):
        power_ = {}
        keys = [key for key in self.data.keys()]
        for key in keys:
            power_[key] = []
            for i in range(0,6):
                for theta in range(0,360):
                    #x = np.asarray(self.steering[i].getH())[theta][0]

                    x = np.asarray(self.steering.T)[i][theta]
                    y = np.asarray(np.matrix(self.eigs[key][i]) * np.conjugate(np.matrix(self.eigs[key][i])))
                    z = np.asarray(self.steering.T)[i][theta]
                    power_[key].append(np.asarray(1/(x * y * z)))
        for key in keys:
            power_[key] = np.asarray([(e)[0] for e in power_[key]]).T[0]
        self.power = power_
        return power_
    def est_thetas(self):
        est_thetaz = {}
        keys = [key for key in self.data.keys()]
        for key in keys:
            peaks = find_peaks([(e*np.conjugate(e)).real for e in self.power[key]],height=None,distance=360)[0]
            est_thetaz[key] = []
            est_thetaz[key].append(0)
            if (len(peaks) == 4):
                for e in peaks:
                    est_thetaz[key].append(e)
                est_thetaz[key].append(2160-peaks[-1])
            elif (len(peaks) == 5):
                for e in peaks:
                    est_thetaz[key].append(e)#[e for e in peaks])
            else:
                print('???')
            #est_thetaz[key].append(peaks[0])
            #for j in range(0, len(peaks)-1):
            #    est_thetaz[key].append(peaks[j+1]-peaks[j])

            #est_thetaz[key].append(peaks[-1])
            est_thetaz[key] = [to_rad(e) for e in est_thetaz[key]]
        self.est_thetas = est_thetaz
        return est_thetaz
    def calc_lag(self, signal_a,signal_b):
        corr = signal.correlate(signal_a, signal_b, mode='full')
        corr_lags = signal.correlation_lags(signal_a.size, signal_b.size, mode='full')
        lag = corr_lags[np.argmax(corr)]
        return lag
    def calc_time_lags(self):
        lags = {}
        keys= [key for key in self.data.keys()]
        for key in keys:
            lags[key] = [(self.skews[key][i]*np.cos(self.est_thetas[key][i]))/float(343.0) for i in range(0,6)]
        self.lags = lags
        return lags

    def sort_and_rank(self):
        keys = [key for key in self.lags.keys()]
        copy = dict(self.lags)
        ranks = {}
        for key in keys:
            copy[key].sort()
            ranks[key] = []
            for i,j in enumerate(self.lags[key]):
                ranks[key].append(copy[key].index(j))
        self.rank = ranks
        return ranks
    def gradient_fn(self, point, target):
        dx = 0.0
        dy = 0.0
        if target[0] > point[0]:
            dx = np.power(target[0]-point[0],2.0)#1/(1*np.sqrt(1*abs(target[0]-point[0])))
        else:
            dx = -np.power(target[0]-point[0],2.0)#-1/(1*np.sqrt(1*abs(target[0]-point[0])))
        if target[1] > point[1]:
            dy = np.power(target[1]-point[1],2.0)#1/(1*np.sqrt(1*abs(target[1]-point[1])))
        else:
            dy = -np.power(target[1]-point[1],2.0)#-1/(1*np.sqrt(1*abs(target[1]-point[1])))
        return (-dx,-dy)

    def grid_search(self,gradient,lr,n_iter,thresh):
        keys = [key for key in self.data.keys()]
        x_min = min(e for e in [min(e[0] for e in self.mic_locs[key]) for key in keys]) 
        y_min = min(e for e in [min(e[1] for e in self.mic_locs[key]) for key in keys])
        x_lim = max(e for e in [max(e[0] for e in self.mic_locs[key]) for key in keys]) 
        y_lim = max(e for e in [max(e[1] for e in self.mic_locs[key]) for key in keys])
        best = (x_min,y_min)
        history = []
        #for x in np.arange(x_min,x_lim, lr, dtype=np.float64):
        #    for y in np.arange(y_min, y_lim, lr, dtype=np.float64):
        for j in range(n_iter):
            estim = 1.0
            diff_x = 0.0
            diff_y = 0.0
            for k in range(len(keys)):
                key = keys[k]
                #skews = [abs(e) for e in self.skews[key]]
                #min_skew = min([e for e in skews if e != 0.0])
                #min_idx = skews.index(min_skew)
                for i in range(0,6):
                    min_idx= i
                    guess_x = best[0]+(0.1)*np.cos(self.est_thetas[key][i])
                    guess_y = best[1]+(0.1)*np.sin(self.est_thetas[key][i])
                    diff_x  += (-lr*gradient((guess_x,guess_y), self.mic_locs[key][min_idx])[0])
                    diff_y  += (-lr*gradient((guess_x,guess_y), self.mic_locs[key][min_idx])[1])
            diff_x /= 48
            diff_y /= 48
            if np.all(np.abs(diff_x) <= thresh) and np.all(np.abs(diff_y) <= thresh):
                print('guess was: {}'.format((guess_x,guess_y)))
                print('coords: {}'.format((best)))
                #best = #(x,y)
                break
            print('guess was: {}'.format((guess_x, guess_y)))
            print('adjustment: {}'.format((diff_x,diff_y)))
            print('best: {}'.format(best))
            #z = input()
            best = (best[0]+diff_x, best[1]+diff_y)
        return best
# In[10]:
            
def main(mic_data_folder):
    ap = AP(mic_data_folder, FS, MIC_OFFSETS)
    ap.get_mic_locs()
    ap.calc_skews()
    ap.calc_steering_vector()
    ap.calc_power()
    return ap
# Your return value should be the user's location in this format (in metres): (L_x, L_y)

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

        
    return (0.0, 1.0) 


# ---
# ## Running and Testing
# Use the cell below to run and test your code, and to get an estimate of your grade.

# In[11]:


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




