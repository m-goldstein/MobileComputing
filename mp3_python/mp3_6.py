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
        self.get_mic_locs()
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


    def gradient_eq(self,pos):
        keys = [key for key in self.data.keys()]
        est = []
        for i in range(len(keys)):
            key = keys[i]
            est.append([1.0])
            for j in range(len(keys)):
                est[i] *= self.aoa[key][int(to_deg(self.peaks[keys[j]][0]))]
        return est
    def estimate_signal(self,sig,lag):
        _sig = np.fft.ifft(np.fft.fft(sig)*np.exp((-1j*2*np.pi*lag)/len(sig)))
        return _sig
    def grid_search(self):
        keys = [key for key in self.data.keys()]
        x_min = min(e for e in [min(e[0] for e in self.mic_locs[key]) for key in keys]) 
        y_min = min(e for e in [min(e[1] for e in self.mic_locs[key]) for key in keys])
        x_lim = max(e for e in [max(e[0] for e in self.mic_locs[key]) for key in keys]) 
        y_lim = max(e for e in [max(e[1] for e in self.mic_locs[key]) for key in keys])
        best = 0#(x_min,y_min)
        history = []
        alt_history = []
        print('searching: x:= {} to {} and y:= {} to {}'.format(x_min,x_lim,y_min,y_lim))
        corrs = {}
        for key in keys:
            corrs[key] = np.corrcoef([self.data[key][i] for i in self.data[key]])
        for x in np.arange(x_min,x_lim,0.25, dtype=np.float64):
            for y in np.arange(y_min,y_lim,0.25,dtype=np.float64):
                est = 1.0
                for k in range(len(keys)):
                    key = keys[k]
                    dx = self.ap_locs[k][0]+x
                    dy = self.ap_locs[k][1]+y
                    est_theta = np.arctan2(dy,dx)
                    
                    est *= self.music_spectrum[key][int(np.ceil(to_deg(np.pi+est_theta)))%360]
                history.append((est,(x,y)))
        return heapq.nlargest(len(history),history, key=lambda y: y[0])

    def grad_(self, pos, target):
        if target[0] > pos[0]:
            dx = np.power(target[0]-pos[0],2.0)
        elif target[0] < pos[0]:
            dx = -np.power(target[0]-pos[0],2.0)
        else:
            dx = 0
        if target[1] > pos[1]:
            dy = np.power(target[1]-pos[1],2.0)
        elif target[1] > pos[1]:
            dy = -np.power(target[1]-pos[1],2.0)
        else:
            dy = 0 
        return (-dx,-dy)
    
    def gradient_descent(self, gradient, start, lr, n, thresh):
        keys = [key for key in self.data.keys()]
        guess = start
        x_min = min(e for e in [min(e[0] for e in self.mic_locs[key]) for key in keys]) 
        y_min = min(e for e in [min(e[1] for e in self.mic_locs[key]) for key in keys])
        x_lim = max(e for e in [max(e[0] for e in self.mic_locs[key]) for key in keys]) 
        y_lim = max(e for e in [max(e[1] for e in self.mic_locs[key]) for key in keys])
        corrs = {}
        for key in keys:
            corrs[key] = np.corrcoef([self.data[key][i] for i in self.data[key]])
        for j in range(n):
            dx = 0
            dy = 0
            for key in keys:
                for i in range(0,6):
                    dx += corrs[key][0][i]*(-lr*gradient(guess, self.mic_locs[key][i])[0])#self.mic_locs[key][i])[0])
                    dy += corrs[key][0][i]*(-lr*gradient(guess, self.mic_locs[key][i])[1])#self.mic_locs[key][i])[1])
                dx /= 6
                dy /= 6    
                if np.all(np.abs(abs(dx)+abs(dy)) < thresh):
                    break
                
                guess = ((guess[0]+dx),(guess[1]+dy))
        return guess
# In[10]:
    
    def calc_corrcoefs(self):
        keys = [key for key in self.data.keys()]
        corr_ = {}
        corr_coefs_ = {}
        eigs_ = {}
        eigvals_ = {}
        for key in keys:
            corr_[key] = np.matrix([self.data[key][i] for i in self.data[key]])
            corr_coefs_[key] = np.cov(corr_[key])
            eigvals_[key],eigs_[key] = ln.eig(corr_coefs_[key])
        self.corr = corr_
        self.corr_coefs = corr_coefs_
        self.eigvals = eigvals_
        self.eigs    = eigs_
    
    def preprocess_eigs(self):
        En_ = {}
        Es_ = {}
        Vn_ = {}
        Vs_ = {}
        keys = [key for key in self.data.keys()]
        for key in keys:
            En_[key] = [self.eigvals[key][self.eigvals[key].argmin()]]
            Es_[key] = [self.eigvals[key][i] for i in range(len(self.eigvals[key])) if i != self.eigvals[key].argmin()]
            Vn_[key] = self.eigs[key][:4,5]
            Vs_[key] = self.eigs[key][:,0:4]
        self.En = En_
        self.Es = Es_
        self.Vn = Vn_
        self.Vs = Vs_
    
    def calc_steering(self):
        keys = [key for key in self.data.keys()]
        s = []
        thetas = np.asarray([theta for theta in np.arange(-180,180)])
        wavelength = float(343)/self.FS
        beta = (-1j*2*np.pi)/wavelength
        for i in range(0,6):
            s.append((1/1)*np.matrix([np.exp(beta*self.mic_dists[i]*np.cos(to_rad(theta))) for theta in thetas]))
        self.steering = np.array([j for j in s]).reshape(6,360).T
    
    def est_AoA(self):
        keys = [key for key in self.data.keys()]
        aoa_ = {}
        spectrum_ = {}
        for key in keys:
            spectrum_[key] = (self.steering@np.matrix([self.data[key][i] for i in range(len(self.data[key]))]))
            aoa_[key] = np.asarray([1/ln.norm(spectrum_[key][i].T @ (self.En[key] * np.conj(self.En[key])) @ spectrum_[key][i].T) for i in range(0,360)])
            aoa_[key]=  np.asarray([e for e in aoa_[key]]) 
        self.aoa = aoa_
        self.spectrum = spectrum_
    
    def est_peaks(self):
        keys = [key for key in self.data.keys()]
        peaks_ = {}
        for key in keys:
            peaks_[key] = find_peaks([np.absolute(e) for e in self.aoa[key]],distance=360)[0]
            peaks_[key] = [to_rad(e) for e in peaks_[key]]
        self.peaks = peaks_
    
    def calc_lag(self, data1, data2):
        corr = signal.correlate(data1,data2, 'full')
        lag = signal.correlation_lags(data1.size,data2.size, 'full')
        return lag[np.argmax(corr)]

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
    def resp_vec(self, src, phi):
        return np.exp(1j*.5*np.pi*src*np.cos(phi))/np.sqrt(src.shape)
    def music(self):
        keys = [key for key in self.data.keys()]
        thetas = np.asarray([to_rad(e) for e in np.arange(-180,180)])
        ps = np.zeros(360)
        spectrum_ = {}
        angles = {}
        for key in keys:
            covdata = np.cov([self.data[key][i] for i in self.data[key]])
            
            for i in range(360):
                tr,Vn = ln.eig(covdata)
                Vn = Vn[:,4:6]
                a = self.resp_vec(np.asarray(self.mic_locs[key]), np.asarray(thetas[i]))
                ps[i] = 1/ln.norm(((Vn.conj().T)@a))
            pB = np.log10(10*ps/ps.min())
            peaks,_ = find_peaks(pB)
            spectrum_[key] = ps
            angles[key] = peaks
        self.music = angles
        self.music_spectrum = spectrum_
            
# In[10]:
            
def main(mic_data_folder):
    ap = AP(mic_data_folder, FS, MIC_OFFSETS)
    ap.calc_corrcoefs()
    ap.calc_steering()
    ap.preprocess_eigs()
    ap.est_AoA()
    ap.est_peaks()
    ap.get_mic_locs()
    ap.calc_skews()
    ap.music()
    coords = ap.grid_search()[0][1]
    print('got: {}'.format(coords))
    #coord = ap.gradient_descent(ap.grad_, coords, 0.2, 50, 1e-6)
    #print('deciding: {}'.format(coord))
    #coords = [e[1] for e in coords]
    #coord = (sum(e[0] for e in coords)/len(coords), sum(e[1] for e in coords)/len(coords))
    return coords

# Your return value should be the user's location in this format (in metres): (L_x, L_y)
ap = AP('dataset0', FS, MIC_OFFSETS)
ap.calc_corrcoefs()
ap.calc_steering()
ap.preprocess_eigs()
ap.est_AoA()
ap.est_peaks()
ap.get_mic_locs()
ap.calc_skews()
ap.music()

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

    ap = AP(mic_data_folder, FS, MIC_OFFSETS)
    ap.calc_steering_vector()
    ap.calc_Rxx()
    ap.calc_power()
    ap.est_thetas()
    ap.get_mic_locs()
    ap.calc_skews()
    ap.calc_time_lags()
    ap.sort_and_rank()
    coord = ap.grid_search(ap.gradient_fn, 4,0.45, 1000, 1e-6, True)
    return (coord[0], coord[1]) 


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




