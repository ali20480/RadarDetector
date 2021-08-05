"""
ROC curves for our proposed detector
Ali Safa - imec & KU Leuven
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

plt.close('all')

"""
Choose detector type, parameters, alpha(Pfa),...
"""
NTT = 10 #number of training cells
NGG = 5 #number of guard cells
alphaOS = np.linspace(2.25, 3, 20) #Threshold multiplicator for OS-CFAR 
order = 0.7 #0.8*2*Ntrain is the ordered statistics that is selected
alphaAD = np.linspace(0.7, 0.95555556, 20) #np.linspace(6, 120, 81) #Threshold multiplicator for our proposed method 
Nbr_range_bin = 254 
r_max = 5 #maximal difference in number of bins before being marked as false alarm
#Feature space dimension D
D = 9
detector_l = ['OS','AD'] #coose between 'OS' and 'AD' (our method)
"""
Load labelled data set
"""
with open('dataset/all_labels.npy', 'rb') as f:
    label_idx = np.load(f,allow_pickle = True)
    
with open('dataset/all_RP.npy', 'rb') as f:
    RP = np.load(f, allow_pickle = True)

"""
All function used within the script
"""
  
#----- Ordered Statistics CFAR

def OS_CFAR_1D(gard_sz, train_sz, alpha, inpt, k):   
    to_ret = np.zeros(inpt.shape)        
    noise_powers = np.zeros(inpt.shape[0])
    inptt = np.array(np.concatenate((inpt[::-1],inpt[:],inpt[::-1])))  #mirror borders for convolution    
    for j in range(inpt.shape[0]):       
        start = inpt.shape[0] + j - (gard_sz + train_sz)      
        stop = inpt.shape[0] + j + (1 + gard_sz + train_sz)      
        lag = inptt[start:start + train_sz]       
        lead = inptt[stop-train_sz:stop]
        to_class = np.append(lag, lead)     
        noise_powers[j] = np.sort(to_class)[int(k*(2*train_sz-1))]
            
    thresholds = alpha * noise_powers   
    selector = (inpt[:] - thresholds > 0).astype(float)   
    to_ret[:] = selector.astype(int)        
    return to_ret

#------ Compute distance matrix

def dist_mat(label, detection, limit, max_dist):
    dist_mat = np.zeros((label.shape[0], detection.shape[0]))
    nbr_fa = 0
    for j in range(detection.shape[0]):
        flag = 0
        for i in range(label.shape[0]):
            d = np.sqrt((label[i] - detection[j])**2)
            if d > limit:
                dist_mat[i,j] = max_dist
            else:
                dist_mat[i,j] = d
                flag = 1
        if flag == 0:
            nbr_fa += 1
                
    return dist_mat, nbr_fa

#-----Creates a CFAR-like connection matrix and weights used at initialization

def detector_conn_array(train_sz, guard_sz, Nbr_N):   
    kernel = np.ones(train_sz*2 + guard_sz*2 + 1)
    kernel[train_sz:train_sz + 2*guard_sz + 1] = 0
    kernel[train_sz + guard_sz] = 0
    kernel_con = kernel.astype(bool) #only connections
    kernel *= (1/(train_sz*2))
    kernel[train_sz + guard_sz] = 0
    conn_mat = np.zeros((Nbr_N, Nbr_N + 2 * (train_sz + guard_sz)), dtype = bool)
    weight_mat = np.zeros((Nbr_N, Nbr_N + 2 * (train_sz + guard_sz)))
    for i in range(Nbr_N):
        b_low = i
        b_high = i + kernel.shape[0]
        conn_mat[i,b_low:b_high] = kernel_con
        weight_mat[i,b_low:b_high] = -kernel
        
    conn_mat = conn_mat[:,train_sz + guard_sz:train_sz + guard_sz + Nbr_N].T 
    weight_mat = weight_mat[:,train_sz + guard_sz:train_sz + guard_sz + Nbr_N].T 
    return conn_mat, weight_mat

#----- adapt detector weights locally

def Adapt_weight(norm_range_mean, weights, train_sz, guard_sz):
    flipped_range = norm_range_mean
    weights = np.zeros(weights.shape)
    for i in range(weights.shape[1]):
        kernel = -weights[np.maximum(-(train_sz + guard_sz)+i, 0):np.minimum(i+(train_sz + guard_sz + 1), weights.shape[1]),i]
        flip = flipped_range[np.maximum(-(train_sz + guard_sz)+i, 0):np.minimum(i+(train_sz + guard_sz + 1), weights.shape[1])]
        flip = 1 - flip/np.nanmax(flip)
        kernel = flip 
        kernel[train_sz-np.maximum((train_sz-i), 0):(train_sz+guard_sz+1-np.maximum((train_sz-i), 0))] = 0
        kernel /= np.nansum(kernel)
        weights[np.maximum(-(train_sz + guard_sz)+i, 0):np.minimum(i+(train_sz + guard_sz + 1), weights.shape[0]),i] = np.maximum(kernel, 0)
    
    weights[weights < 0] = 0
        
    return -weights

#------ execute our proposed detector

def AD_detect(inpt, theta, principal_weight, lateral_weight):
    V = np.zeros(inpt.shape[0])
    S = np.zeros(inpt.shape)
    for n in range(inpt.shape[1]):
        Jp = np.dot(np.eye(inpt.shape[0]), inpt[:, n])
        Jm = np.dot(lateral_weight, inpt[:, n])
        V = Jp + Jm
        idx = np.argwhere(V > theta)[:,0]
        S[idx,n] = 1 #event
           
    return S

def AD_detect2(inpt, theta, principal_weight, lateral_weight):
    V = np.zeros(inpt.shape[0])
    S = np.zeros(inpt.shape)
    Vcoll = np.zeros((inpt.shape[0], inpt.shape[1]))
    for n in range(inpt.shape[1]):
        Jp = np.dot(np.eye(inpt.shape[0]), inpt[:, n])
        Jm = np.dot(lateral_weight, inpt[:, n])
        V = Jp + Jm
        idx = np.argwhere(V > theta)[:,0]
        Vcoll[:,n] = V
        S[idx,n] = 1 #event
           
    return S, Vcoll

#init neural pool weights (not really needed)
conn_mat1, weight_conn1 = detector_conn_array(NTT, NGG, Nbr_range_bin) 
#init neural pool weights (not really needed)
principal_weights = np.eye(Nbr_range_bin) 
    
"""
Assess performance
"""
PFA_MAT = np.zeros((len(detector_l), alphaOS.shape[0]))
PD_MAT = np.zeros((len(detector_l), alphaOS.shape[0]))
Vcollcoll = []
for det in range(len(detector_l)):
    detector = detector_l[det]
    detector1_train = NTT
    detector1_guard = NGG
    for k in range(alphaOS.shape[0]): 
        if detector == "OS":
            theta = alphaOS[k]
        elif detector == "AD":
            theta = alphaAD[k]
            
        Nfa = 0
        collection = []
        Pdet = 0
        for frame in range(label_idx.shape[0]):
            print("Frame " + str(frame) + "/" + str(label_idx.shape[0]) + " starting")
            #Normalized Range profiles and quantized versions
            range_profile = RP[frame,:]
            gt_idx = label_idx[frame].astype(int)
            #Compute Dirac train periods
            if detector == "OS":
                detections = OS_CFAR_1D(detector1_guard, detector1_train, theta, range_profile, order)
            elif detector == "AD":
                period = np.round((1 - range_profile)*D).astype(int) + 1
                #Non-linear projection of the input to the feature space of dimension D = spike_start_time - spike_end_time
                events = []
                for i in range(Nbr_range_bin):
                    if period[i] == D:
                        locs = np.array([]).astype(int)
                    else:
                        locs = np.arange(0, D, period[i])
                        locs = np.round(locs[1:]).astype(int)
                        
                    etrain = np.zeros(D)
                    if period[i] < D:
                        etrain[period[i]] = 1
                    events.append(etrain)
                    
                events = np.array(events) 
                #Get modified maximum likelihood coefficients for feature pooling and roll the detection
                weight_conn1 = Adapt_weight(range_profile, weight_conn1, detector1_train, detector1_guard) #learn before inference
                detections = AD_detect(events, theta, principal_weights.T, weight_conn1.T)
                detections, Vcoll = AD_detect2(events, theta, principal_weights.T, weight_conn1.T)
                Vcollcoll.append(Vcoll)
    
            #get the detected indexes   
            detected_idx = np.unique(np.argwhere(detections == 1)[:,0])
            #save some examples for plotting later on
            if np.mod(frame, 25) == 0:
                collection.append(detected_idx)
            
            #compute Nfa, Pdetect and performance cost
            if detected_idx.shape[0] > 0:
                #---- assignement cost
                D_mat, nbr_fa = dist_mat(gt_idx, detected_idx, r_max, range_profile.shape[0])
                Nfa += nbr_fa / label_idx.shape[0]
                row_ind, col_ind = linear_sum_assignment(D_mat)
                nbr_detec = np.argwhere(D_mat[row_ind, col_ind] <= 5)[:,0].shape[0]
                p_detect = nbr_detec / (label_idx.shape[0] * gt_idx.shape[0])
                Pdet += p_detect
                
        
        PFA_MAT[det,k] = Nfa/Nbr_range_bin   
        PD_MAT[det,k] = Pdet
        print("Pd: " + str(Pdet)) 
        print("Average Pfa: " + str(Nfa/Nbr_range_bin)) 

  
"""
Plots
"""
plt.figure(1)  
for i in range(len(detector_l)):
    plt.plot(PFA_MAT[i,:], PD_MAT[i,:], '.-')
    plt.grid('on')
    plt.xlabel('$P_{FA}$')
    plt.ylabel('$P_{D}$')


plt.legend(['OS-CFAR', 'Proposed Detector'])

