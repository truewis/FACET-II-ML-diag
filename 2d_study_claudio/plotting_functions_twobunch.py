#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:13:43 2018

@author: cemma
"""

# Plotting functions for Neural network
# Plotting the pv data from scandata structure 
def plot_pv_data_v2(X):
    import matplotlib.pyplot as plt
    plt.subplot(241)
    plt.plot(X[:,2])
    plt.ylabel('L1S amp')

    plt.subplot(242)
    plt.plot(X[:,0])
    plt.ylabel('L1S phase [deg]')    

    plt.subplot(243)
    plt.plot(X[:,5])
    plt.ylabel('BC11 current [kA]')    

    plt.subplot(244)
    plt.plot(X[:,6])
    plt.ylabel('BC14 current [kA]')    

    plt.subplot(245)
    plt.plot(X[:,1])
    plt.ylabel('L2 phase [deg]')    

    plt.subplot(246)
    plt.plot(X[:,3])
    plt.ylabel('L2 amp')

    plt.subplot(247)
    plt.plot(X[:,7])
    plt.ylabel('IP current [kA]')
    
    plt.tight_layout()
    plt.show()
    
def plot_model_history(history):
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    # summarize history for loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.show()
# Plotting the pv data from scandata structure 
def plot_training_data(X):
    import matplotlib.pyplot as plt
    plt.subplot(241)
    plt.plot(X[:,2])
    plt.ylabel('L1S amp')

    plt.subplot(242)
    plt.plot(X[:,0])
    plt.ylabel('L1S phase [deg]')    

    plt.subplot(243)
    plt.plot(X[:,4])
    plt.ylabel('BC11 current [kA]')    

    plt.subplot(244)
    plt.plot(X[:,5])
    plt.ylabel('BC14 current [kA]')    

    plt.subplot(245)
    plt.plot(X[:,1])
    plt.ylabel('L2 phase [deg]')    

    plt.subplot(246)
    plt.plot(X[:,3])
    plt.ylabel('L2 amp')

    plt.subplot(247)
    plt.plot(X[:,8])
    plt.ylabel('IP current [kA]')
    
    plt.tight_layout()
    plt.show()

# Plotting the predicted vs actual current profiles    
def plot_pred_vs_actual_v2(tztrain,Iz_scaled,predict_Iz):
    import matplotlib.pyplot as plt
    import numpy as np
    f,axarr = plt.subplots(2,3)
    for i in range(6):        
        ns = int(Iz_scaled[:,0].shape+Iz_scaled[:,0].shape*(0.5*(2*np.random.rand(1,1))-1));                
        curr_integral = np.trapz(Iz_scaled[ns,:],x=tztrain[ns,:]*1e-15);
        conv_factor = 180e-12/curr_integral*1e-3;
        if i<3:
            axarr[0,i].plot(tztrain[ns,:],Iz_scaled[ns,:]*conv_factor,label='Actual');
            axarr[0,i].plot(tztrain[ns,:],predict_Iz[ns,:]*conv_factor,'r--',label='Predicted');

        else:
            axarr[1,i-3].plot(tztrain[ns,:],Iz_scaled[ns,:]*conv_factor,label='Actual');
            axarr[1,i-3].plot(tztrain[ns,:],predict_Iz[ns,:]*conv_factor,'r--',label='Predicted');
 
        for ax in axarr.flat:
            ax.set(xlabel='t [fs]', ylabel='Current [kA]')
    f.tight_layout()            
    plt.show()
    
def plot_lps_vs_prediction_contour(lps,predicted_lps,X,Y):
    import matplotlib.pyplot as plt
    import numpy as np    
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    fig, (ax, ax2) = plt.subplots(1,2)
    ax.contourf(X,Y,lps[:,:,ns],cmap = plt.cm.viridis)
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax2.contourf(X,Y,predicted_lps[:,:,ns],cmap = plt.cm.viridis)
    ax2.set_xlabel('Time [fs]')
    plt.show()
    
def plot_lps_vs_prediction_v2(lps,predicted_lps,x,y):
    import matplotlib.pyplot as plt
    import numpy as np    
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    fig, (ax, ax2) = plt.subplots(1,2)
    ax.imshow(lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),interpolation = 'none')
    ax.set_aspect(3)
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax2.imshow(predicted_lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]))
    ax2.set_xlabel('Time [fs]')
    ax2.set_aspect(3)
    plt.show()
    
def plot_lps_and_current(lps,x,y,tztrain,Iz_scaled):
    import matplotlib.pyplot as plt
    import numpy as np    
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    curr_integral = np.trapz(Iz_scaled[ns,:],x=tztrain[ns,:]*1e-15);
    conv_factor = 180e-12/curr_integral*1e-3;
    fig, (ax, ax2) = plt.subplots(1,2,figsize=(10,3))
    ax.imshow(lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),aspect = "auto")
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax2.plot(tztrain[ns,:],Iz_scaled[ns,:]*conv_factor)
    ax2.set_xlabel('Time [fs]')
    ax2.set_ylabel('Current [kA]')
    ax2.set_aspect("auto")
    plt.show()
    
def plot_lps_and_current_w_prediction(lps,predicted_lps,x,y,tztrain,Iz_scaled,predict_Iz):
    import matplotlib.pyplot as plt
    import numpy as np    
    ns = int(lps.shape[2]+lps.shape[2]*(0.5*(2*np.random.rand(1,1))-1));        
    curr_integral = np.trapz(Iz_scaled[ns,:],x=tztrain[ns,:]*1e-15);
    conv_factor = 180e-12/curr_integral*1e-3;
    fig, (ax, ax2, ax3) = plt.subplots(1,3,figsize=(10,3))
    ax.imshow(lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),aspect = "auto")
    ax.set_xlabel('Time [fs]')
    ax.set_ylabel('Energy Deviation [MeV]')
    ax3.plot(tztrain[ns,:],Iz_scaled[ns,:]*conv_factor,label='XTCAV')
    ax3.plot(tztrain[ns,:],predict_Iz[ns,:]*conv_factor,label='Predicted')
    ax3.set_xlabel('Time [fs]')
    ax3.set_ylabel('Current [kA]')
    ax2.set_aspect("auto")
    ax2.imshow(predicted_lps[:,:,ns],extent=(x[0],x[x.shape[0]-1],y[0],y[y.shape[0]-1]),aspect = "auto")
    ax2.set_xlabel('Time [fs]')
    ax2.set_aspect("auto")
    plt.show()
    
def plot_2bunch_prediction_vs_lucretia(curprof,predicted_curprof,Imax):
    import matplotlib.pyplot as plt
    import numpy as np
    dz = 0.1744;# um per pixel
    tvector = dz*(np.arange(curprof.shape[1]))
    ns = int(curprof[:,0].shape+curprof[:,0].shape*(0.5*(2*np.random.rand(1,1))-1))
    plt.figure()
    plt.plot(tvector,curprof[ns,:]*Imax,label='Simulation');
    plt.plot(tvector,predicted_curprof[ns,:]* Imax,'r--',label='Neural Net');
    plt.xlabel('z [um]')
    plt.ylabel('I [kA]')
    plt.legend()
    plt.show()

def nn_vs_lucretia_goodness_of_fit(Iz_test_scaled,predict_Iz_test,Izmax):    
    import numpy as np
    import matplotlib.pyplot as plt
    pkI_test = np.amax(Iz_test_scaled*Izmax,axis=1);
    ind = np.argsort(pkI_test);
    pkI_sorted = np.amax(Iz_test_scaled[ind]*Izmax,axis=1);
    nn_pkI_sorted = np.amax(predict_Iz_test[ind]*Izmax,axis=1);
    plt.plot(pkI_sorted)
    plt.plot(nn_pkI_sorted)
    #plt.plot(np.sort(np.amax(Iz_test_scaled*np.max(Iz),axis=1)))
    plt.show()
    plt.xlabel('Shot number')
    plt.ylabel('Peak current [kA]')