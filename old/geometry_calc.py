#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:53:30 2018

@author: walther
"""

import gc
import numpy as np
from scipy.interpolate import griddata
import matplotlib.path as mpath

def shband_angle(a,
                 r,
                 p,
                 y,
                 dr,
                 dp,
                 dy,
                 Di=2,
                 phi=np.linspace(0,180,50),
                 shcfg={'radius':108.7,
                        'width':25.4,
                        'axis_offset':-12.7,
                        'diffuser_diameter':22.5}):
    """
    Calculate karthesian coordinates ( x (North), y(West), z (zenith)) of 
    points on the shadowband with a certain,rotation angle and measures.

    Parameters
    ----------
    
    a : float or iterable
        Rotation angle of the shadowband -> [degree]
    r : same type and size as a
        ship roll angle -> [degree] - positive portside up
    p : same type and size as a
        ship pitch angle -> [degree] - positive bow up
    y : same type and size as a
        ship yaw angle -> [degree] - positive North -> East
    dr : float
        setup offset roll angle [degree]
    dp : float
        setup offset pitch angle [degree]
    dy : float
        setup offset yaw angle (yaw_init) [degree]
    Di : int,optional
        count of dot lines on shadowband spanning phi (default:2 (edges))
    phi : iterable (float), optional
        spann the shadowband over a certain angle region (default: np.linspace(0,180,50))
    shcfg: dict,optional
        dict including shadowband measures
        - radius
        - width
        - axis_offset - rotation axis offset on z-axis
        - diffuser_diameter
    
    Returns
    -------
    
    Bt : array
        shape =    (3,len(r),len(a),len(phi),Di) 
        Karthesian coordinates in global system x (North), y(West), z (zenith)    
    """
    gc.enable()
    c=np.pi/180.
    if not np.isscalar(a):
        a=np.array(a,dtype=float)
    if not np.isscalar(r):
        r=np.array(r,dtype=float)
        p=np.array(p,dtype=float)
        y=np.array(y,dtype=float)
    
    a,r,p,y,dr,dp,dy=a*c,r*c,p*c,y*c,dr*c,dp*c,dy*c
    
    ###measures
    Rb=shcfg['radius']
    D=shcfg['width']
    A=shcfg['axis_offset']

    Db=np.linspace(-0.5*D,0.5*D,Di)
    phi=phi*c    
    
    Db,phi=np.meshgrid(Db,phi)
    
    ### band in zero position
    ### B.shape: (3,len(phi),len(Db))
    B=np.array([Rb*np.cos(phi),
                Rb*np.sin(phi),
                Db])
    
    ### band rotated
    if np.isscalar(a):
        M=np.array([[1., 0.       ,  0.      ],
                     [0., np.cos(a),-np.sin(a)],
                     [0., np.sin(a), np.cos(a)]])
        Br=np.tensordot(M,B,1)        ##Br.shape : (3,len(phi),len(Db)) 
    else:
        M=np.array([[np.ones(len(a)) , np.zeros(len(a)) , np.zeros(len(a))],
                    [np.zeros(len(a)), np.cos(a)       , -np.sin(a)],
                    [np.zeros(len(a)), np.sin(a)       , np.cos(a)]])
        Br=np.tensordot(M,B,(1,0))    ##Br.shape : (3,len(a),len(phi),len(Db))   
    Br[2]+=A
    
    ### transformed to ship koordinates
    Mz=np.array([[np.cos(dy) , -np.sin(dy) , 0.], #first roate offset yaw
                 [np.sin(dy) ,  np.cos(dy) , 0.],
                 [    0      ,      0      , 1.]])
    Mx=np.array([[1. ,      0      ,      0.   ],
                 [0. ,  np.cos(dr) , np.sin(dr)],
                 [0. , -np.sin(dr) , np.cos(dr)]])
    My=np.array([[np.cos(dp) ,  0  , np.sin(dp)],
                 [  0        ,  1. ,     0.    ],
                 [-np.sin(dp) , 0. , np.cos(dp)]])
    M=np.matmul(My,np.matmul(Mx,Mz))
    Bs=np.tensordot(M,Br,1) ##Bs.shape : (3,len(a),len(phi),len(Db))  or  (3,len(phi),len(Db)) 
    
    ### transformed to global coordinates (ship correction)
    ##convention : first roll, second pitch, last yaw
    if np.isscalar(r):
        Mx=np.array([[1. ,    0      ,      0.   ], ## positive portside up
                     [0. , np.cos(r) , -np.sin(r)],
                     [0. , np.sin(r) ,  np.cos(r)]])
        My=np.array([[np.cos(p) ,  0  , -np.sin(p)], ##positive bow up
                     [    0     ,  1. ,   0.      ],
                     [np.sin(p) ,  0. ,  np.cos(p)]])
        Mz=np.array([[ np.cos(y) ,  np.sin(y) , 0.], ##positive north to east
                     [-np.sin(y) ,  np.cos(y) , 0.],
                     [    0      ,      0     , 1.]])
        M=np.matmul(Mz,np.matmul(My,Mx))
        Bt=np.tensordot(M,Bs,1) ##Bt.shape : (3,len(a),len(phi),len(Db))  or  (3,len(phi),len(Db)) 
    else:
        Mx=np.array([[np.ones(len(r)) , np.zeros(len(r)) , np.zeros(len(r))], ## positive portside up
                     [np.zeros(len(r)) , np.cos(r) , -np.sin(r)],
                     [np.zeros(len(r)) , np.sin(r) ,  np.cos(r)]])
        My=np.array([[np.cos(p) , np.zeros(len(r))  , -np.sin(p)], ##positive bow up
                     [np.zeros(len(r)), np.ones(len(r)) , np.zeros(len(r)) ],
                     [np.sin(p) , np.zeros(len(r)),  np.cos(p)]])
        Mz=np.array([[ np.cos(y) ,  np.sin(y) ,np.zeros(len(r))], ##positive north to east
                     [-np.sin(y) ,  np.cos(y) ,np.zeros(len(r))],
                     [np.zeros(len(r)) , np.zeros(len(r)), np.ones(len(r))]])
        M1=np.tensordot(My,Mx,((1),(0)))[:,np.arange(len(r)),:,np.arange(len(r))].T
        M=np.tensordot(Mz,M1,((1),(0)))[:,np.arange(len(r)),:,np.arange(len(r))].T
        Bt=np.tensordot(M,Bs,(1,0)) ##Bt.shape : (3,len(r),len(a),len(phi),len(Db)) or  (3,len(r),len(phi),len(Db))        

    ##uniform output dimensions ->Bt.shape : (3,len(r),len(a),len(phi),len(Db))
    if np.isscalar(r) or np.isscalar(a):
        a,r=np.array(a),np.array(r)
        Bt=Bt.reshape((3,len(r),len(a),len(Db[:,0]),len(Db[0,:])))
    return Bt

def check_collision(a,r,p,y,dr,dp,dy,zen,azi,
                 Di=2,
                 phi=np.linspace(0,180,50),
                 shcfg={'radius':108.7,
                        'width':25.4,
                        'axis_offset':-12.7,
                        'diffuser_diameter':22.5}):
    c=np.pi/180.
    gc.enable()
    if not np.isscalar(r):
        if not (len(r)==len(a) and len(r)==len(p) and len(r)==len(y) and len(r)==len(zen) and len(r)==len(azi)):
            raise ValueError("shape missmatch of a,r,p,y,zen,azi -> all should have same length")
    N=len(np.array(r))
    ### solar vector in karthesian coordinates 
    ### szen, sazi .shape = (11,len(r))
    da=np.linspace(-0.25,+0.25,11)
    szen=np.ones((len(da),len(r)))*zen+da[:,np.newaxis]
    sazi=np.ones((len(da),len(r)))*azi+da[:,np.newaxis]
    
    szen,sazi=np.meshgrid(szen,sazi)
    ### szen,sazi, S .shape = (11,11,len(r))
    szen=szen.reshape((len(r),11,11,len(r)))[0,:,:,:]
    sazi=sazi.reshape((len(r),11,11,len(r)))[0,:,:,:]
    
    S=np.array([np.sin(szen*c)*np.cos(sazi*c),
                -np.sin(szen*c)*np.sin(sazi*c),
                np.cos(szen*c)]) 

    ### shadowband coordinates
    Bt=shband_angle(a=a,r=r,p=p,y=y,dr=dr,dp=dp,dy=dy,Di=Di,phi=phi,shcfg=shcfg)

    
    Bt=Bt[:,np.arange(N),np.arange(N),:,:]###Bt.shape(3,N,len(phi),Di)
    
    ### radius vector pointing to edge of sensor
    B0=shband_angle(a=a,r=r,p=p,y=y,dr=dr,dp=dp,dy=dy,Di=Di,shcfg=shcfg,
                    phi=np.array([90]))
    B0=B0[:,np.arange(N),np.arange(N),:,:]###B0.shape(3,N,len(phi),Di)
    
    absrB0=np.sqrt(B0[0,:,0,0]**2+B0[1,:,0,0]**2)
    rB0=np.array([B0[0,:,0,0]/absrB0, #rB0.size=(3,N)
                  B0[1,:,0,0]/absrB0,
                        np.zeros(N)    ])
    So=0.5*22.5*rB0           #So.size=(3,N)


    ### length between sensor and shadowband on different angles
    Bs=shband_angle(r=r,p=p,y=y,dr=dr,dp=dp,dy=dy,shcfg=shcfg,
                    a=np.linspace(0,180,10),
                    Di=3,
                    phi=np.linspace(0,180,10))
    Bs1=np.zeros(Bs.shape[:-1])
    Bs2=np.zeros(Bs.shape[:-1])
    for i in range(3):
        Bs1[i,:,:,:]=(Bs[i,:,:,:,0].T-So[i,:]).T
        Bs2[i,:,:,:]=(Bs[i,:,:,:,2].T+So[i,:]).T
    Bs=Bs[:,:,:,:,1]
    
    ### shapes of rs,zens and azis : (len(r),len(a),len(phi))
    rs=np.sqrt(Bs[0]**2+Bs[1]**2+Bs[2]**2)
    zens=np.arccos(Bs[2]/rs)
    azis=np.arctan2(Bs[1],Bs[0])
    rs1=np.sqrt(Bs1[0]**2+Bs1[1]**2+Bs1[2]**2)
    zens1=np.arccos(Bs1[2]/rs1)
    azis1=np.arctan2(Bs1[1],Bs1[0])
    rs2=np.sqrt(Bs2[0]**2+Bs2[1]**2+Bs2[2]**2)
    zens2=np.arccos(Bs2[2]/rs2)
    azis2=np.arctan2(Bs2[1],Bs2[0])    

   
    #append maximum values at the edges
    R=np.ones((len(np.array(r)),4,10))*(shcfg['radius']+shcfg['axis_offset'])
    Z=np.zeros((len(np.array(r)),4,10))
    A=np.ones((len(np.array(r)),4,10))*np.array([0.,90.,180.,270.])[np.newaxis,:,np.newaxis]
    
    rs=np.concatenate((rs,R),axis=1)
    zens=np.concatenate((zens,Z),axis=1)
    azis=np.concatenate((azis,A),axis=1)
    rs1=np.concatenate((rs1,R),axis=1)
    zens1=np.concatenate((zens1,Z),axis=1)
    azis1=np.concatenate((azis1,A),axis=1)
    rs2=np.concatenate((rs2,R),axis=1)
    zens2=np.concatenate((zens2,Z),axis=1)
    azis2=np.concatenate((azis2,A),axis=1)
    
    ### calculate radius of band at solar angles
    R,R1,R2=[],[],[]
    for i in range(N):
        points=np.vstack((zens[i,:,:].flatten(),azis[i,:,:].flatten())).T
        R.append(griddata(points,rs[i,:,:].flatten(),(zen[i]*c,azi[i]*c)))
        points=np.vstack((zens1[i,:,:].flatten(),azis1[i,:,:].flatten())).T
        R1.append(griddata(points,rs1[i,:,:].flatten(),(zen[i]*c,azi[i]*c)))
        points=np.vstack((zens2[i,:,:].flatten(),azis2[i,:,:].flatten())).T
        R2.append(griddata(points,rs2[i,:,:].flatten(),(zen[i]*c,azi[i]*c)))
    R,R1,R2 =np.array(R),np.array(R1),np.array(R2)

    #print S.shape,R1.shape,So.shape
    S1=np.zeros(S.shape)
    S2=np.zeros(S.shape)
    for i in range(3):
        S1[i,:,:,:]=S[i,:,:,:]*R1-So[i,:]
        S2[i,:,:,:]=S[i,:,:,:]*R2+So[i,:]
    S=S*R
    
    falls,fanys=[],[]
    for i in range(N):
        xs=np.array(list(Bt[0,i,:,0].flatten())+list(Bt[0,i,:,-1].flatten()[::-1]))
        ys=np.array(list(Bt[1,i,:,0].flatten())+list(Bt[1,i,:,-1].flatten()[::-1]))
        coords=np.vstack((xs,ys)).T
    
        poly=mpath.Path(coords,closed=True)
        points=list(zip(S[0,:,:,i].flatten(),S[1,:,:,i].flatten()))
        check=poly.contains_points(points,radius=1e-9)
        points1=list(zip(S1[0,:,:,i].flatten(),S1[1,:,:,i].flatten()))
        check1=poly.contains_points(points1,radius=1e-9)
        points2=list(zip(S2[0,:,:,i].flatten(),S2[1,:,:,i].flatten()))
        check2=poly.contains_points(points2,radius=1e-9)
        
        falls.append(np.all([np.all(check),np.all(check1),np.all(check2)]))
        fanys.append(np.any([np.any(check),np.any(check1),np.any(check2)]))
    falls,fanys=np.array(falls,dtype=bool),np.array(fanys,dtype=bool)
    return falls,fanys