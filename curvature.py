# coding=utf-8

import numpy as np

def curvature_cal(p):
    [p1,p2,p3,p4] = p
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    x2 = p2[0]
    y2 = p2[1]
    z2 = p2[2]
    x3 = p3[0]
    y3 = p3[1]
    z3 = p3[2]
    x4 = p4[0]
    y4 = p4[1]
    z4 = p4[2]
    a11=2*(x2-x1); a12=2*(y2-y1); a13=2*(z2-z1)
    a21=2*(x3-x2); a22=2*(y3-y2); a23=2*(z3-z2)
    a31=2*(x4-x3); a32=2*(y4-y3); a33=2*(z4-z3)
    b1=x2*x2-x1*x1+y2*y2-y1*y1+z2*z2-z1*z1
    b2=x3*x3-x2*x2+y3*y3-y2*y2+z3*z3-z2*z2
    b3=x4*x4-x3*x3+y4*y4-y3*y3+z4*z4-z3*z3
    d=a11*a22*a33+a12*a23*a31+a13*a21*a32-a11*a23*a32-a12*a21*a33-a13*a22*a31
    d1=b1*a22*a33+a12*a23*b3+a13*b2*a32-b1*a23*a32-a12*b2*a33-a13*a22*b3
    d2=a11*b2*a33+b1*a23*a31+a13*a21*b3-a11*a23*b3-b1*a21*a33-a13*b2*a31
    d3=a11*a22*b3+a12*b2*a31+b1*a21*a32-a11*b2*a32-a12*a21*b3-b1*a22*a31
    x=d1/d
    y=d2/d
    z=d3/d
    r = np.sqrt((x1-x)**2+(y1-y)**2+(z1-z)**2)
    return 1/r



def curvature_2d_cal(p1):
    p2=[[1,1,1],[1,1,1],[1,1,1]]
    dPx=p1[0][0];   
    dQx=p1[1][0];   
    dRx=p1[2][0];   
   
    dPy=p1[0][1];   
    dQy=p1[1][1];   
    dRy=p1[2][1];   
   
    dPz=p1[0][2];   
    dQz=p1[1][2];   
    dRz=p1[2][2];   
   
    x1=dQx-dPx;  
    x2=dRx-dPx;   
   
    y1=dQy-dPy;   
    y2=dRy-dPy;   
   
    z1=dQz-dPz;   
    z2=dRz-dPz;   
       
    pi=y1*z2-z1*y2;   
    pj=z1*x2-x1*z2;   
    pk=x1*y2-y1*x2;   
   
    if((pi==0) and (pj==0) and (pk==0)):    
        return FALSE;   
    
    dMx=(dPx+dQx)/2;   
    dMy=(dPy+dQy)/2;   
    dMz=(dPz+dQz)/2;   

    dMi=pj*z1-pk*y1;   
    dMj=pk*x1-pi*z1;   
    dMk=pi*y1-pj*x1;   

    dNx=(dPx+dRx)/2;   
    dNy=(dPy+dRy)/2;   
    dNz=(dPz+dRz)/2;   

    dNi=pj*z2-pk*y2;   
    dNj=pk*x2-pi*z2;   
    dNk=pi*y2-pj*x2;   
   
    ds=pi*pi+pj*pj+pk*pk;   
    ds=np.sqrt(ds);   
   
    if(ds==0):      
        p2[1][0]=0;   
        p2[1][1]=0;   
        p2[1][2]=0;   
    else:   
        p2[1][0]=pi/ds;   
        p2[1][1]=pj/ds;   
        p2[1][2]=pk/ds;   
        
    tn=((dMy-dNy)*dMi+dMj*(dNx-dMx))/(dNj*dMi-dMj*dNi+1e-10);   
    tm=(dNx+dNi*tn-dMx)/dMi;   
   
    dX0=dMx+dMi*tm;   
    dY0=dMy+dMj*tm;   
    dZ0=dMz+dMk*tm;   
   
    p2[0][0]=dX0;   
    p2[0][1]=dY0;   
    p2[0][2]=dZ0;   
 
    dR=(dX0-dPx)*(dX0-dPx)+(dY0-dPy)*(dY0-dPy)+(dZ0-dPz)*(dZ0-dPz);   
    dR=np.sqrt(dR);   
   
    radium=dR;   
    return 1/radium;    
   
def curvature(myinput,thre=0.01):
    curvature = np.array([])
    new_myinput = np.array([[0,0,0]])   
    for i in range(1,myinput.shape[0]-2):
        p1=[myinput[i-1],myinput[i],myinput[i+1],myinput[i+2]]
        this_curvature=curvature_cal(p1)
        if i is 1:
            if this_curvature>=thre:
                new_myinput=[myinput[i]]
        if this_curvature<thre:
            continue
        else:
            curvature = np.append(curvature,this_curvature)
            new_myinput = np.append(new_myinput,[myinput[i]],0)
            if i is 1:
                curvature=np.insert(curvature,0,this_curvature,0)
                
            if i is myinput.shape[0]-3:
                curvature=np.insert(curvature,-1,this_curvature,0)
                new_myinput=np.insert(new_myinput,0,myinput[i],0)
                
    return new_myinput,curvature

def curvature_com(myinput,thre=0.01):
    curvature = np.array([])
    for i in range(1,myinput.shape[0]-1):
        p1=[myinput[i-1],myinput[i],myinput[i+1]]
        this_curvature=curvature_cal(p1)
        curvature = np.append(curvature,this_curvature)
        if i == 1:
            curvature=np.insert(curvature,0,this_curvature,0)
        if i == myinput.shape[0]-2:
            curvature=np.insert(curvature,i+1,this_curvature,0)
    return myinput,curvature


def curvature_any_bothway(myinput,thre=1e-6,point_num=53868):
    curvature = np.array([])
    new_myinput = np.array([[0,0,0]]) 
    count = 0
    this_shape = myinput.shape[0]-1
    for i in range(1,int(this_shape/2)):
        
        p1=[myinput[i-1],myinput[i],myinput[i+1]]
        p2=[myinput[this_shape-i-1],myinput[this_shape-i],myinput[this_shape-i+1]]
        this_curvature=curvature_cal(p1)

        this_curvature_2=curvature_cal(p2)
        if i is 1:
            if this_curvature>=thre:
                new_myinput=np.array([myinput[i]])
                curvature=np.insert(curvature,0,this_curvature,axis=0) 
                count += 1
            if this_curvature_2 >= thre:
                new_myinput = np.insert(new_myinput,new_myinput.shape[0],[myinput[this_shape-i]],0)
                curvature=np.insert(curvature,curvature.shape[0],this_curvature_2,0)
                count += 1
                
        if this_curvature>=thre and count < point_num:
            curvature = np.insert(curvature,i,this_curvature,0)
            new_myinput = np.insert(new_myinput,i,[myinput[i]],0)
            count += 1

            if i is int(this_shape/2)-1:
                curvature=np.insert(curvature,-i,this_curvature,0)
                new_myinput=np.insert(new_myinput,-i,myinput[i],0)
                count += 1
                if (this_shape%2) != 0:
                    curvature=np.insert(curvature,-i+1,this_curvature,0)
                    new_myinput=np.insert(new_myinput,-i+1,myinput[i],0)
        
        if this_curvature_2>=thre and count < point_num:
            curvature = np.insert(curvature,curvature.shape[0],this_curvature_2,0)
            new_myinput = np.insert(new_myinput,new_myinput.shape[0],[myinput[i]],0)
            count += 1

            if i is int(this_shape/2)-1:
                curvature=np.insert(curvature,-i,this_curvature_2,0)
                new_myinput=np.insert(new_myinput,-i,myinput[i],0)
                count += 1
    return new_myinput,curvature

def curvature_del_to_anynum(myinput,thre=1e-6,point_num=48128):
    curvature = np.array([])
    new_myinput = np.array([[0,0,0]]) 
    count = myinput.shape[0]-point_num-1
    left_count = myinput.shape[0]-new_myinput.shape[0]-1
    for i in range(1,myinput.shape[0]-2):
        left_count -= 1
        p1=[myinput[i-1],myinput[i],myinput[i+1],myinput[i+2]]
        this_curvature=curvature_cal(p1)
        if i is 1:
            new_myinput=[myinput[i]]
            curvature=np.insert(curvature,0,this_curvature,0) 
            
        if count==0:
            curvature = np.append(curvature,this_curvature)
            new_myinput = np.append(new_myinput,[myinput[i]],0)
            if i is myinput.shape[0]-3:
                curvature=np.insert(curvature,-1,this_curvature,0)
                new_myinput=np.insert(new_myinput,0,myinput[i],0)
        else:
            if this_curvature<thre*(count/(left_count-count+1e-6)):
                count -= 1
            else:
                curvature = np.append(curvature,this_curvature)
                new_myinput = np.append(new_myinput,[myinput[i]],0)
                if i is myinput.shape[0]-3:
                    curvature=np.insert(curvature,-1,this_curvature,0)
                    new_myinput=np.insert(new_myinput,0,myinput[i],0)
    return new_myinput,curvature