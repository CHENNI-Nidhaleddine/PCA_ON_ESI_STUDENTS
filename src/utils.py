import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn.decomposition
import re

# NIDHALEDDINE CHENNI FROM SID

# CONSTANTS
CLASSEMENT=21
SUBJECTS_S1=np.array(('SYS1','RES1','ANUM','RO','ORG','IGL','THP','LANG1'))
SUBJECTS_S2=np.array(('MCSI','BDD','SEC','CPROJ','PROJ','LANG2','ARCH','SYS2','RES2'))
COEF_S1=np.array((5,5,5,3,3,5,4,2))
COEF_S2=np.array((5,5,1,3,3,2,4,4,4))

# FUNCTIONS
def readData(path):
    """ JUST READ DATA FROM EXCEL :) """
    data=pd.read_excel(path)
    return data
    
def writeData(data,path):
    """ JUST WRITE DATA TO EXCEL :) """
    data.to_excel(path,index=False)

def Preprocess(data):
    """ TRAIT CELLS LIKE (ANAD 4.5 < 5.6) WILL BECOME JUST 4.5 """
    data.columns=SUBJECTS_S2
    
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value=data.iloc[row,col]
            try:
                float(value)
            except:
                if(value==''):
                    data.iloc[row,col]=0
                else:
                    floats=re.findall("\d\.\d\d", value)
                    data.iloc[row,col]=floats[0]
    return data


def Cov(X):
    N=X.shape[0]
    meanX=X.mean()
    return (np.dot((X-meanX).T,X-meanX))/N

def PCA(X):
    X=np.asarray(X)*COEF_S2
    # center and normalize data
    STD=np.std(X,axis=0)
    M=np.mean(X,axis=0)
    X=X-M
    X=X/STD

    # Calculate eigen values and eigen vectors
    covX=np.cov(X.T)
    lambdas, Q= np.linalg.eigh(covX)
  
    # Order from biggest to smallest
    idx=np.argsort(-lambdas)
    lambdas=lambdas[idx]
    lambdas=np.maximum(lambdas,0) # in case there are some negative close to zero
    Q=Q[:,idx]
    
    # project the data
    Z=X.dot(Q)

    # project variables
    V=-np.sqrt(lambdas)*Q

    return lambdas,Q,Z,V

def plot_eigen_values(lambdas):
    plt.scatter(np.arange(1,lambdas.shape[0]+1),lambdas)
    plt.title('les valeurs propres')
    plt.show()

def plot_inertias(lambdas):
    plt.plot(lambdas/sum(lambdas))
    plt.title('les inerties')
    plt.show()

def plot_cum_inertias(lambdas):
    plt.plot(np.cumsum(lambdas/sum(lambdas)))
    plt.title('les inerties cumelee')
    plt.show()

def plot_indv_factor_map(Z,i=0,j=1):
    arr=np.zeros(Z.shape[0])+0.7
    arr[CLASSEMENT-1]=1 #my rank point with different color
    plt.scatter(-Z[:,i],-Z[:,j],c=arr,alpha=arr)
    plt.title('individuals factor map for axes ')
    axex='dim '+str(i+1)
    axey='dim '+str(j+1)
    plt.xlabel(axex)
    plt.ylabel(axey)
    plt.axhline()
    plt.axvline()
    plt.show()



def plot_var_facrot_map(lambdas,Q,labels,axe_x,axe_y):

    # calculate variable projection
    V=-np.sqrt(lambdas)*Q
    print("variable projection: ",V)

    figsize=(8,8)
    #Plot
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': 12}) 
    
    #Plot circle
    x=np.linspace(start=-1,stop=1,num=500)
    y_positive=lambda x: np.sqrt(1-x**2) 
    y_negative=lambda x: -np.sqrt(1-x**2)
    plt.plot(x,list(map(y_positive, x)), color='maroon')
    plt.plot(x,list(map(y_negative, x)),color='maroon')

    #Plot smaller circle
    x=np.linspace(start=-0.5,stop=0.5,num=500)
    y_positive=lambda x: np.sqrt(0.5**2-x**2) 
    y_negative=lambda x: -np.sqrt(0.5**2-x**2)
    plt.plot(x,list(map(y_positive, x)), color='maroon')
    plt.plot(x,list(map(y_negative, x)),color='maroon')

    #Create broken lines
    x=np.linspace(start=-1,stop=1,num=30)
    plt.scatter(x,[0]*len(x), marker='_',color='maroon')
    plt.scatter([0]*len(x), x, marker='|',color='maroon')
    for i in range(V.shape[0]):
        plt.text(V[i,axe_x],V[i,axe_y], s=labels[i] )
        plt.arrow(0,0, dx=V[i,axe_x], dy=V[i,axe_y], head_width=0.03, head_length=0.03,  length_includes_head=True)

    plt.xlabel(f"Dim {axe_x+1} ({round(lambdas[axe_x]/np.sum(lambdas)*100,2)}%)")
    plt.ylabel(f"Dim {axe_y+1} ({round(lambdas[axe_y]/np.sum(lambdas)*100,2)}%)")
    plt.show()

def plot_3d(Z,V):
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    idx=np.array((0,1,2,3,4,5,CLASSEMENT-1,21,22,150))
    arr=np.zeros(10)+0.7
    arr[6]=1
    ax.scatter3D(Z[idx,0],Z[idx,1], Z[idx,2], c=arr )
    plt.title("3D scatter plot")
    i=4 #PROJ
    ax.plot([0,-V[i,0]],[0,-V[i,1]],[0,-V[i,2]])
    ax.text(-V[i,0],-V[i,1],-V[i,2],SUBJECTS_S2[i])
    i=7 #SYS
    ax.plot([0,-V[i,0]],[0,-V[i,1]],[0,-V[i,2]])
    ax.text(-V[i,0],-V[i,1],-V[i,2],SUBJECTS_S2[i])
    i=5 #LANG
    ax.plot([0,-V[i,0]],[0,-V[i,1]],[0,-V[i,2]])
    ax.text(-V[i,0],-V[i,1],-V[i,2],SUBJECTS_S2[i])
    plt.show()
