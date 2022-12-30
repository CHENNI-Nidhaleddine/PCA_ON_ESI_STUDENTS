import utils
import numpy as np 

# NIDHALEDDINE CHENNI FROM SID

def main():

    path='data/PV1CSS2.xlsx'
    axe_x=0
    axe_y=1

    print("Reading and processing data ...")
    data=utils.readData(path)
    data=utils.Preprocess(data)
    utils.writeData(data,path)
    print("Data is ready ...")

    print("Performing PCA ...")
    # Coef multiplication
    data=np.asarray(data)
    data[:,0]=data[:,0]*5
    data[:,1]=data[:,1]*5
    data[:,2]=data[:,2]*1
    data[:,3]=data[:,3]*3
    data[:,4]=data[:,4]*4
    data[:,5]=data[:,5]*2
    data[:,6]=data[:,6]*4
    data[:,7]=data[:,7]*5
    data[:,8]=data[:,8]*5
    
    # lambdas: eigen values
    # Q: eigen vectors
    # Z: individuals projection
    # V: variables projection
    lambdas,Q,Z,V=utils.PCA(data)

    print("PCA finished ...")

    utils.plot_eigen_values(lambdas)
    utils.plot_inertias(lambdas)
    utils.plot_cum_inertias(lambdas)
    utils.plot_indv_factor_map(Z,axe_x,axe_y)
    utils.plot_var_facrot_map(lambdas,Q,utils.SUBJECTS_S2,axe_x,axe_y)
    utils.plot_3d(Z,V) # STATIC FUNCTION TO SEE THE RANK ON PROJ
    
    print("End ...")
    
if __name__=='__main__':
    main()