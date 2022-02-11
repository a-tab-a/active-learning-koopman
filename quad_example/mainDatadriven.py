import numpy as np
from koopman_operator import KoopmanOperator
from scipy.io import loadmat
from quatmath import euler2mat 
from group_theory import VecTose3, TransToRp, RpToTrans


#helper function
def get_measurement(x):
    g = x[0:16].reshape((4,4)) ## SE(3) matrix 
    R,p = TransToRp(g)
    twist = x[16:]
    grot = np.dot(R, [0., 0., -9.81]) ## gravity vec rel to body frame
    return np.concatenate((grot, twist)) 

def prepare_data(x):
    
    print(x)
    p = x[1:4]
    v = x[4:7]
    rpy = x[7:10]
    print(rpy)
    R = euler2mat(rpy)
    g = RpToTrans(R, p).ravel()
    omega = x[10:13]
    twist = np.r_[omega,v] #this is the serail in the measurement code
    state = np.r_[g, twist]
    return state


#load the data
file_train = 'train_data_A.mat'
data_train = loadmat(file_train)['data_set_train']
#in the training data; there are
# time, pos vel, acceration, rpy, omega, RPM in that order


simulation_time = len(data_train[:,0])
simulation_step_size = data_train[2,1]-data_train[1,1]
simulation_data = data_train[:,1:]


#instiate koopman operator class
koopman_operator = KoopmanOperator(simulation_step_size)


#solve for koopman based on the data

def main():

    err = np.zeros(simulation_time)
    for t in range(simulation_time-1):

        #### measure state and transform through koopman observables
        current_step_data = simulation_data[t,:]
        next_state_data = simulation_data[t+1,:]
        state = prepare_data(current_step_data)
        next_state = prepare_data(next_state_data)
        m_state = get_measurement(state)
        m_next_state = get_measurement(next_state)
        u = current_step_data[-4:]
        t_state = koopman_operator.transform_state(m_state) #probably dont need it
        koopman_operator.compute_operator_from_data(m_state,
                                                    u, 
                                                    m_next_state)
        Kx, Ku = koopman_operator.get_linearization() ### grab the linear matrices
        print(Ku)
        return(Kx,Ku)


if __name__=='__main__':
    main()
   


