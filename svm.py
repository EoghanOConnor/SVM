#
#
# Authors :     Eoghan O'Connor Luke Vickery
# Student I.Ds: 16110625        16110501
#
#
# File Name:    Project 2 , SVM soft margin classifier
#
# Description:  A soft margin, kernal based Support Vector Machine
#               is created.
#               It uses a radial basis function kernal with σ^2 = 0.25 
#               and and the CVXOPT library. 
#               
#               The program is trained and tested using two datasets.
#               Two different C parameters are invoked. C=1 and C=1e6
#               This shows the difference in performance between "soft"
#               and "hard" margin.
#               
#               The dataset implemented to classifiers to test both margins
#               of C values.
#               This programme produces the decision boundaries amd margins
#               for both classifiers, allowing their performance to be 
#               evaluted.
#
# Inputs:       Training-dataset --> cvx quadratic solver()-->
#               Krbf() --> makeB()
#               Testing-dataset-->kernClassify()--> 
#               plotContours()-->activationLevel()
#               
#                

#Import necessary imports
import cvxopt as cvx
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from cvxopt import solvers
from mpl_toolkits import mplot3d

# A Function to load in training and testing datasets and organise for use by SVM
def loadData():
    # Load in training data sets
    dataset = np.loadtxt("training-dataset.txt", dtype='float')
    print(f"Dataset is of shape {dataset.shape[0]} x {dataset.shape[1]}.")
    points, labels = dataset[:, :2], dataset[:, 2]

    # Load in testing data sets
    dataset_test = np.loadtxt("testing-dataset.txt", dtype='float')
    print(f"Dataset testing is of shape {dataset_test.shape[0]} x {dataset_test.shape[1]}.")
    points_t, labels_t = dataset_test[:, :2], dataset_test[:, 2]
    
    return points, labels, points_t, labels_t

# Function to train the support vector machine using a margin of hardness/softess C
def SVM(C):    
    N = len(labels)
    bv = cvx.matrix(0.0)
    
    # CVXOPT inputs to qp.
    qn = cvx.matrix(-np.ones(N))
    Gn = cvx.matrix(np.vstack((-np.eye(N), np.eye(N))))
    hn = cvx.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
    X = cvx.matrix(points)
    t = cvx.matrix(labels)
    
    # Variance for RBF kernel.
    S2 = 0.25  

    # Builds our non-linear matrix as part of the Wolfe dual. Is an input of solver.qp
    P_rbf = cvx.mul(t * t.T, Krbf(X, X, S2))

    # Solves a quadratic program built by the matrices defined previously.
    r_rbf = cvx.solvers.qp(P_rbf, qn, Gn, hn, t.T, bv)

    # These lambdas are our legrange mulitpliers from the solvers.qp function.
    # That takes in various matrix inputs defined above and returns them
    lambdas_rbf = cvx.matrix([round(li, 6) for li in r_rbf['x']])

    # Lambda function to pass values to Krbf function and set return values equal to x & y
    Krbf_02 = lambda x, y: Krbf(x, y, S2)  # RBF kernel with s2 = 0.2

    # Results in a single value average bias
    b_rbf = makeB(Krbf_02, lambdas_rbf, X, t)

    # output variables required for classification and plotting
    return X, t, lambdas_rbf, Krbf_02, b_rbf

# This function calculated the average bias for all support vectors
def makeB(K, lambdas, X, t, zero_tol=1e-10, precision=6):

    #Make bias under kernel K.  Lambdas and ts cvxopt column vectors.
    #X is a cvxopt matrix with 1 training vector per row.

    supports = 0
    b = 0
    for s in range(len(lambdas)):
        if lambdas[s] > zero_tol:  # lambdas[s] is a support vector if > 0.
            supports += 1
            b += t[s] - sum(cvx.mul(cvx.mul(lambdas, t), K(X, X[s, :])))
    return round(b / supports, precision)

# This function checks the calculated value for each point and returns the classification of the point
def kernClassify(K, lambdas, X, t, b, zero_tol=1e-10):
    #Requires Xs to be a matrix of training inputs arranged as cvxopt row vectors.
    #'xvec', the input to be classified can be a cvxopt row vector, a Python list
    # representing a row vector, or a NumPy 1-d array.
    
    # Initialise the misclassification number variable
    misclass = 0
    
    # Import testing data and run classifier? 
    for xvec, lab in zip(points_t, labels_t):
        # Do conversions on xvec if needed to coerce to a cvxopt matrix with 1 row and n cols
        # (i.e., a cvxopt row vector).
        if isinstance(xvec, list):  # Convert Python list to cvxopt row vector.
            xvec = cvx.matrix(xvec).T
        elif isinstance(xvec, np.ndarray):  #  Convert NumPy array to cvxopt row vector.
            xv = xvec
        xvec = cvx.matrix(xv) if xv.shape[0] == 1 else cvx.matrix(xv).T
        
        # Classify the point relative to the decision boundry
        op = b + sum(cvx.mul(cvx.mul(lambdas, t), K(X, xvec)))

        if op > 0:
            y = 1
        else:
            y = -1
        # Verification step to check if point is classified correctly or not
        if y != lab:
            misclass += 1
            # print(f"[{xvec[0]:7.4f},{xvec[1]:7.4f}] --> {y:+2d} ({lab:+2.0f})")

    # Print the ouput for each misclassified point
    print(f"\nThere are {misclass}/{len(labels_t)} misclassifications")
    
    return

# Function to calculate the radial basis kernel
def Krbf(x, y, s2):
    #RBF kernel on two CVXOPT row vectors, or matrices. s2 is the RBF variance parameter.
    return cvx.matrix(np.exp(-scipy.spatial.distance.cdist(x, y, metric='sqeuclidean') / (2 * s2)))

def activationLevel(xvec, K, lambdas, Xs, ts, b, zero_tol=1e-10):
    # Vector cooercion implemented as needed, same as in kernCalssify
    if isinstance(xvec, list):  # Convert Python list to cvxopt row vector.
        xvec = cvx.matrix(xvec).T
    elif isinstance(xvec, np.ndarray):  #  Convert NumPy array to cvxopt row vector.
        xv = xvec
        xvec = cvx.matrix(xv) if xv.shape[0] == 1 else cvx.matrix(xv).T
    # Calculate activation level for each point on plot grid
    y = b + sum(cvx.mul(cvx.mul(lambdas, ts), K(Xs, xvec)))
    return y 

# This function takes in the trained kernel, lamdas matrix and bias to plot the contour lines
def plotContours(Krbf_02, lambdas_rbf, X, t, b_rbf):
    # Create grid for contour plot
    x, y = np.meshgrid(np.linspace(-6,6,100), np.linspace(-6,6,100))
    # Store grid points into a 2 column matrix xx to get activation value for each row
    x1 = np.reshape(x, (x.size,1))
    y1 = np.reshape(y, (y.size,1))
    xx = np.hstack((x1,y1))

    # Declare op for storing activation levels
    op = []
    # put each grid point into calcActivation function and store activation value in z vector
    for coord in xx:
        op.append(activationLevel(coord,Krbf_02, lambdas_rbf, X, t, b_rbf))

    #reshape z for contour plotting
    z = np.reshape(op, (x.shape))

    #plot contour plot and scatter plot of data
    plt.figure(figsize=(12, 12))
    plt.contourf(x,y,z,levels=[np.min(z),0,np.max(z)],colors=["#A6BDFB","#F0AEDC"]) 
    plt.contour(x,y,z,levels=[-1,0,1],colors=["white","green","red"], linestyles = ['dashed', 'solid', 'dashed'])
    plt.scatter(points_t.T[0], points_t.T[1], c=labels_t.T, cmap='bwr')
    plt.axis('equal')
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.grid()
    plt.show()
    
    
    from mpl_toolkits import mplot3d
    plt.figure(figsize = (12,12))
    ax = plt.axes(projection='3d')
    ax.contour3D(x,y,z,200)
    ax.contour(x,y,z,levels=[-1,0,1],colors=["white","green","red"], linestyles = ['dashed', 'solid', 'dashed'])
    ax.scatter3D(points_t.T[0], points_t.T[1], c=labels_t.T, cmap='bwr')
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis") 
    ax.grid()
    
# Main function    
def main():

    #Dont show solving progress
    solvers.options['show_progress'] = False
    
    #---------------------------------------------------------------------------------------------
    # For soft margin SVM with C = 1
    
    X, t, lambdas_rbf, Krbf_02, b_rbf = SVM(C=1)
    
  
    np.set_printoptions(precision=2, suppress=True)

    # Uncomment the below code to see to print the lambda values
    #print(np.array(lambdas_rbf).ravel())

    # Prints out our bias value
    print(f'\nBias = {b_rbf}')

    # At this stage the kernel and bias are calculated and "trained"
    # Training --> testing
    
    # Run the classifier on the testing data using trained model  
    kernClassify(Krbf_02, lambdas_rbf, X, t, b_rbf)
    
    # Counts number of significant supports vectors (Legrangian multipliers greater than 1e-10)
    print(f"There are {np.sum(np.array(lambdas_rbf) != 0)} support vectors")
    
    plotContours(Krbf_02, lambdas_rbf, X, t, b_rbf)
    
    #---------------------------------------------------------------------------------------------
    # For soft margin SVM with C = 1e6
    
    X, t, lambdas_rbf, Krbf_02, b_rbf = SVM(C=1e6)
    
    # Not important
    np.set_printoptions(precision=2, suppress=True)

    # Uncomment the below code to see to print the lambda values
    #print(np.array(lambdas_rbf).ravel())

    # Prints out our bias value
    print(f'\nBias = {b_rbf}')

    # At this stage the kernel and bias are calculated and "trained"
    # Training --> testing

    # Run the classifier on the testing data using trained model
    kernClassify(Krbf_02, lambdas_rbf, X, t, b_rbf)

    # Counts number of significant supports vectors (Legrangian multipliers greater than 1e-10)
    print(f"There are {np.sum(np.array(lambdas_rbf) != 0)} support vectors")
    
    #plotting contours
    plotContours(Krbf_02, lambdas_rbf, X, t, b_rbf)

points, labels, points_t, labels_t = loadData()
main()
