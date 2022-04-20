import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit

def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple                                                        
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()



def fit_gauss(predicted_heatmap, predicted_coords, targ_coords=None, visualize=False):
    # Create x and y indices
    x = np.linspace(0, predicted_heatmap.shape[1], predicted_heatmap.shape[1])
    y = np.linspace(0, predicted_heatmap.shape[0], predicted_heatmap.shape[0])
    x, y = np.meshgrid(x, y)

    
    # initial_guess = (3,predicted_heatmap.shape[0]/2,predicted_heatmap.shape[1]/2,3,3,0,0)
    initial_guess = (predicted_heatmap.max(),predicted_coords[0], predicted_coords[1],6,6,0,0)
    bounds = ([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.pi,-np.inf] , [np.inf,np.inf,np.inf,np.inf,np.inf, np.pi,np.inf])
    # bounds = ([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf] , [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])

    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), predicted_heatmap.ravel(), p0=initial_guess, bounds=bounds)
    data_fitted = twoD_Gaussian((x, y), *popt)

    sigma_prod= popt[3] * popt[4]
    sigma_ratio = max(popt[3], popt[4]) /min(popt[3], popt[4])

    print("prod and ratio: ", sigma_prod, sigma_ratio)
    if visualize and sigma_ratio>= 2:
        fig, ax = plt.subplots(1, 3)

        data_fitted_notheta = twoD_Gaussian((x, y), *[popt[0],popt[1],popt[2],popt[3],popt[4],0,popt[6]])
        ax[0].imshow(predicted_heatmap, cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        ax[0].contour(x, y, data_fitted_notheta.reshape(predicted_heatmap.shape[0], predicted_heatmap.shape[1]), 8, colors='w')

        data_fitted_halftheta = twoD_Gaussian((x, y), *[popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]/2,popt[6]])
        ax[1].imshow(predicted_heatmap, cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        ax[1].contour(x, y, data_fitted_halftheta.reshape(predicted_heatmap.shape[0], predicted_heatmap.shape[1]), 8, colors='w')

        ax[2].imshow(predicted_heatmap, cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
        ax[2].contour(x, y, data_fitted.reshape(predicted_heatmap.shape[0], predicted_heatmap.shape[1]), 8, colors='w')

        print("Pred coords: ", predicted_coords, " and targ coords: ", targ_coords)
        print("Covariance: Amplitude %s, coords: (%s,%s) sigmaXY (%s, %s), theta %s, offset %s " % (popt[0],popt[1],popt[2],popt[3],popt[4],np.degrees(popt[5]),popt[6]))
        plt.show()
        plt.close()

    return {"amplitude": popt[0], "mean": [popt[1],popt[2]],"sigma": [popt[3],popt[4]],"theta": np.degrees(popt[5]), "offset": popt[6]}