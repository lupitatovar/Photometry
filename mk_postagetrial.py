import numpy as np
import matplotlib.pyplot as plt
import pyfits as p
#from photutils import CircularAperture
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import sys
import kplr
from scipy.optimize import curve_fit

plt.rcParams['axes.linewidth']=3
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.minor.width'] = 2
plt.rc('xtick.major', size=8, pad=8)
plt.rc('xtick.minor', size=6, pad=5)
plt.rc('ytick.major', size=8, pad=8)
plt.rc('ytick.minor', size=6, pad=5)

qdate = np.array([54964, 55002, 55092, 55184, 55277, 55371, 55462, 55567, 55649, 55739, 55834, 55930, 56014, 56103, 56203, 56307, 56397])

fmt = ['ko', 'rD', 'b^', 'gs']
offsets = np.array([0.001, 0.002, 0.0, 0.006])



#plt.gcf().subplots_adjust(bottom=0.17, wspace=0.0, top=0.92, right=0.95, left=0.16)

#incorporate the amplitdue into findothersources, import fitter after that function

def findothersources(imgt, xtarg, ytarg):
    
    image = imgt + 0.0
    sources = np.zeros((10, 2))
    counter = 0
    
    print(xtarg, ytarg)

    sizeimg = np.shape(imgt)[0]-1
    #image = np.log(image)

    #image -= np.median(image)
    
    #image /= np.std(image)
    
    fig2=plt.figure(2)
    ax=fig2.add_subplot(111)
    
    bb = ax.imshow(image, interpolation='nearest', cmap='gray', vmax = np.percentile(image, 99))
    fig2.colorbar(bb)
    fig2.show()

    image[ytarg-4:ytarg+5,xtarg-4:xtarg+5] = 0
    
    for peaks in range(10):   #will change the number of stars boxed
    
        foundone = 0
        while foundone == 0:
            k = np.argmax(image)
            j,i = np.unravel_index(k, image.shape)
            if image[j,i] > 175000.9:
                image[max(j-4, 0):min(j+5,sizeimg),max(i-4,0):min(i+5, sizeimg)] = 0.0
            else:
                foundone = 1
        if image[j,i] > 5:
            sources[peaks] = [j,i]
            image[max(j-4, 0):min(j+5,sizeimg),max(i-4,0):min(i+5, sizeimg)] = 0.0
            #print sources[peaks]
            counter += 1


    sources[:,0] -= xtarg
    sources[:,1] -= ytarg
    
    #test = plt.imshow(image, interpolation='nearest', cmap='gray')
    #plt.colorbar(test)
    #plt.show()
    #print(sources)
    #print sources
    return sources[0:counter,0], sources[0:counter, 1]

############## Fitting Multiple Gausians ##############

def gaussian(xdata_tuple,*params):
    (x,y) = xdata_tuple
    params=list(params[0])
    #print (params)
    amp=params[3]
    xc=params[4]
    yc=params[5]
    sigma_x=params[0]
    sigma_y=params[1]
    theta=params[2]
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    
    return amp*np.exp( - (a*(x-xc)**2 - 2*b*(x-xc)*(y-yc) + c*(y-yc)**2))

def fitter(xdata_tuple,*params):
    (x,y)= xdata_tuple
    if len(params) == 1:
        params=list(params[0])
    params=list(params)
    #print (params)
    N=int((len(params)-3)/3)
    sum= np.zeros_like(x)                   
    for A in range(N):                                
        specifics= params[-3:] + params[A:3*N:N]      #change parameters to be based on initial guess
        #print (specifics)
#gives the specific parameters we will be using, we always want 3 times(amp,x,y positions) the number of gaussians + the 3 fixed params (sigma_x,sigma_y, theta)
        one = gaussian(xdata_tuple, specifics)
        sum = sum + one                                #Adds up the gaussians
    #print (np.sum(sum))
    return sum

#print(np.shape(x))

initial_guess=[12,30,35,12,31,29,21,42,40,31,24,12,5,3,np.pi/4] #change initial guess 

#optimal result-popt, and covariance-pcov
xdata= np.vstack((x.ravel(),y.ravel())) #rearranges the data from (x,y,z) to (x,y)
zdata = pimg.ravel()
#print(xdata)
#print (zdata)

popt,pcov=curve_fit(fitter,xdata,zdata,p0=initial_guess)
model=fitter(xdata,popt)

################ Photometry Section #######################
def calc_slope(channel, col, row, source):

    d0 = np.array([])
    d1 = np.array([])
    d2 = np.array([])
    d3 = np.array([])


    ffilist = ['kplr2009114174833_ffi-cal.fits', 'kplr2009114204835_ffi-cal.fits', 'kplr2009115002613_ffi-cal.fits',
               'kplr2009115053616_ffi-cal.fits', 'kplr2009115080620_ffi-cal.fits', 'kplr2009115131122_ffi-cal.fits',
               'kplr2009115173611_ffi-cal.fits', 'kplr2009116035924_ffi-cal.fits', #'kplr2009170043915_ffi-cal.fits',
               'kplr2009231194831_ffi-cal.fits', 'kplr2009260000800_ffi-cal.fits', 'kplr2009292020429_ffi-cal.fits',
               'kplr2009322233047_ffi-cal.fits', 'kplr2009351005245_ffi-cal.fits', 'kplr2010019225502_ffi-cal.fits',
               'kplr2010020005046_ffi-cal.fits', 'kplr2010049182302_ffi-cal.fits', 'kplr2010078174524_ffi-cal.fits',
               'kplr2010111125026_ffi-cal.fits', 'kplr2010140101631_ffi-cal.fits', 'kplr2010174164113_ffi-cal.fits', 
               'kplr2010203012215_ffi-cal.fits', 'kplr2010234192745_ffi-cal.fits', 'kplr2010265195356_ffi-cal.fits',
               'kplr2010296192119_ffi-cal.fits', 'kplr2010326181728_ffi-cal.fits', 'kplr2010356020128_ffi-cal.fits',
               'kplr2011024134926_ffi-cal.fits', 'kplr2011053174401_ffi-cal.fits', 'kplr2011116104002_ffi-cal.fits',
               'kplr2011145152723_ffi-cal.fits', 'kplr2011177110110_ffi-cal.fits', 'kplr2011208112727_ffi-cal.fits',
               'kplr2011240181752_ffi-cal.fits', 'kplr2011271191331_ffi-cal.fits', 'kplr2011303191211_ffi-cal.fits',
               'kplr2011334181008_ffi-cal.fits', 'kplr2012004204112_ffi-cal.fits', 'kplr2012032101442_ffi-cal.fits',
               'kplr2012060123308_ffi-cal.fits', 'kplr2012088132324_ffi-cal.fits', 'kplr2012121122500_ffi-cal.fits',
               'kplr2012151105138_ffi-cal.fits', 'kplr2012179140901_ffi-cal.fits', 'kplr2012211123923_ffi-cal.fits',
               'kplr2012242195726_ffi-cal.fits', 'kplr2012277203051_ffi-cal.fits', 'kplr2012310200152_ffi-cal.fits',
               'kplr2012341215621_ffi-cal.fits', 'kplr2013011160902_ffi-cal.fits', 'kplr2013038133130_ffi-cal.fits',
               'kplr2013065115251_ffi-cal.fits', 'kplr2013098115308_ffi-cal.fits']

######## Makes the plot look nice #####    
    fig = plt.figure(figsize=(11, 8)) 
    #ax1 = fig.add_subplot(2, 2, j+1)
    ax1 = plt.gca()

    plt.gcf().subplots_adjust(bottom=0.17, wspace=0.0, top=0.86, right=0.94, left=0.16)


    flag = 0

    slope = np.zeros(4)
##########################################

#defines postage stamp#
    if channel[3]in[49,50,51,52]:
        vals=[1,2,3,0]
    else:
        vals=[0,1,2,3]
    for k,j in enumerate(vals):
        phot1 = np.array([])
        phot2 = np.array([]) 
        phot3 = np.array([]) 
        phot4 = np.array([])
        phot5 = np.array([])
        counter = 0
        d0 = np.array([])

#####opens fits files one at a time#####
        #ax2 = ax1.twiny()
        for icount, i in enumerate(ffilist):     
            a = p.open(i)

            quarter = a[0].header['quarter']

            if int(quarter) == 0:
                season = 3
            else:
                season = (int(quarter) - 2) % 4

########## selects pixels we care about #########
            if season == j:

                img = a[channel[season]].data 
              


                img -= np.median(img)
                #print img.shape
                npix = 120
                aper = 11
                ymin = int(max([int(row[season])-npix/2,0]))
                ymax = int(min([int(row[season])+npix/2+1,img.shape[0]]))
                xmin = int(max([int(col[season])-npix/2,0]))
                xmax = int(min([int(col[season])+npix/2+1,img.shape[1]]))

                ymin2 = int(max([int(row[season])-aper/2,0]))
                ymax2 = int(min([int(row[season])+aper/2+1,img.shape[0]]))
                xmin2 = int(max([int(col[season])-aper/2,0]))
                xmax2 = int(min([int(col[season])+aper/2+1,img.shape[1]]))
#########
                pimg = img[ymin:ymax,xmin:xmax] ####chunk around the star we care about#
                
                
                if np.max(pimg) > -1005:
                
                    #print np.max(pimg)
                    '''
                    aplot = plt.imshow((pimg), interpolation='nearest', cmap='gray')
                    plt.colorbar(aplot)
                    plt.show()
                    '''

                    if flag == 0:
                        #print row[season] - ymin, col[season] - xmin
                        flag = 1
                        try: 
                            rowd
                        except:
                            rowd, cold =findothersources(pimg, row[season] - ymin, col[season] - xmin) ##postitions of 10 brightest stars surrounding
                        
#optimal result, and covariance
                    xdata= np.vstack((x.ravel(),y.ravel())) #rearranges the data from (x,y,z) to (x,y)
                    zdata = pimg.ravel()
#print(xdata)
#print (zdata)

                    popt,pcov=curve_fit(fitter,xdata,zdata,p0=initial_guess)
                    model=fitter(xdata,popt)

######only runs from python command line, not from import########                    
if __name__ == '__main__':

    source = 3629717
    client = kplr.API()
    targ = client.target(source)
    
    channel = [targ.params['Channel_0'], targ.params['Channel_1'], 
                targ.params['Channel_2'], targ.params['Channel_3']]
    
    col = [targ.params['Column_0'], targ.params['Column_1'], 
                targ.params['Column_2'], targ.params['Column_3']]
    
    row = [targ.params['Row_0'], targ.params['Row_1'], 
                targ.params['Row_2'], targ.params['Row_3']]
    

    #cold = np.array([-7, -10, 45, -33])
    #rowd = np.array([-22, 27, 21, 34])

    calc_slope(channel, col, row, source)

##0.00546878979552
##0.00963833403853
