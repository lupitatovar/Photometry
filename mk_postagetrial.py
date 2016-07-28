# The code begins at the bottom by identifying the sun-like stars and putting them into a list
# Section 2 is the photometry section where the ffi's and uncertainties are loaded in and unpacked. 
# In order to fit different number of gaussians remeber to change NG, initial_guess, and the title of the plot

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


def findothersources(imgt, xtarg, ytarg):
    
    image = imgt + 0.0
    sources = np.zeros((35, 3)) 
    counter = 0

    print(xtarg, ytarg)

    sizeimg = np.shape(imgt)[0]-1
    #image = np.log(image)

    #image -= np.median(image)
    
    #image /= np.std(image)
    

    #image[ytarg-4:ytarg+5,xtarg-4:xtarg+5] = 0
    
    for peaks in range(35):   #changes the number of stars boxed
    
        foundone = 0
        while foundone == 0:
            k = np.argmax(image)
            j,i = np.unravel_index(k, image.shape)
            if image[j,i] > 150000.9:
                image[max(j-4, 0):min(j+5,sizeimg+1),max(i-4,0):min(i+5, sizeimg+1)] = 0.0
            else:
                foundone = 1
        if image[j,i] > 300:
            sources[peaks] = [j,i,image[j,i]]
            image[max(j-4, 0):min(j+5,sizeimg+1),max(i-4,0):min(i+5, sizeimg+1)] = 0.0
            #print sources[peaks]
            counter += 1


    #sources[:,0] -= xtarg
    #sources[:,1] -= ytarg
    
    #test = plt.imshow(image, interpolation='nearest', cmap='gray')
    #plt.colorbar(test)
    #plt.show()
    #print(sources)
    #print sources
    return sources[0:counter,0], sources[0:counter, 1],sources[0:counter,2] #return amp

#---------------------------------------------------------------------------------------------------------------------------
#                                                Parameters' Class  
#---------------------------------------------------------------------------------------------------------------------------

  
class Parameters (object):

    def calc_gaussian(self,xdata_tuple,ngaussian):
         (x,y)=xdata_tuple
         a = np.cos(self.theta)**2/(2*self.sigma_x**2) + np.sin(self.theta)**2/(2*self.sigma_y**2)
         b = -np.sin(2*self.theta)/(4*self.sigma_x**2) + np.sin(2*self.theta)/(4*self.sigma_y**2)
         c = np.sin(self.theta)**2/(2*self.sigma_x**2) + np.cos(self.theta)**2/(2*self.sigma_y**2)
         t= self.amp*np.exp( - (a*(x-self.xc)**2 - 2*b*(x-self.xc)*(y-self.yc) + c*(y-self.yc)**2))

         if ngaussian > 1:
             a = np.cos(self.theta2)**2/(2*self.sigma_x2**2) + np.sin(self.theta2)**2/(2*self.sigma_y2**2)
             b = -np.sin(2*self.theta2)/(4*self.sigma_x2**2) + np.sin(2*self.theta2)/(4*self.sigma_y2**2)
             c = np.sin(self.theta2)**2/(2*self.sigma_x2**2) + np.cos(self.theta2)**2/(2*self.sigma_y2**2)
             t+= self.amp2*np.exp( -(a*(x-self.xc2)**2 - 2*b*(x-self.xc2)*(y-self.yc2) + c*(y-self.yc2)**2))
             
         if ngaussian > 2:
             a = np.cos(self.theta3)**2/(2*self.sigma_x3**2) + np.sin(self.theta3)**2/(2*self.sigma_y3**2)
             b = -np.sin(2*self.theta3)/(4*self.sigma_x3**2) + np.sin(2*self.theta3)/(4*self.sigma_y3**2)
             c = np.sin(self.theta3)**2/(2*self.sigma_x3**2) + np.cos(self.theta3)**2/(2*self.sigma_y3**2)
             t+= self.amp3*np.exp( - (a*(x-self.xc3)**2 - 2*b*(x-self.xc3)*(y-self.yc3) + c*(y-self.yc3)**2))

         return t   

    def gaussian1 (self,f):
        (sigma_x,sigma_y,theta,amp,xc,yc)=f
        self.amp=amp
        self.xc=xc
        self.yc=yc
        self.sigma_x= sigma_x
        self.sigma_y= sigma_y
        self.theta= theta

    def gaussian2 (self,f):
        self.amp2 = f[0]*self.amp
        self.xc2 = f[1]+self.xc
        self.yc2 = f[2]+self.yc
        self.sigma_x2 = f[3]
        self.sigma_y2 = f[4]
        self.theta2 =  f[5]


    def gaussian3 (self,f):
        self.amp3 = f[0]*self.amp
        self.xc3 = f[1]+self.xc
        self.yc3 = f[2]+self.yc
        self.sigma_x3 = f[3]
        self.sigma_y3 = f[4]
        self.theta3 = f[5]

#----------------------------------------------------------------------------------------------------------------
#                                         Fitting Multiple Gausians 
#----------------------------------------------------------------------------------------------------------------

def gaussian(xdata_tuple,N,*params):
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
    #print (amp,xc,yc,sigma_x,sigma_y,theta)
    return amp*np.exp( - (a*(x-xc)**2 - 2*b*(x-xc)*(y-yc) + c*(y-yc)**2))

def fitter(xdata_tuple,*params):
    NG=3 #number of gaussians we are fitting
    (x,y)= xdata_tuple
    if len(params) == 1:
        params=list(params[0])
    params=list(params)
    
    #print (params)
    N=int((len(params)-3-(6*(NG-1)))/3)  #number of stars we are fitting, inverts (amp,rowd,cold )
    #print(N)
    sum= np.zeros_like(x)
    for A in range(N):                                

        specifics= params[3*N:(3*N)+3] + params[A:3*N:N] #change this it will not be the last -3 it will be the 3 before the last 6
        
#gives the specific parameters we will be using, we always want 3 times(amp,x,y positions) the number of gaussians + the 3 fixed params (sigma_x,sigma_y, theta)

        pclass= Parameters() #call first gaussian function from parameters class to define g1
        g= pclass.gaussian1(specifics)
        if NG ==3:
            g=pclass.gaussian2(params[-12:-6]) #call last 6 in params
            g=pclass.gaussian3(params [-6:]) #once this one is called it will need the last 6 params, meaning g2 will need the 6 before the 6)
        if NG==2:
            g=pclass.gaussian2(params [-6:])
        #print (specifics)

        calcgauss = pclass.calc_gaussian(xdata_tuple,NG)
        sum = sum + calcgauss
            
    return sum

#------------------------------------------------------------------------------------------------------------------
#                                         Photometry Section (2) 
#------------------------------------------------------------------------------------------------------------------

def calc_slope(channel, col, row, source):

    d0 = np.array([])
    d1 = np.array([])
    d2 = np.array([])
    d3 = np.array([])


    ffilist = ['kplr2009114174833_ffi-cal.fits', 'kplr2009114204835_ffi-cal.fits', 'kplr2009115002613_ffi-cal.fits',
               'kplr2009115053616_ffi-cal.fits', 'kplr2009115080620_ffi-cal.fits', 'kplr2009115131122_ffi-cal.fits',
               'kplr2009115173611_ffi-cal.fits', 'kplr2009116035924_ffi-cal.fits', ##'kplr2009170043915_ffi-cal.fits',
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
    #ffilist = ['kplr2009114174833_ffi-cal.fits']

    ffielist = ['kplr2009114174833_ffi-uncert.fits','kplr2009114204835_ffi-uncert.fits', 'kplr2009115002613_ffi-uncert.fits',
               'kplr2009115053616_ffi-uncert.fits', 'kplr2009115080620_ffi-uncert.fits', 'kplr2009115131122_ffi-uncert.fits',
               'kplr2009115173611_ffi-uncert.fits', 'kplr2009116035924_ffi-uncert.fits', ##'kplr2009170043915_ffi-uncert.fits',
               'kplr2009231194831_ffi-uncert.fits', 'kplr2009260000800_ffi-uncert.fits', 'kplr2009292020429_ffi-uncert.fits',
               'kplr2009322233047_ffi-uncert.fits', 'kplr2009351005245_ffi-uncert.fits', 'kplr2010019225502_ffi-uncert.fits',
               'kplr2010020005046_ffi-uncert.fits', 'kplr2010049182302_ffi-uncert.fits', 'kplr2010078174524_ffi-uncert.fits',
               'kplr2010111125026_ffi-uncert.fits', 'kplr2010140101631_ffi-uncert.fits', 'kplr2010174164113_ffi-uncert.fits', 
               'kplr2010203012215_ffi-uncert.fits', 'kplr2010234192745_ffi-uncert.fits', 'kplr2010265195356_ffi-uncert.fits',
               'kplr2010296192119_ffi-uncert.fits', 'kplr2010326181728_ffi-uncert.fits', 'kplr2010356020128_ffi-uncert.fits',
               'kplr2011024134926_ffi-uncert.fits', 'kplr2011053174401_ffi-uncert.fits', 'kplr2011116104002_ffi-uncert.fits',
               'kplr2011145152723_ffi-uncert.fits', 'kplr2011177110110_ffi-uncert.fits', 'kplr2011208112727_ffi-uncert.fits',
               'kplr2011240181752_ffi-uncert.fits', 'kplr2011271191331_ffi-uncert.fits', 'kplr2011303191211_ffi-uncert.fits',
               'kplr2011334181008_ffi-uncert.fits', 'kplr2012004204112_ffi-uncert.fits', 'kplr2012032101442_ffi-uncert.fits',
               'kplr2012060123308_ffi-uncert.fits', 'kplr2012088132324_ffi-uncert.fits', 'kplr2012121122500_ffi-uncert.fits',
               'kplr2012151105138_ffi-uncert.fits', 'kplr2012179140901_ffi-uncert.fits', 'kplr2012211123923_ffi-uncert.fits',
               'kplr2012242195726_ffi-uncert.fits', 'kplr2012277203051_ffi-uncert.fits', 'kplr2012310200152_ffi-uncert.fits',
               'kplr2012341215621_ffi-uncert.fits', 'kplr2013011160902_ffi-uncert.fits', 'kplr2013038133130_ffi-uncert.fits',
               'kplr2013065115251_ffi-uncert.fits', 'kplr2013098115308_ffi-uncert.fits']

    
#--------------------------------- Makes the plot look nice ---------------------------------------------------------

    plt.figure(figsize=(11,8)) 
    #ax1 = fig.add_subplot(2, 2, j+1)
    ax1 = plt.gca() 

    plt.gcf().subplots_adjust(bottom=0.17, wspace=0.0, top=0.86, right=0.94, left=0.16)


    flag = 0

    slope = np.zeros(4)

#---------------------------------------------------------------------------------------------------------------------
#                                               Defining the  Postage Stamp 
#---------------------------------------------------------------------------------------------------------------------

    if channel[3] in [49,50,51,52]:
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

#------------- Opens fits files one at a time ----------------------------

        #ax2 = ax1.twiny()
        for icount, i in enumerate(ffilist):     
            a = p.open(i)
            print (i)
            time=a[5].header['MJDEND']
            e =p.open (ffielist [icount])

            quarter = a[0].header['quarter']

            if int(quarter) == 0:
                season = 3
            else:
                season = (int(quarter) - 2) % 4

#------------- Selects pixels we care about ----------------------------

            if season == j:

                img = a[channel[season]].data
                err= e[channel[season]].data
              
                img -= np.median(img)
                #print img.shape
                npix = 100
                aper = 11
                ymin = int(row[season]) - npix/2
                ymax = int(row[season]) + npix/2

                xmin = int(col[season]) - npix/2
                xmax = int(col[season]) + npix/2

                if ymin < 0:
                    ymin = 0
                    ymax = npix
                elif ymax > img.shape[0]:
                    ymax = img.shape[0]
                    ymin = img.shape[0] - npix

                if xmin < 0:
                    xmin = 0
                    xmax = npix
                elif xmax > img.shape[1]:
                    xmax = img.shape[1]
                    xmin = img.shape[1] - npix
                    

                ymin2 = int(max([int(row[season])-aper/2,0]))
                ymax2 = int(min([int(row[season])+aper/2,img.shape[0]]))
                xmin2 = int(max([int(col[season])-aper/2,0]))
                xmax2 = int(min([int(col[season])+aper/2,img.shape[1]]))
                
                pimg = img[ymin:ymax,xmin:xmax] ##big chunk around the star we care about, the large box not the little boxes    
                perr= err[ymin:ymax,xmin:xmax]
                # print(np.shape(pimg))          
                
                if np.max(pimg) > -1005:
                
                    #print np.max(pimg)
                    '''
                    aplot = plt.imshow((pimg), interpolation='nearest', cmap='gray',vmax=np.percentile(pimg,99))
                    plt.colorbar(aplot)
                    plt.show()
                    '''

                    if flag == 0:
                        #print row[season] - ymin, col[season] - xmin
                        flag = 1
                        try: 
                            rowd
                        except:
                            rowd, cold, amp =findothersources(pimg, row[season] - ymin, col[season] - xmin) ##postitions of 10 brightest stars surrounding
                            #print (rowd,cold,amp)
                
                    try:
                        initial_guess= popt
                    except:
                        initial_guess=np.concatenate((amp,rowd,cold,[2,3,np.pi/4],[0.5,2,2,2,3,np.pi/4],[0.5,1,-1,3,2,np.pi/5]))
                    #print (initial_guess)

#we need 6(NG-1) but they need to be actual numbers in order (rel.amp,xoffset,yoffset,sigx,sigy,theta)
                        
                #print (initial_guess)               
                xlist=list(range(0,npix))
                y=np.tile(xlist,npix).reshape((npix,npix))
                x=np.transpose(y)
                #print (np.shape(x))
                xdata= np.vstack((x.ravel(),y.ravel())) #rearranges the data from (x,y,z) to (x,y)
                zdata = pimg.ravel()
                edata = perr.ravel()
                #print (np.shape(xdata))
                #print (len(zdata))
                #print (np.min(zdata))
                
                popt,pcov=curve_fit(fitter,xdata,zdata,p0=initial_guess,sigma=edata)
#optimal result,covarience  -> (matrix that tells you how much changing one guess affects the rest), if you take the square root you get the error bars
                print (popt)
                model=fitter(xdata,popt)
                
                f = open("stardata/" + str(source) + '.txt','a')
                f.write('%s ' %time)
                for item in range(len(popt)):
                    f.write('%s ' %popt[item])
                    
                f.write('\n') #makes a new line
                f.close()
#----------------------------------------------------------------------------------------------------------------------------------                
#                                                  Plotting Section 
#----------------------------------------------------------------------------------------------------------------------------------
#if plotting =1 do this
#insert flag
    
    plt.subplot(2,2,2)
    b=plt.imshow(model.reshape(npix,npix),vmax=np.percentile(pimg,99),interpolation='nearest',cmap='seismic')
    plt.colorbar(b)
    plt.title('3 Gaussian Model')
    plt.subplot(2,2,1)
    a=plt.imshow(pimg,interpolation='nearest',vmax=np.percentile(pimg,99),cmap='seismic')
    plt.colorbar(a)
    plt.title('Raw Data')
    plt.subplot(2,2,3)
    guess= fitter (xdata,initial_guess)
    c=plt.imshow(guess.reshape(npix,npix),interpolation='nearest',cmap='seismic')
    plt.colorbar(c)
    plt.title('Initial Guess')
    plt.subplot(2,2,4)
    residuals= pimg-model.reshape(npix,npix)
    d=plt.imshow(residuals,vmax=np.percentile(pimg,99), vmin = -1*np.percentile(pimg, 99),interpolation='nearest',cmap='seismic')
    plt.colorbar(d)
    plt.title('Residuals')
    plt.show()
    return pimg,popt

#----------------- Only runs from python command line, not from import,  start point 1 ---------------------------------------------                  
if __name__ == '__main__':

    source = 3629717 #this will become a list of sun like stars, put it into a for loop to run calc_slope
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

    pimg,popt =calc_slope(channel, col, row, source) 

    
