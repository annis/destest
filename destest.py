import numpy as np
import fitsio as fio
import h5py
import cPickle as pickle
import yaml
import os
import sys
import time
import data_read
import cProfile, pstats
# and maybe a bit optimistic...
from multiprocessing import Pool
# from mpi4py import MPI
try:
    from sharedNumpyMemManager import SharedNumpyMemManager as snmm 
    use_snmm = True
except:
    use_snmm = False

import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab


if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,


def child_testsuite( calc ):

    params, selector, calibrator = calc

    Testsuite( params, selector=selector, calibrator=calibrator, child=True )

class Testsuite(object):
    """
    Testsuite manager class. Initiated with a yaml file path or dictionary that contains necessary settings and instructions.
    """

    def __init__( self, param_file, selector = None, calibrator = None, child = False, **kwargs ):

        # Read in yaml information
        self.params     = yaml.load(open(param_file))
        self.params['param_file'] = param_file


        # Read in the data
        data = data_read.Dataread(param_file)

        # Run tests
        if 'general_stats' in self.params:
            GeneralStats(self.params,data.selector,data.calibrator,data.source,self.params['general_stats'])

        if 'hist_1d' in self.params:
            test = Hist1D(self.params,data.selector,data.calibrator,data.source,self.params['hist_1d'])
            test.plot()

        if 'hist_2d' in self.params:
            Hist2D(self.params,data.selector,data.calibrator,data.source,self.params['hist_2d'])

        if 'split_mean' in self.params:

            if self.params['use_mpi'] and (not child):
                procs = comm.Get_size()
                iter_list = [self.params['split_x'][i::procs] for i in xrange(procs)]
                calcs = []
                for proc in range(procs):
                    if iter_list[proc] == []:
                        continue
                    params = self.params.copy()
                    params['split_x'] = iter_list[proc]
                    self.selector.kill_source()
                    calcs.append((params,data.selector,data.calibrator))
                pool.map(child_testsuite, calcs)
            else:
                test = LinearSplit(self.params,data.selector,data.calibrator,data.source,self.params['split_x'],self.params['split_mean'])
                test.plot()

class Splitter(object):
    """
    A class for managing splitting the data set into bins and accessing the binned data.
    Initiate with a testsuite object.
    """

    def __init__( self, params, selector, calibrator, source, nbins = None ):

        self.params     = params
        self.selector   = selector
        self.calibrator = calibrator
        self.source     = source
        self.bins       = self.params['linear_bins']
        self.x          = None
        self.y          = None
        self.xcol       = None
        self.ycol       = None
        self.order      = None

        if 'split_x' in self.params:
            for col in self.params['split_x']:
                if col not in self.source.cols:
                    raise NameError(col + ' not in source.')
        else:
            self.params['split_x'] = self.source.cols

        if nbins is not None:
            self.bins = nbins

        return

    def get_x( self, col, xbin=None, return_mask=False ):
        """
        Get the 'x' column - the column you're binning the data by. 
        If you haven't already called splitter with this x col, the data will be read from the source and the binning edges will be set up.
        Optionally give a bin number, it will return the portion of the x array that falls in that bin. Can also optionally return the mask for that bin.
        """

        # If column doesn't already exist in splitter, read the data and define bins in self.split().
        if col != self.xcol:
            self.xcol  = col
            self.order = None
            self.split(col)

        # If not asking for a bin selection, return
        if xbin is None:
            return

        # If asking for a bin selection, find the appropriate mask and return that part of the x array.
        start,end = self.get_bin_edges(xbin)
        # print 'returning x bin',start,end
        mask      = [np.s_[start_:end_] for start_,end_ in tuple(zip(start,end))] # np.s_ creates an array slice 'object' that can be passed to functions
        mask      = [ order_[mask_] for order_,mask_ in tuple(zip(self.order,mask)) ]
        if return_mask:
            return self.x[start[0]:end[0]],mask
        return self.x[start[0]:end[0]]

    def get_y( self, col, xbin=None, return_mask=False ):
        """
        Get the 'y' column - the column you're doing stuff with in bins of the x col. If you haven't called splitter.get_x(), an error will be raised, since you haven't defined what you're binning against.  
        If you haven't already called splitter with this y col, the data will be read from the source.
        Optionally give a bin number, it will return the portion of the y array that falls in that bin. Can also optionally return the mask for that bin.
        """

        if self.xcol is None:
            raise NameError('There is no x column associated with this splitter.')

        # If column doesn't already exist in splitter, read the data and order it to match x ordering for efficient splitting.
        if col != self.ycol:
            self.ycol = col
            self.y = self.selector.get_col(col,nosheared=True)
            for i,y_ in enumerate(self.y):
                self.y[i] = y_[self.order[i]]
            self.y = self.y[0]

        print 'ysize',len(self.y),self.y.nbytes

        # If not asking for a bin selection, return
        if xbin is None:
            return

        # If asking for a bin selection, find the appropriate mask and return that part of the y array.
        start,end = self.get_bin_edges(xbin)
        # print 'returning y bin',start,end
        mask      = [np.s_[start_:end_] for start_,end_ in tuple(zip(start,end))]
        mask      = [ order_[mask_] for order_,mask_ in tuple(zip(self.order,mask)) ]
        if return_mask:
            return self.y[start[0]:end[0]],mask
        return self.y[start[0]:end[0]]

    def split( self, col ):
        """
        Reads in a column (x) and sorts it. If you allowed cache reading, it will check if you've already done this and just read it in from the pickle cach. Then finds the edges of the bins you've requested.
        """

        # Check if cache file exists and use it if you've requested that.
        sort_file = data_read.file_path(self.params,'cache','sort',var=col,ftype='pickle')
        if self.params['load_cache']:
            print 'loading split sort cache',sort_file

            if os.path.exists(sort_file):
                self.order,self.x = data_read.load_obj(sort_file)

        # Cache file doesn't exist or you're remaking it
        if self.order is None:
            print 'split sort cache not found'
            # Read x
            self.x     = self.selector.get_col(col)
            # save the index order to sort the x array for more efficient binning
            self.order = []
            for i,x_ in enumerate(self.x):
                self.order.append( np.argsort(x_) )
                self.x[i] = x_[self.order[i]]
            # save cache of sorted x and its order relative to the source
            data_read.save_obj( [self.order,self.x], sort_file )

        # get bin edges
        self.get_edge_idx()
        self.x = self.x[0]

        return

    def get_edge_idx( self ):
        """
        Find the bin edges that split the data into the ranges you set in the yaml or into a number of equal-weighted bins.
        """

        self.edges = []
        # You've provided a number of bins. Get the weights and define bin edges such that there exists equal weight in each bin.
        if not self.params['split_by_w']:

            for x_ in self.x:
                xw = np.ones(len(x_))
                normcumsum = xw.cumsum() / xw.sum()
                self.edges.append( np.searchsorted(normcumsum, np.linspace(0, 1, self.bins+1, endpoint=True)) )

        else:

            w,R = self.calibrator.calibrate(self.xcol,return_full_w=True,weight_only=True,include_Rg=True)
            for x_,w_ in tuple(zip(self.x,w)):
                xw = w_*R
                normcumsum = xw.cumsum() / xw.sum()
                self.edges.append( np.searchsorted(normcumsum, np.linspace(0, 1, self.bins+1, endpoint=True)) )

        return

    def get_bin_edges( self, xbin ):
        """
        Helper function to return the lower and upper bin edges.
        """
        return [edge[xbin] for edge in self.edges],[edge[xbin+1] for edge in self.edges]


class LinearSplit(object):
    """
    Test class to do linear splitting (operations on binned data not at the 2pt level).
    """

    def __init__( self, params, selector, calibrator, source, split_x, split_y, nbins = None, func=np.mean, **kwargs ):

        self.params = params
        self.source = source
        if self.params['split_mean'] is not None:
            for col in self.params['split_mean']:
                if col not in self.source.cols:
                    raise NameError(col + ' not in source.')
        else:
            self.params['split_mean'] = self.source.cols

        self.calibrator = calibrator
        self.splitter   = Splitter(params,selector,calibrator,source,nbins=nbins)
        self.split_x    = split_x
        self.split_y    = split_y
        self.step       = 0

        if not self.params['plot_only']:
            # 'step' and this separate call is meant as a placeholder for potential parallelisation
            self.iter()

    def iter( self ):
        """
        Loop over x columns (quantities binned by) and y columns (quantities to perform operations on in bins of x), perform the operations, and save the results
        """

        for x in self.split_x:
            print 'x col',x
            n     = []
            xmean = []
            xlow  = []
            xhigh = []
            for xbin in range(self.splitter.bins):
                # get x array in bin xbin
                xval       = self.splitter.get_x(x,xbin)
                n.append( len(xval) )
                xlow.append( xval[0] )
                xhigh.append( xval[-1] )
                # get mean values of x in this bin
                xmean.append( mean(x,xval,self.calibrator,return_std=False) )
            for y in self.split_y:
                if x==y:
                    continue
                print 'y col',y
                ymean = []
                ystd  = []
                for xbin in range(self.splitter.bins):
                    # get y array in bin xbin
                    yval,mask  = self.splitter.get_y(y,xbin,return_mask=True)
                    # get mean and std (for error) in this bin
                    ymean_,ystd_ = mean(y,yval,self.calibrator,mask=mask)
                    ymean.append( ymean_ )
                    ystd.append( ystd_/np.sqrt(n[xbin]) )

                # Save results
                table = np.array([n,xlow,xmean,xhigh,ymean,ystd]).T
                print 'mean',table
                data_read.write_table(self.params, table,'test_output','linear_split',var=x,var2=y,var3=str(self.splitter.bins))

    def plot( self ):

        for x in self.split_x:
            for y in self.split_y:
                if x==y:
                    continue
                fpath = data_read.file_path(self.params,'test_output','linear_split',var=x,var2=y,var3=str(self.splitter.bins))
                data = np.loadtxt(fpath)
                plt.figure()
                if x in self.params['plot_log_x']:
                    data[data[:,2]<0,2] = -np.log(-data[data[:,2]<0,2])
                    data[data[:,2]>0,2] = np.log(-data[data[:,2]>0,2])
                plt.errorbar(data[:,2],data[:,4],yerr=data[:,5],marker='.',linestyle='',color='b')
                plt.minorticks_on()
                if y in self.params['e']:
                    plt.axhline(0.,color='k')
                plt.xlabel(x)
                plt.ylabel(y)
                plt.tight_layout()
                fpath = data_read.file_path(self.params,'test_output','linear_split',var=x,var2=y,var3=str(self.splitter.bins),ftype='png')
                plt.savefig(fpath, bbox_inches='tight')
                plt.close()


class GeneralStats(object):
    """
    Test class to calculate general statistics for a list of columns.
    """

    def __init__( self, params, selector, calibrator, source, split, func=np.mean, **kwargs ):

        self.params = params
        self.source = source
        if self.params['general_stats'] is not None:
            for col in self.params['general_stats']:
                if col not in self.source.cols:
                    raise NameError(col + ' not in source.')
        else:
            self.params['general_stats'] = self.source.cols

        self.calibrator = calibrator
        self.splitter   = Splitter(params,selector,calibrator,source,nbins=1)
        self.split      = split
        self.step       = 0

        # 'step' and this separate call is meant as a placeholder for potential parallelisation
        self.iter()

    def iter( self ):
        """
        Loop over x columns, perform basic statistics, and save the results
        """

        fpath = file_path( self.params,'test_output','general_stats' )
        with open(fpath,'w') as f:
            f.write('col min max mean std rms\n')
            for i,x in enumerate(self.split):
                print 'x col',x
                # get x array in bin xbin
                xval = self.splitter.get_x(x,0)
                min_ = xval[0]
                max_ = xval[-1]
                # get mean values of x in this bin
                mean_,std_,rms_ = mean(x,xval,self.calibrator,return_std=True,return_rms=True)
                # Save results
                f.write(x+' '+str(min_)+' '+str(max_)+' '+str(mean_)+' '+str(std_)+' '+str(rms_)+'\n')
        f.close()


class Hist1D(object):
    """
    Test class to do linear 1D histograms on a set of columns.
    """

    def __init__( self, params, selector, calibrator, source, split, func=np.mean, **kwargs ):

        self.params = params
        self.source = source
        if self.params['hist_1d'] is not None:
            for col in self.params['hist_1d']:
                if col not in self.source.cols:
                    raise NameError(col + ' not in source.')
        else:
            self.params['hist_1d'] = self.source.cols

        self.calibrator = calibrator
        self.splitter   = Splitter(params,selector,calibrator,source,nbins=1)
        self.split      = split
        self.step       = 0

        if not self.params['plot_only']:
            # 'step' and this separate call is meant as a placeholder for potential parallelisation
            self.iter()

    def iter( self ):
        """
        Loop over x columns (quantities binned by) and y columns (quantities to perform operations on in bins of x), perform the operations, and save the results
        """

        for i,x in enumerate(self.split):
            print 'x col',x
            # get x array in bin xbin
            self.splitter.get_x(x)
            bins,edges = np.histogram(self.splitter.x,bins=self.params['hist_bins'])
            # bins  = []
            # edges = np.linspace(self.splitter.x[0], self.splitter.x[-1], self.params.hist_bins+1, endpoint=True)
            # self.splitter.get_x(x)
            # for xbin in range(self.params.hist_bins):
            #     np.searchsorted(self.splitter.x,edges)
            #     bins.append( self.splitter.edges )
            #     edges.append( self.splitter.x )

            # Save results
            data_read.write_table(self.params, edges,'test_output','hist_1d',var=x,var2='edges')
            data_read.write_table(self.params, bins, 'test_output','hist_1d',var=x,var2='bins' )

    def plot( self ):

        for x in self.split:
            fpath = file_path(self.params,'test_output','hist_1d',var=x,var2='edges')
            edges = np.loadtxt(fpath)
            fpath = file_path(self.params,'test_output','hist_1d',var=x,var2='bins')
            bins = np.loadtxt(fpath)
            plt.figure()
            if x in self.params['plot_log_x']:
                edges[edges<0] = -np.log(-edges[edges<0])
                edges[edges>0] = np.log(edges[edges>0])
            plt.plot(edges[:-1],bins,marker='',linestyle='-',color='b',drawstyle='steps-pre',fillstyle='bottom')
            plt.minorticks_on()
            plt.xlabel(x)
            plt.ylabel('N')
            plt.tight_layout()
            fpath = file_path(self.params,'test_output','hist_1d',var=x,ftype='png')
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()


class Hist2D(object):
    """
    Test class to do linear 1D histograms on a set of columns.
    """

    def __init__( self, params, selector, calibrator, source, split, func=np.mean, **kwargs ):

        self.params = params
        self.source = source
        if self.params['hist_2d'] is not None:
            for col in self.params['hist_2d']:
                if col not in self.source.cols:
                    raise NameError(col + ' not in source.')
        else:
            self.params['hist_2d'] = self.source.cols

        self.calibrator = calibrator
        self.splitter   = Splitter(params,selector,calibrator,source,nbins=1)
        self.split      = split
        self.step       = 0

        # 'step' and this separate call is meant as a placeholder for potential parallelisation
        self.iter()

    def iter( self ):
        """
        Loop over x columns (quantities binned by) and y columns (quantities to perform operations on in bins of x), perform the operations, and save the results
        """

        for x in self.split:
            print 'x col',x
            # get x array in bin xbin
            self.splitter.get_x(x)
            for y in self.split:
                if x==y:
                    continue
                print 'y col',y
                self.splitter.get_y(y)
                bins,xedges,yedges = np.histogram2d(self.splitter.x,self.splitter.y,bins=self.params['hist_bins'])

                # Save results
                data_read.write_table(self.params, np.array([xedges,yedges]).T,'test_output','hist_2d',var=x,var2=y,var3='edges')
                data_read.write_table(self.params, bins, 'test_output','hist_2d',var=x,var2=y,var3='bins' )


def mean( col, x, calibrator, mask=None, return_std=True, return_rms=False ):
    """
    Function to do mean, std, rms calculations
    """

    # Get response and weight.
    if mask is None:
        R,c,w = calibrator.calibrate(col)
    else:
        R,c,w = calibrator.calibrate(col,mask=mask)
    print 'Rcw',col,R,c,w

    # do the calculation
    if R is not None:

        x  = np.copy(x)-c
        Rw = scalar_sum(w*R,len(x))
        if return_std:
            Rw2 = scalar_sum(w*R**2,len(x))

    else:

        Rw  = scalar_sum(w,len(x))
        if return_std:
            Rw2 = Rw

    mean = np.sum(w*x)/Rw
    if not (return_std or return_rms):
        return mean
    if return_std:
        std=np.sqrt(np.sum(w*(x-mean)**2)/Rw2)
        if not return_rms:
            return mean,std
    if return_rms:
        rms=np.sqrt(np.sum((w*x)**2)/Rw)
        if not return_std:
            return mean,rms

    return mean,std,rms

def scalar_sum(x,length):
    # catches scalar weights, responses and multiplies by the vector length for the mean
    if np.isscalar(x):
        return x*length
    return np.sum(x)


pr = cProfile.Profile()

if __name__ == "__main__":
    """
    """
    pr.enable()

    # from mpi_pool import MPIPool
    # comm = MPI.COMM_WORLD
    # pool = MPIPool(comm)
    # if not pool.is_master():
    #     pool.wait()
    #     sys.exit(0)

    Testsuite( sys.argv[1] )

    # pool.close()


    pr.disable()
    ps = pstats.Stats(pr).sort_stats('time')
    ps.print_stats(20)

