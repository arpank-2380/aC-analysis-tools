#!/usr/bin/python3
# Written by Arpan Kundu, arpan.kundu@gmail.com
# C 2022 
# Version: March 19, 2022

import sys,os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


fermi_energy = {"2ac": -5.55838, "3ac": -6.00572, "4ac": -6.196095, "7ac": -5.75367, \
               "8ac": -5.659515, "9ac": -5.861625, "10ac": -5.81378, "dia": -4.68608}


def relaxed_ipr_eigval_ind(path='../../Data/IPR-vs-Eigenvalues/Relaxed/',sample='9ac', orb_start_ind=400):
    raw_data = np.genfromtxt(path + sample + '_relaxed_ipr_eigval.dat')
    ipr = raw_data[:,0]
    eigval = raw_data[:,1] - fermi_energy[sample]
    indices = [i for i in range(orb_start_ind,orb_start_ind+len(ipr)+1)]
    return ipr,eigval,indices


def block_average(data,block_size):
    """
    This function chops the received data into different blocks
    Then it calculates the mean and variance for each blocks and
    also calculates the variance of mean and return these three 
    values
    """
    nblock = int(len(data)/block_size)
    #print(nblock)
    mean_block = np.zeros(nblock)                    # this is the mean at each block
    #print(mean_block)
    for i in range(nblock):
        istart = i*block_size
        iend   = (i+1)*block_size
        mean_block[i] = np.mean(data[istart:iend])
    block_mean = np.mean(mean_block)
    block_var = np.var(mean_block)/float(nblock-1)
    #print(" %16.8f    %16.8f " %(block_mean, block_var))
    return block_mean, block_var

def mobility_edge(vbm,cbm,ipr_cut):
    """
       This function accepts 2 lists: vbm and cbm. 
       These lists contain [KS-eigenvalue, IPR] for vbm/cbm orbitals for a range of indices 
       for a particular configuration, and returns the movility gap of that configuration.
    """

    homo_cutoff_reached = False
    lumo_cutoff_reached = False
    for i in range(len(vbm)-1,-1,-1):
        if vbm[i][1] <= ipr_cut:
           homo_cutoff_reached = True
           #print("VBM: Index = %d En = %f IPR = %f" %(i,vbm[i,0],vbm[i,1]))
           break
    vbm_index = i

    for i in range(len(cbm)):
        if cbm[i][1] <= ipr_cut:
           lumo_cutoff_reached = True
           #print("CBM: Index = %d En = %f IPR = %f" %(i,cbm[i,0],cbm[i,1]))
           break
    cbm_index = i
    mobility_cb_edge = cbm[cbm_index][0]
    mobility_vb_edge = vbm[vbm_index][0]
    mobility_gap = mobility_cb_edge - mobility_vb_edge
    cutoff_reached = homo_cutoff_reached and lumo_cutoff_reached
    return  mobility_vb_edge, mobility_cb_edge, mobility_gap, cutoff_reached



def collect_mobility_edges(input_data_path = '../../Data/IPR-vs-Eigenvalues/' , output_path = 'Mobility_Gap_Data/', \
                           thermostats=[], samples=[], temperatures=[],vbm_ind=432,ipr_cut=0.0012):
    ev2mev=1000
    static_mobility_vb = {}
    static_mobility_cb = {}
    static_mobility_gap = {}
    print ("#System     Static mobility gap w IPR cutoff = %15.8f"%ipr_cut)
    for sample in samples:
        static_ipr, static_eigval, static_indices  = relaxed_ipr_eigval_ind(sample = sample)
        loc_cbm = static_indices.index(vbm_ind) + 1
        static_vbm = [[static_eigval[i],static_ipr[i]] for i in range(loc_cbm)]
        static_cbm = [[static_eigval[i],static_ipr[i]] for i in range(loc_cbm,len(static_indices)-1)]
        tmp_mobility_vb, tmp_mobility_cb, tmp_mobility_gap, \
               tmp_cutoff_reached = mobility_edge(static_vbm,static_cbm,ipr_cut)
        print(" %s         %12.6f  "%(sample,tmp_mobility_gap))
        static_mobility_vb[sample] = tmp_mobility_vb
        static_mobility_cb[sample] = tmp_mobility_cb
        static_mobility_gap[sample] = tmp_mobility_gap


    for thermostat in thermostats:
        for sample in samples:
            outfile = open(output_path + thermostat + '_' + sample +'_iprc_' + str(ipr_cut) + '.dat','w')
            outfile.write("#  T(K)   Mobility_gap (eV)   gap_renorm (meV)  gap_err bar(meV)  VB_renorm(meV)  CB_renorm(meV)\n")
            for temp in temperatures:
                directory = input_data_path + thermostat + "-" + str(temp) + "K/"
                prefix = sample + "_" + str(temp) + "_wf"
                m_edge = avg_mobility_edge(directory = directory, prefix = prefix, ipr_cut = ipr_cut)
                outfile.write("%8.2f    %12.6f     %12.6f     %12.6f     %12.6f    %12.6f \n"%(temp,m_edge.avg_gap, \
                             (m_edge.avg_gap - static_mobility_gap[sample])*ev2mev, m_edge.error_gap*ev2mev,\
                             (m_edge.avg_mob_vb - static_mobility_vb[sample])*ev2mev, \
                             (m_edge.avg_mob_cb - static_mobility_cb[sample])*ev2mev ))

                print("%s | %s | %4d | IPR cutoff reached for %d configurations"%(thermostat, sample, temp, m_edge.cutoff_reached))




class avg_mobility_edge:
      """
         This class takes directory location and input file prefixes as input and returns the mobility gap.
         The mobility gap is computed based on an IPR cut-off. 
         The gap is defined as the gap between mobility-vbm and mobility-cbm. 
         The former one is defined as the 1e state with highest index, which has IPR just less than a predefined cut-off.
         The latter one is defined as the 1e state with lowest indix, which has IPR just less than a predefined cut-off.
         Arg:
             vbm_init, vbm_final = Range of VB orbitals for which IPR calculated and stored
             cbm_init,cbm_final =  Range of CB orbitals for which IPR calculated and stored
             ipr_cut = A predefined IPR cut off.
      """
      def __init__(self, directory, prefix, vbm_init=420, vbm_final=432, cbm_init=433, cbm_final=444, ipr_cut=10):
          
          if directory[-1] != '/':
             directory += '/'
          vbm_files = ["%s%d"%(prefix,i)+'_ipr.dat' for i in range(vbm_init,vbm_final+1)]
          cbm_files = ["%s%d"%(prefix,i)+'_ipr.dat' for i in range(cbm_init,cbm_final+1)]

          i_vbm = 0
          self.vbm_data=[]
          for i in range(vbm_init,vbm_final+1):
              self.vbm_data.append(np.genfromtxt(directory+vbm_files[i_vbm]))
              i_vbm += 1

          i_cbm = 0
          self.cbm_data=[]
          for i in range(cbm_init,cbm_final+1):
              self.cbm_data.append(np.genfromtxt(directory+cbm_files[i_cbm]))
              i_cbm += 1
          
          nconfig = len(self.vbm_data[0])

          self.mobility_gap = np.float64([])
          self.mobility_vb_edge = np.float64([]) 
          self.mobility_cb_edge = np.float64([])

          self.cutoff_reached = 0
          for i_config in range(1,nconfig+1):
              vbm, cbm = self.collect_eigval_ipr(i_config)
              tmp_mobility_vb_edge, tmp_mobility_cb_edge, tmp_mobility_gap, \
                    tmp_cutoff_reached = mobility_edge(vbm,cbm,ipr_cut)         
              #self.mobility_gap = np.append(self.mobility_gap,tmp_mobility_gap)
              if tmp_cutoff_reached:
                 self.mobility_gap = np.append(self.mobility_gap,tmp_mobility_gap)
                 self.mobility_vb_edge = np.append(self.mobility_vb_edge, tmp_mobility_vb_edge)
                 self.mobility_cb_edge = np.append(self.mobility_cb_edge, tmp_mobility_cb_edge)
                 self.cutoff_reached += 1
         #     #print("Mobility gap for Config-%d = %f"%(i_config,tmp_mobility_gap))
          
          self.avg_mob_vb, self.var_mob_vb =  block_average(self.mobility_vb_edge,50)
          self.avg_mob_cb, self.var_mob_cb =  block_average(self.mobility_cb_edge,50)
          self.avg_gap, self.var_gap = block_average(self.mobility_gap,50)
          self.error_vb =  2*np.sqrt(self.var_mob_vb)
          self.error_vb =  2*np.sqrt(self.var_mob_cb)
          self.error_gap = 2*np.sqrt(self.var_gap)

      def collect_eigval_ipr(self,config):
          #print("VBM of configuration: " + str(config))
          vbm = []
          for i in range(len(self.vbm_data)):
              vbm.append([self.vbm_data[i][config-1,1],self.vbm_data[i][config-1,0]])

          cbm = []
          for i in range(len(self.cbm_data)):
              cbm.append([self.cbm_data[i][config-1,1],self.cbm_data[i][config-1,0]])

          vbm = np.array(vbm)
          cbm = np.array(cbm)
          return vbm, cbm




class ensemble_averaged_overlap:
      """
      This class computes the ensemble average overlap matrics for VBM and CBM and plots the same

      Arguments:
      sample = a string that defines the sample number (s9, s8, dia etc)
      temperature = A number, temperature of the simulation that directs the directory of the data directory (eg, Q-100K/ Q-500K/ etc..)
      thermostat = Classical (Q) or Quantum (Q) thermostat so that directory can be found (C-100K/, Q-100K/ etc.)
      dpi = The dpi of the png output images
      font_size = Font size of the .png output images
      cmap = A string. Name for availablepython colormap
      show_plot = if True it would show the plot while running
      data_path = Data directory path where overlap matrices are stored
      figure_path = Where figure PNG files would be dumped
      """

      def __init__(self,sample=9,thermostat='Q',temperature=100,\
                                dpi=600, font_size = 20, show_plot=True, cmap=None, \
                                data_path = '../../Data/overlap_matrix_wrt_static/',\
                                figure_path = '../../Figures/Overlap_matrix/'):
           self.sample = sample; self.dpi = dpi; self.font_size = font_size; 
           self.show_plot = show_plot; self.cmap=cmap
           self.figure_file = figure_path +  str(self.sample) + '-' + \
                              thermostat + '_' + str(temperature) + 'K.png'
           self.overlap_data_path = data_path + thermostat + '-' + str(temperature) + 'K/'
           self.vb_data_file = self.overlap_data_path + str(self.sample) + '-orbital-0.overlap'
           self.cb_data_file = self.overlap_data_path + str(self.sample) + '-orbital-1.overlap'
           self.vb_orbital_list, self.vb_coeff = self.ensemble_average(band='vb')
           self.cb_orbital_list, self.cb_coeff = self.ensemble_average(band='cb')
           self.plot()

      def ensemble_average(self,band='vb'):
          """
            This method takes the overlap matrix for each configuration and 
            computes ensemble average.
            This also prints a mapping between orbitals from MD snapshot and
            orbitals of the energy minimized structure.
          """
          if band == 'vb':
             overlap_file = self.vb_data_file
          else:
             overlap_file = self.cb_data_file
          
          orbital_list = self.get_orbital_info(overlap_file)
          overlap_data = np.genfromtxt(overlap_file)
          
          #print(orbital_list)
          #print("%d, %d"%(len(overlap_data),len(orbital_list)))
          norb = len(orbital_list)
          no_configs = len(overlap_data) // norb
 
          outfile_path = self.overlap_data_path + 'Logfiles/'
          if not os.path.isdir(outfile_path):
             os.system("mkdir "+outfile_path)
         
          outfile = open(outfile_path + str(self.sample) + '-' + band + ".out",'w+')
          outfile.write("# This logfile is created by overlap_analysis.py script written by Arpan Kundu\n")
          outfile.write("# This logfile is showing the mapping of orbital indices b/w MD snapshot and energy minimized configuration\n")
          
          row = 0; coeff_array_avg = np.zeros((norb,norb),np.float64)
          
          for i in range(no_configs):
              iorb = 0
              for orbital in orbital_list:
                  coeff = np.square(overlap_data[row])
                  norm_coeff = coeff/np.sum(coeff)
                  coeff_array_avg[iorb,:] +=  norm_coeff[:]
                  max_coeff_index = np.argmax(np.square(overlap_data[row]))
                  mapped_orbital = orbital_list[max_coeff_index]
                  outfile.write("%d=>%d "%(orbital,mapped_orbital)) 
                  row += 1
                  iorb += 1
              outfile.write("\n")
              
          coeff_array_avg = coeff_array_avg / np.float(no_configs)    
          return orbital_list, coeff_array_avg
       
      def get_orbital_info(self,overlap_file):
          """ Extracts orbital indices related information from an overlap file"""
          overlap_file_object = open(overlap_file,"r")
          overlap_file_header = overlap_file_object.readline().split()[2:]
          overlap_file_object.close()
          last_index = overlap_file_header.index('Reference')
          orbital_indices_string = overlap_file_header[:last_index]
          orbital_indices = []
          for orbital_string in orbital_indices_string:
              orbital_indices.append(int(orbital_string.strip(",'[]")))
          return orbital_indices


      def plot(self,vb_ticks = [420, 424, 428,432],cb_ticks = [433,437, 441], num_cm_ticks=4, shrink_cm=0.35):
          """
            This method plots the VBM and CBM ensemble averaged overlap matrices 
          """
          plt.rc('font', size = self.font_size)
          plt.rc('axes', titlesize = self.font_size)
          fig, axs = plt.subplots(1,2)
          fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.1)
          #plt.setp(axs,xlim=custom_xlim,ylim=custom_ylim)

          vb_cm_ticks = np.linspace(np.min(self.vb_coeff), np.max(self.vb_coeff), num_cm_ticks)
          cb_cm_ticks = np.linspace(np.min(self.cb_coeff), np.max(self.cb_coeff), num_cm_ticks)
          for i in range(2):
              axs[i].set_xlabel("Orbital indices")
              axs[i].set_ylabel("Orbital indices")
          axs[0].xaxis.set_ticks(vb_ticks)
          axs[0].yaxis.set_ticks(vb_ticks)
          vb = axs[0].imshow(self.vb_coeff, interpolation='none', extent=[self.vb_orbital_list[0], self.vb_orbital_list[-1],\
                     self.vb_orbital_list[-1],self.vb_orbital_list[0]], aspect=1, cmap=self.cmap)
          fig.colorbar(vb, ax = axs[0], shrink=shrink_cm, ticks=vb_cm_ticks, format='%.2f')
          axs[1].xaxis.set_ticks(cb_ticks)
          axs[1].yaxis.set_ticks(cb_ticks)
          cb = axs[1].imshow(self.cb_coeff, interpolation='none', extent=[self.cb_orbital_list[0], self.cb_orbital_list[-1],\
                     self.cb_orbital_list[-1],self.cb_orbital_list[0]], aspect=1, cmap=self.cmap)
          fig.colorbar(cb, ax = axs[1], shrink=shrink_cm, ticks=cb_cm_ticks, format='%.2f')
          plt.savefig(self.figure_file, dpi = self.dpi , bbox_inches = 'tight', pad_inches = 0.1)
          if self.show_plot:
             plt.show()


class eigenvalue_ipr_density:
      """
      This class computes and plots the various probability densities and cumulative probability densities
      of the inverse participation ratio (IPR) and either (i) Kohn-Sham orbital energy or (ii) Orbital indices
      Arguments:
      sample = a string that defines the sample number (2ac, 4ac, dia etc)
      temperature = A number, temperature of the simulation that directs the directory of the data directory (eg, Q-100K/ Q-500K/ etc..)
      thermostat = Classical (Q) or Quantum (Q) thermostat so that directory can be found (C-100K/, Q-100K/ etc.)
      vbm_init = Indices of VBM orbital from where IPR values are saved.
      vbm_final = Last VBM index (HOMO index)
      cbm_init = CBM (LUMO) index
      cbm_final = Last CBM orbital for which IPR values are saved
      en_lim = A tuple: range of orbital energy values to be shown in the plot
      orb_lim = A tuple: range of orbital indices to be shown in the plot 
      ipr_lim = A tuple: the range of IPR values to be shown in the plot
      den_lim = A tuple: the range of probability density to be shown in the color map
      dpi = The dpi of the png output images
      font_size = Font size of the .png output images
      show_plot = if True it would show the plot while running
      static_data = Static data path, if None does not show static data
      cmap = A string. Any python colormap
      cum_prob_cut = Value is a float between 0 and 1. 
                     If 0.9 is chosen, it would print IPR values for each orbital indices for which 
                     atleast 90% observations are below that IPR value.  
      """
      def __init__(self, sample='4ac', temperature=100, thermostat='Q',\
                         vbm_init = 420, vbm_final = 432, cbm_init = 433, cbm_final = 444,\
                         en_lim=None, orb_lim = None, ipr_lim=None, den_lim=None,\
                         en_ticks = None, orb_ticks = None, ipr_ticks=None,\
                         dpi = 600, font_size = 20, show_plot = True, cmap=None, hline=None,\
                         cum_prob_cut = 0.9,\
                         static_data_path = '../../Data/IPR-vs-Eigenvalues/Relaxed/',\
                         data_path = '../../Data/IPR-vs-Eigenvalues/',\
                         figure_path = '../../Figures/Eigenvalue_IPR_density/'):
 
          print("#<=== This is eigenvalue_ipr density class speaking for:")
          print("# Sample = %s, Temperature = %f K, Thermostat = %s ===>"%(sample, float(temperature), thermostat))
          self.sample = sample; self.temperature = temperature; self.thermostat = thermostat
          self.vbm_init = vbm_init; self.vbm_final = vbm_final; self.cbm_init  = cbm_init; self.cbm_final = cbm_final
          self.dpi = dpi; self.font_size = font_size; self.show_plot = show_plot; self.cmap = cmap; self.hline=hline
          self.fig_file_prefix = figure_path + str(sample) + '_' + thermostat + '-' + str(temperature) + 'K_'
          self.data_path = data_path + thermostat + "-" + str(temperature) + "K/"
          self.static_data_path = static_data_path
          self.norb = self.cbm_final - self.vbm_init + 1

          prefix = sample + "_" + str(temperature) + "_wf" 
          self.vbm_files = ["%s%d"%(prefix,i)+'_ipr.dat' for i in range(self.vbm_init,self.vbm_final+1)]
          self.cbm_files = ["%s%d"%(prefix,i)+'_ipr.dat' for i in range(self.cbm_init,self.cbm_final+1)]
          self.fig_counter=1

          i_vbm = 0
          self.vbm_data=[]
          for i in range(self.vbm_init,self.vbm_final+1):
              self.vbm_data.append(np.genfromtxt(self.data_path+self.vbm_files[i_vbm]))
              i_vbm += 1
          
          i_cbm = 0
          self.cbm_data=[]
          for i in range(self.cbm_init,self.cbm_final+1):
              self.cbm_data.append(np.genfromtxt(self.data_path+self.cbm_files[i_cbm]))
              i_cbm += 1
          
          self.nconfig = len(self.vbm_data[0])
          #print("vbm_data=" + str(len(vbm_data[0])))

          if self.static_data_path is not None:
             r_ipr, r_en, r_indeces =  relaxed_ipr_eigval_ind(path = self.static_data_path, sample = self.sample)
             relaxed_ipr = r_ipr[r_indeces.index(self.vbm_init):r_indeces.index(self.cbm_final)+1]
             relaxed_en = r_en[r_indeces.index(self.vbm_init):r_indeces.index(self.cbm_final)+1]
             relaxed_indices = r_indeces[r_indeces.index(self.vbm_init):r_indeces.index(self.cbm_final)+1]
             static_ipr_en = np.zeros((len(relaxed_ipr),2),np.float)
             static_ipr_ind = np.zeros((len(relaxed_ipr),2),np.float)
             for i in range(len(relaxed_ipr)):
                 static_ipr_en[i,0] = relaxed_en[i]; static_ipr_en[i,1] = relaxed_ipr[i]
                 static_ipr_ind[i,0] = float(relaxed_indices[i]); static_ipr_ind[i,1] = relaxed_ipr[i] 
          else:  
             static_ipr_en = None
             static_ipr_ind = None

          en,ipr1,ind,ipr2 = self.prepare_plot_data()

          ### IPR vs Energy plot
          h_en_ipr, xedge_en_ipr, yedge_en_ipr, image_en_ipr, plot_en_ipr = \
          self.plot_hist2d( x=en, y=ipr1, xlabel="$E-E_{\\rm Fermi}$ (eV)", ylabel="IPR",\
                            xlim=en_lim, ylim=ipr_lim, xticks = en_ticks, yticks = ipr_ticks ,\
                            bins=(100,100), vlim = den_lim  , fig_suffix = 'ipr_en', hline=self.hline, \
                            cmap = self.cmap, static = static_ipr_en)   
          ### IPR vs Indices plot
          h_ind_ipr, xedge_ind_ipr, yedge_ind_ipr, image_ind_ipr, plot_ind_ipr = \
          self.plot_hist2d( x=ind, y=ipr2, xlabel="Orbital indices", ylabel="IPR",\
                            xlim = orb_lim,ylim = ipr_lim, xticks = en_ticks, yticks = ipr_ticks,\
                            bins=(self.norb,100), vlim = den_lim, fig_suffix = 'ipr_ind', cmap=self.cmap,\
                            hline = self.hline, static = static_ipr_ind)

          prob_den, cum_prob_den_gt, cum_prob_den_lt = \
          self.calc_cum_prob_den( xedge = xedge_ind_ipr , yedge = yedge_ind_ipr, \
                                  h = h_ind_ipr, cum_prob_cut = cum_prob_cut)

          ### IPR vs Indices plot: Normalized along indices axis
          plot_prob_den = self.plot_prob_den(xedge_ind_ipr, yedge_ind_ipr, prob_den, \
                                             log_cmap=True,cmap=self.cmap,fig_suffix='den_norm_ind',\
                                             hline = self.hline, static = static_ipr_ind)

          ### Plotting cumulative densities
          plot_cum_den_gt = self.plot_prob_den(xedge_ind_ipr, yedge_ind_ipr, cum_prob_den_gt, \
                                               fig_suffix='cum_gt', cmap=self.cmap,\
                                               cbar_label='Cumulative probability density',\
                                               hline = self.hline, static = static_ipr_ind)

          plot_cum_den_lt = self.plot_prob_den(xedge_ind_ipr, yedge_ind_ipr, cum_prob_den_lt, \
                                               fig_suffix='cum_lt', cmap=self.cmap,\
                                               cbar_label='Cumulative probability density',\
                                               hline = self.hline, static = static_ipr_ind)

          if self.show_plot:
             plt.show()
          
      def collect_eigval_ipr(self,config):
          """
          This method collects ipr and eigenvalue for a specific configuration, say 10th.
          """
          #print("VBM of configuration: " + str(config))
          en = []; ipr = []; vbm_ipr = []
          for i in range(len(self.vbm_data)):
              #print( str(i+self.vbm_init)+ " :   " + str(self.vbm_data[i][config-1,:]))
              en.append(self.vbm_data[i][config-1,1])
              ipr.append(self.vbm_data[i][config-1,0])
              vbm_ipr.append(self.vbm_data[i][config-1,0])
              
          #print("CBM of configuration: " + str(config))
          cbm_ipr = []
          for i in range(len(self.cbm_data)):
              #print(str(i+self.cbm_init)+ " :   " + str(self.cbm_data[i][config-1,:]))
              en.append(self.cbm_data[i][config-1,1])
              ipr.append(self.cbm_data[i][config-1,0])
              cbm_ipr.append(self.cbm_data[i][config-1,0])
      
          
          return en, ipr, vbm_ipr, cbm_ipr  #Once the data are corrected this should be the correct "return"

      def plot_hist2d(self,x,y,xlabel,ylabel,bins,\
                      xlim=None, ylim=None, vlim=None, xticks=None, yticks=None,  \
                      normed=True, fig_suffix='ipr_v_energy', cmap=None,\
                      cbar_label="Probability density", hline=None, static=None):
          """
          This method plots a 2d hisotogram so that default area is 1.0
          x,y = Python lists containing x-values and y-values.
          fig_suffix = To differentiate different figure names --- an user specified suffix
          cbar_label = A string to label the colorbar
          xlabel, ylabel = Strings to label X and Y axis
          static = A 2D numpy array with 1st column x data, and 2nd column y-data
          hline = A float. Y-position of a horizontal line to show IPR cutoff used
          bins = a tuple with two elements defining no of x-bins and y-bins
          """
          #cbar = None
          plt.rc('font', size=self.font_size)
          plt.rc('axes', titlesize=self.font_size)
          #### New lines
          if xlim is not None:
             plt.xlim(np.array(xlim)) 
          if ylim is not None:
             plt.ylim(np.array(ylim))
          if vlim is None:
             vlim=(5,5000)
         
          rnge=np.array([[np.array(x).min(),np.array(x).max()],[np.array(y).min(),np.array(y).max()]]) 
          if (xlim is None) and (ylim is None):
             pass
          elif (xlim is None) and (ylim is not None):
             rnge[1] = np.array(ylim)
          elif (xlim is not None) and (ylim is None):
             rnge[0] = np.array(xlim)
          else:
             rnge[0] = np.array(xlim)
             rnge[1] = np.array(ylim)
        
          plot = plt.figure(fig_suffix)
          plt.xlabel(xlabel)
          plt.ylabel(ylabel)
          if xticks is not None:
             plt.xticks(xticks)
          if yticks is not None:
             plt.yticks(yticks)
          #h, xedge, yedge, image = plt.hist2d(x, y, bins=bins, range=rnge, density = normed, norm = colors.LogNorm(), cmap=cmap)
          h, xedge, yedge, image = plt.hist2d(x, y, bins=bins, range=rnge, density = normed, norm = colors.LogNorm(vmin=vlim[0],vmax=vlim[1]), cmap=cmap)
          cbar = plt.colorbar()
          cbar.set_label(cbar_label)
          if static is not None:
             plt.scatter(static[:,0], static[:,1], s=200, marker='*', color='black')
          #plot.colorbar()
          #plt.colorbar().set_label(cbar_label)
          if hline is not None:
             plt.axhline(y=0.0012, color='black', linestyle='--',linewidth=4)
          plot.savefig(self.fig_file_prefix + fig_suffix + ".png", dpi = self.dpi , bbox_inches = 'tight', pad_inches = 0.1)
          self.fig_counter += 1
          if not self.show_plot:
             plt.close()
          return h, xedge, yedge, image, plot
      

      def plot_prob_den(self,xedge,yedge,density, fig_suffix='cum_lt', log_cmap = False, cmap=None, \
                        cbar_label="Probability density", hline=None, static=None): 
          """
          This method replot the objects received from a plot_hist2d method, for example we can modify some aspects
          e.g. modify the height (h) which is represent as a colormap etc.
          xedge, yedge = Lists generated by matplotlib hist2d method. It contains all x-bins and y-bins
          density = a 2D list which we want to plot as a colormap, must be of the size len(xedge) X len(yedge)
          fig_suffix = A string To differentiate different figure names --- an user specified suffix
          cbar_label = A string to label the colorbar.
          """
          plot = None
          plot = plt.figure(fig_suffix)
          plt.xlim(xedge[0],xedge[-1])
          plt.ylim(yedge[0],yedge[-1])
          plt.rc('font', size=self.font_size)
          plt.rc('axes', titlesize=self.font_size)
          plt.xlabel("Orbital indices")
          plt.ylabel("IPR")
          if log_cmap:
             plt.pcolormesh(xedge, yedge, density.T, cmap=cmap, norm = colors.LogNorm())
          else:
             plt.pcolormesh(xedge, yedge, density.T, cmap=cmap)
          cbar = plt.colorbar()
          cbar.set_label(cbar_label)

          if static is not None:
             plt.scatter(static[:,0], static[:,1], s=200, marker='*', color='black')
          if hline is not None:
             plt.axhline(y=hline, color='black', linestyle='--',linewidth=4)

          plot.savefig(self.fig_file_prefix + "ipr_indices_" + fig_suffix + ".png",\
                        dpi = self.dpi , bbox_inches = 'tight', pad_inches = 0.1)

          self.fig_counter += 1
          if not self.show_plot:
             plt.close()
          return plot

      def calc_cum_prob_den(self,xedge,yedge,h, cum_prob_cut = 0.9, verbose=False):
          """
          This method takes the objects produced by plot_hist2d method and
               (i) re-normalizes the probability density along X asis so that sum of a column is 1
               (ii) computes cumulative probability density of greater than (gt) type and less than (lt) type.
               (iii) Based on the defined "cum_prob_cut", it prints the Y-values below which at least 
                     cum_prob_cut*100% observations are bound.
          Args:
               xedge, yedge, h = Lists generated by matplotlib hist2d method. It contains all x-bins, y-bins and heights
               cum_prob_cut = A float b/w 0.0 and 1.0. User defined cumulative density cutoff   
               verbose = If true it will print the probability densities.
          """
          bins = (len(xedge)-1, len(yedge)-1)
          cum_prob_den = np.cumsum(h*bins[0],axis=1)
          norm_fac = cum_prob_den[:,-1]
          h_renorm = np.zeros(h.shape, np.float)
          cum_prob_den_lt = np.zeros(cum_prob_den.shape, np.float)
          cum_prob_den_gt = np.zeros(cum_prob_den.shape, np.float)
          for i in range(bins[0]):
              cum_prob_den_lt[i,:] = cum_prob_den[i,:] / norm_fac[i]
              h_renorm[i,:] = h[i,:] * bins[0] / norm_fac[i]
              cum_prob_den_gt[i,0] = cum_prob_den[i,-1] / norm_fac[i]
              for j in range(1, bins[1]):
                  cum_prob_den_gt[i,j] = (cum_prob_den[i,-1] - cum_prob_den[i,j-1]) / norm_fac[i] 
             
             
          ipr_cut_value = np.array([])
          for i in range(bins[0]):
              if verbose:
                 print("Orbital-index: %d"%(self.vbm_init+i))
                 # print(cum_prob_den[0,i])
                 print("Grid-point IPR-Value  Probability_Density  Cumulative_Prob_density(LT | GT)")
                 for j in range(bins[1]):
                     print("%4d  |  %12.6f | %12.6f | %12.6f | %12.6f "\
                         %(j,yedge[j],h_renorm[i,j],cum_prob_den_lt[i,j],cum_prob_den_gt[i,j]))
              for j in range(bins[1]):
                  if cum_prob_den_lt[i,j] >= cum_prob_cut:
                     #print(yedge[j])
                     ipr_cut_value = np.append(ipr_cut_value, yedge[j])
                     break
          print("#Orbital-index    IPR-cutoff")
          for i in range(bins[0]):
              print("%4d   %12.6f"%(self.vbm_init+i,ipr_cut_value[i]))

          return h_renorm, cum_prob_den_gt, cum_prob_den_lt

      def prepare_plot_data(self):
          scatter_en = []
          scatter_ipr = []
          for i in range(1,self.nconfig+1):
              tmp_en, tmp_ipr, tmp_ipr_vbm, tmp_ipr_cbm = self.collect_eigval_ipr(i)
              scatter_en.append(tmp_en)
              scatter_ipr.append(tmp_ipr)
          
          x1 = np.array(scatter_en).flatten() - fermi_energy[self.sample]
          y1 = np.array(scatter_ipr).flatten()
          x2 = []; y2 = []
          for i in range(len(scatter_ipr)):
              for ind in range(len(scatter_ipr[i])):
                  orb_ind = self.vbm_init + ind
                  y2.append(scatter_ipr[i][ind])
                  x2.append(orb_ind)
          return x1, y1, x2, y2


