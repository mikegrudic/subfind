#!/usr/bin/env python
"""                                                                            
Algorithm that identifies the largest possible self-gravitating structures of a certain particle type. This, newer version relies on the load_from_snapshot routine from GIZMO.

Usage: SubFind.py <snapshots> ... [options]

Options:                                                                       
   -h --help                  Show this screen.

   --outputfolder=<name>      Specifies the folder to save the outputs to, None defaults to the same location as the snapshot [default: None]
   --ptype=<N>                GIZMO particle type to analyze [default: 4]
   --G=<G>                    Gravitational constant to use; should be consistent with what was used in the simulation. [default: 4.301e4]
   --boxsize=<L>              Box size of the simulation; for neighbour-search purposes. [default: None]
   --cluster_ngb=<N>          Length of particle's neighbour list. [default: 32]
   --nmin=<n>                 Minimum H number density to cut at, in cm^-3 [default: 0]
   --softening=<L>            Force softening for potential, if species does not have adaptive softening. [default: 5.6e-5]
   --fuzz=<L>                 Randomly perturb particle positions by this small fraction to avoid problems with particles at the same position in 32bit floating point precision data [default: 0]
   --alpha_crit=<f>           Critical virial parameter to be considered bound [default: 2.]
   --np=<N>                   Number of snapshots to run in parallel [default: 1]
   --ntree=<N>                Number of particles in a group above which PE will be computed via BH-tree [default: 10000]
   --overwrite                Whether to overwrite pre-existing clouds files [default: True]
   --units_already_physical   Whether to convert units to physical from comoving
   --max_linking_length=<L>   Maximum radius for neighbor search around a particle [default: 1e100]
   --potential_mode
"""
#   --snapdir=<name>           path to the snapshot folder, e.g. /work/simulations/outputs

#alpha_crit = 2

from __future__ import print_function
import load_from_snapshot #routine to load snapshots from GIZMo files
import h5py
from time import time
from time import sleep
from numba import jit, vectorize
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import numpy as np
from sys import argv
from os import sys
sys.setrecursionlimit(10000)
from glob import glob
from meshoid import Meshoid
from docopt import docopt
from multiprocessing import Pool
from collections import OrderedDict
import pykdgrav
from pykdgrav.treewalk import GetPotential
from pykdgrav.kernel import *
from pykdgrav import BruteForcePotential
from os import path
from mpl_toolkits.mplot3d import Axes3D
from os import getcwd
from os import mkdir
from natsort import natsorted
import cProfile
from numba import njit, jit
#potential_mode = False

@njit(fastmath=True)
def BruteForcePotential2(x_target,x_source, m,h=None,G=1.):
    if h is None: h = np.zeros(x_target.shape[0])
    potential = np.zeros(x_target.shape[0])
    for i in range(x_target.shape[0]):
        for j in range(x_source.shape[0]):
            dx = x_target[i,0]-x_source[j,0]
            dy = x_target[i,1]-x_source[j,1]
            dz = x_target[i,2]-x_source[j,2]
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            if r == 0: continue
#            if r>0: rinv = 1/r
            if r < h[j]:
                potential[i] += m[j] * PotentialKernel(r, h[j])
            else:
                if r>0: potential[i] -= m[j]/r
    return G*potential
#@jit
#def TotalEnergy(xc, mc, vc, hc, uc):
#    phic = Potential(xc, mc, hc)
#    v_well = vc - np.average(vc, weights=mc,axis=0)
#    vSqr = np.sum(v_well**2,axis=1)
#    return np.sum(mc*(0.5*vSqr + 0.5*phic + uc))

#@jit
def PotentialEnergy(xc, mc, vc, hc, uc, tree=None, particles_not_in_tree=None, x=None, m=None, h=None):
#    if len(xc) > 1e5: return 0 # this effective sets a cutoff in particle number so we don't waste time on things that are clearly too big to be a GMC
    if len(xc)==1: return -2.8*mc/hc**2 / 2
    if tree:
        phic = pykdgrav.Potential(xc, mc, hc, tree=tree, G=G)
        if particles_not_in_tree: # have to add the potential from the stuff that's not in the tree
            phic += BruteForcePotential2(xc, x[particles_not_in_tree], m[particles_not_in_tree], h=h[particles_not_in_tree], G=G)
    else:
        phic = BruteForcePotential(xc, mc, hc, G=G)
    return np.sum(mc*0.5*phic)


def InteractionPotential(x,m,h, group_a, tree_a, particles_not_in_tree_a, target_group):
    xb, mb, hb = x[target_group], m[target_group], h[target_group]
    if tree_a:
        phi = GetPotential(xb, tree_a, G=G,theta=.7)
        xa, ma, ha = np.take(x, particles_not_in_tree_a,axis=0), np.take(m, particles_not_in_tree_a,axis=0), np.take(h, particles_not_in_tree_a,axis=0)
        phi += BruteForcePotential2(xb, xa, ma, ha,G=G)
    else:
        xa, ma, ha = x[group_a], m[group_a], h[group_a]        
        phi = BruteForcePotential2(xb, xa, ma, ha,G=G)
#        print(phi, BruteForcePotential(xa, ma, ha, G=G), BruteForcePotential(xb, mb, hb, G=G))
    return phi



#@jit
def KineticEnergy(xc, mc, vc, hc, uc):
#    phic = Potential(xc, mc, hc)
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return np.sum(mc*(0.5*vSqr + uc))

#@jit
#def Potential(xc,mc,hc, tree=None):
#    if len(xc)==1: return -2.8*mc/hc
#    if len(xc) > 10000:
#        #phic = pykdgrav.Potential(xc, mc, hc, G=G, parallel=True)
#        phic = pykdgrav.Potential(xc, mc, hc, G=G, parallel=False)
#    else:
#        phic = BruteForcePotential(xc, mc, hc, G=G)
#    if tree: phic = 
#    phic = BruteForcePotential(xc, mc, hc, G=G)
#    return phic

#@jit
def KE(c, x, m, h, v, u):
    xc, mc, vc, hc = x[c], m[c], v[c], h[c]
    v_well = vc - np.average(vc, weights=mc,axis=0)
    vSqr = np.sum(v_well**2,axis=1)
    return (mc*(vSqr/2 + u[c])).sum()

def PE(c, x, m, h, v, u):
    if len(c) > 1000:
        phic = pykdgrav.Potential(x[c], m[c], h[c], G=G, theta=0.7)
    else:
        phic = BruteForcePotential(x[c], m[c], h[c], G=G)
    return 0.5*(phic*m[c]).sum()
    
def VirialParameter(c, x, m, h, v, u):
    ke, pe = KE(c,x,m,h,v,u), PE(c,x,m,h,v,u)
#    print(ke, pe)
    return(np.abs(2*ke/pe))
#    xc, mc, vc, hc = x[c], m[c], v[c], h[c]
 #   phic = pykdgrav.Potential(xc,mc, hc, G=G)
 #   v_well = vc - np.average(vc, weights=mc,axis=0)
 #   vSqr = np.sum(v_well**2,axis=1)
 #   return np.abs(2*(0.5*vSqr.sum() + u[c].sum())/phic.sum())

@njit
def Quadrupole(m, x):
    result = np.zeros((3,3))
    rSqr = np.zeros_like(m)
    for n in range(len(m)):
        rSqr[n] = x[n,0]*x[n,0] + x[n,1]*x[n,1] + x[n,2]*x[n,2]
    for i in range(3):
        for j in range(3):
            for n in range(len(m)):
                result[i,j] += 3 * m[n] * x[n,i] * x[n,j]
                if i == j: result[i,j] -= m[n] * rSqr[n]
    return result

#
def EnergyIncrement(i, c, m, M, x, v, u, h, v_com, tree=None, particles_not_in_tree = None):
    phi = 0.
    if particles_not_in_tree:
        xa, ma, ha = np.take(x,particles_not_in_tree,axis=0), np.take(m,particles_not_in_tree,axis=0), np.take(h,particles_not_in_tree,axis=0)
        phi += BruteForcePotential2(np.array([x[i],]), xa, ma, h=ha, G=G)[0]
    if tree:
        phi += G * pykdgrav.PotentialWalk(x[i], 0., tree, theta=0.7)
    vSqr = np.sum((v[i]-v_com)**2)
    mu = m[i]*M/(m[i]+M)
    return 0.5*mu*vSqr + m[i]*u[i] + m[i]*phi

#@jit
def KE_Increment(i,m,  v, u, v_com, mtot):
    vSqr = np.sum((v[i]-v_com)**2)
#    M = m[c].sum()
    mu = m[i]*mtot/(m[i]+mtot)
    return 0.5*mu*vSqr + m[i]*u[i]

#@jit
def PE_Increment(i, c, m, x, v, u, v_com):
    phi = -G * np.sum(m[c]/cdist([x[i],],x[c]))
    return m[i]*phi

def SaveArrayDict(path, arrdict):
    """Takes a dictionary of numpy arrays with names as the keys and saves them in an ASCII file with a descriptive header"""
    header = ""
    offset = 0
    
    for i, k in enumerate(arrdict.keys()):
        if type(arrdict[k])==list: arrdict[k] = np.array(arrdict[k])
        if len(arrdict[k].shape) == 1:
            header += "(%d) "%offset + k + "\n"
            offset += 1
        else:
            header += "(%d-%d) "%(offset, offset+arrdict[k].shape[1]-1) + k + "\n"
            offset += arrdict[k].shape[1]
            
    data = np.column_stack([b for b in arrdict.values()])
    data = data[(-data[:,0]).argsort()] 
    np.savetxt(path, data, header=header,  fmt='%.15g', delimiter='\t')


class Subhalo:
    def __init__(self):
        self.child1 = None
        self.child2 = None
        self.gravtree = None
        self.particles_not_in_tree = []
        self.particles_since_last_merge = []
        self.particles_at_last_merge = []
        self.bound_subset = []
        self.mother = None
        self.group = []
        self.phi = None
        self.COM = np.zeros(3)
        self.mass = 0
        self.VCOM = np.zeros(3)

    def append(self, i, pos, m, v):
        self.group.append(i)
        self.particles_not_in_tree.append(i)
        self.particles_since_last_merge.append(i)
        self.COM = (self.mass*self.COM + m[i]*pos[i])/(self.mass + m[i])
        self.VCOM = (self.mass*self.VCOM + m[i]*v[i])/(self.mass + m[i])
        self.mass += m[i]
    
    def SetChildren(self, child1, child2, x, m, h, phig):
        self.child1 = child1
        self.child2 = child2
        
        self.group = self.child1.group + self.child2.group
#        self.particles_not_in_tree = self.child1.particles_not_in_tree + self.child2.particles_not_in_tree
        self.particles_since_last_merge = [] #self.child1.particles_since_last_merge + self.child2.particles_since_last_merge
        self.particles_at_last_merge = self.group.copy()
        self.mass = self.child1.mass + self.child2.mass
        self.COM = (self.child1.mass * self.child1.COM + self.child2.mass * self.child2.COM)/self.mass
        self.VCOM = (self.child1.mass * self.child1.VCOM + self.child2.mass * self.child2.VCOM)/self.mass

        if self.child1:
            phig[self.child1.group] += InteractionPotential(x, m, h, self.child2.group, self.child2.gravtree, self.child2.particles_not_in_tree, self.child1.group)
        if self.child2:
            phig[self.child2.group] += InteractionPotential(x, m, h, self.child1.group, self.child1.gravtree, self.child1.particles_not_in_tree, self.child2.group)
        
        if len(self.child1.group) > len(self.child2.group):
            self.gravtree = self.child1.gravtree
            self.particles_not_in_tree = self.child1.particles_not_in_tree + self.child2.group
        else:
            self.gravtree = self.child2.gravtree
            self.particles_not_in_tree = self.child2.particles_not_in_tree + self.child1.group

        self.child1.gravtree = None
        self.child1.particles_not_in_tree = []
        self.child2.gravtree = None
        self.child2.particles_not_in_tree = []        
        
        if len(self.particles_not_in_tree) > 300:
            self.gravtree = pykdgrav.ConstructKDTree(x[self.group], m[self.group], h[self.group])
            self.particles_not_in_tree = []

    def UpdatePotential(self, phig, x, m, h):
        if len(self.group) < 2: return
        # for each group, need to update the potential - calculate the total potential from the group
        # for particles_since_last_merge, and the contribution to the rest from particles_since_last_merger

        x_target = np.take(x, self.particles_since_last_merge, axis=0)
        if self.gravtree:
            phig[self.particles_since_last_merge] += GetPotential(x_target, self.gravtree, G=G,theta=.7)

        xa, ma, ha = np.take(x, self.particles_not_in_tree, axis=0), np.take(m, self.particles_not_in_tree), np.take(h, self.particles_not_in_tree)

        phig[self.particles_since_last_merge] += BruteForcePotential2(x_target, xa, ma, ha,G=G)
        
        xa, ma, ha = np.take(x, self.particles_since_last_merge,axis=0), np.take(m, self.particles_since_last_merge), np.take(h, self.particles_since_last_merge)

        phig[self.particles_at_last_merge] += BruteForcePotential2(x[self.particles_at_last_merge], xa, ma, ha, G=G)
            
#        xa, ma, ha = np.take(x, self.group,axis=0), np.take(m, self.group,axis=0), np.take(h, self.group,axis=0)
#        phi_true = BruteForcePotential(xa, ma, ha, G=G)

        
    
    def GetBoundSubset(self, pos, m, v, h, u,phig):
        if len(self.group) < 2:
#            self.phi_bound = np.array([])
            return
        self.bound_subset = self.group.copy()
        phi_bound = np.take(phig,self.group)        
        mg, vg, hg, xg, ug = m[self.bound_subset], v[self.bound_subset], h[self.bound_subset], pos[self.bound_subset], u[self.bound_subset]
#        phi_true = BruteForcePotential(xg, mg, hg, G=G)
#        error = np.abs((phi_bound - phi_true))/np.abs(phi_true)
#        error_quantiles = np.percentile(error,[16,50, 84,100])
#        print("Error: ", error_quantiles)#, len(self.particles_since_last_merge), len(self.group), len(self.particles_not_in_tree), len(particles_before_last_merge), self.gravtree)                      
#        print("Getting bound subset...")
        while(len(self.bound_subset) > 0):
            if len(self.bound_subset) == 1: break
            max_bound = -1
            bound = 0 
            for w in mg*hg**-3, -phi_bound, mg, hg**-6:
                v_center = np.average(vg, axis=0, weights=w)
                vSqr = np.sum((vg - v_center)**2,axis=1)
                bound_test = (0.5*vSqr + phi_bound + ug < 0)
                if bound_test.sum() > max_bound:
                    bound = bound_test
                    max_bound = bound_test.sum()
                
            if bound.sum() == len(self.bound_subset): break
            elif bound.sum() < 2:
                self.bound_subset = np.array(self.bound_subset)[bound]
                break

            bound_indices = np.array(self.bound_subset)[bound]
#            bset = set(bound_indices)

            # want to economize past potential evaluations from previous bound subsets

            unbound = np.invert(bound)
            
            t = time()
     
            if unbound.sum() < bound.sum(): # cheaper to subtract the contributions of the few particles we've removed
                xu, mu, hu = xg[unbound], mg[unbound], hg[unbound]
                if unbound.sum() < 100:
                    phi_bound[bound] -= BruteForcePotential2(xg[bound], xu, mu, hu, G=G)
                else:
                    tree = pykdgrav.ConstructKDTree(xu, mu, hu)
                    phi_bound[bound] -= pykdgrav.Potential(xg[bound], mg[bound], hg[bound], G=G, tree=tree, theta=1.)#, parallel=parallel)
                phi_bound = phi_bound[bound]
            else:
                xb, mb, hb = xg[bound], mg[bound], hg[bound]
#                self.bound_tree = None
                if bound.sum() < 3000:
                    phi_bound = BruteForcePotential(xb, mb, hb, G=G)
                else:
                    phi_bound = pykdgrav.Potential(xb, mb, hb, G=G, theta=1.)#, parallel=parallel)
#            phi_true = BruteForcePotential(xg[bound], mg[bound], hg[bound], G=G)
#            error = np.abs((phi_bound - phi_true))/np.abs(phi_true)
#            error_quantiles = np.percentile(error,[16,50, 84,100])
#            print("Error: ", error_quantiles)#, len(self.particles_since_last_merge), len(self.group), len(self.particles_not_in_tree), len(particles_before_last_merge), self.gravtree)              
#            if len(self.bound_subset) > 200: print(time() - t, unbound.sum(), bound.sum(), len(self.bound_subset))        
            self.bound_subset = np.array(self.bound_subset)[bound]
            mg, vg, hg, xg, ug = m[self.bound_subset], v[self.bound_subset], h[self.bound_subset], pos[self.bound_subset], u[self.bound_subset]
            
 #       self.phi_bound = phi_bound
#        print("Bound subset of %d members"%len(self.bound_subset))
        return self.bound_subset
        

    def SetMother(self, mother):
        self.mother = mother

    def get_hierarchy(self, i, hierarchy):
        if not i in hierarchy.keys(): hierarchy[i] = [self.group]
        else: hierarchy[i].append(self.group)
        for c in self.child1, self.child2:
            if c is None: continue
            c.get_hierarchy(i+1, hierarchy)
        
    def print(self, i=0):
        if len(self.group) > 1: print(i, len(self.group))
        if self.child1 is not None:
            self.child1.print(i+1)
        if self.child2 is not None:
            self.child2.print(i+1)

    def GetMoments(self, m, x):
        M_tot = 0
        moment_tot = np.zeros(3)
        for c in self.child1, self.child2:
            if c is None: continue
            M, com = c.GetMoments(m, x)
            M_tot += M
            moment_tot += M*com
        if M_tot == 0:
            self.masses = m[self.group]
            self.positions = x[self.group]
            self.mass = self.masses.sum()
            self.COM = np.average(self.positions, axis=0, weights=self.masses)
        else:
            self.mass = M_tot
            self.COM = moment_tot / M_tot

        return self.mass, self.COM

#    def Potential(self, x_target):

#     def ComputePotential(self, x, m, h):
#         Ntot = len(self.group)
#         self.phi = np.zeros(len(self.group))
# #        print(len(self.group), len(self.child1.group), len(self.child2.group))
#         if self.child1:
#             self.phi[:len(self.child1.group)] = self.child1.ComputePotential(x,m,h)
#             self.gravtree1 = self.child1.gravtree
#             self.particles_not_in_tree1 = self.child1.particles_not_in_tree
#         if self.child2:
#             self.phi[len(self.child1.group):] = self.child2.ComputePotential(x,m,h)
#             self.gravtree2 = self.child2.gravtree
#             self.particles_not_in_tree2 = self.child2.particles_not_in_tree
#         else:
#             self.phi = BruteForcePotential(x[self.group], m[self.group], h[self.group])
#             return self.phi


#         self.phi[:len(self.child1.group)] += InteractionPotential(x,m,h, self.child1.group, self.child1.gravtree, self.child1.particles_not_in_tree, self.child2.group)
#         self.phi[len(self.child1.group):] += InteractionPotential(x,m,h, self.child2.group, self.child2.gravtree, self.child2.particles_not_in_tree, self.child1.group)
#         if len(self.child1.group) > len(self.child2.group):
#             self.gravtree = self.child1.gravtree
#             self.particles_not_in_tree = self.child1.particles_not_in_tree + self.child2.group
#         else:
#             self.gravtree = self.child2.gravtree
#             self.particles_not_in_tree = self.child2.particles_not_in_tree + self.child1.group
            

            
#         return self.phi
        
        

        
#            pot1 = self.child1.phi # self-potential of child 1
            #if self.child1.tree: tree1 = self.child1.tree
    def BoundSubhaloWalk2(self, m, v, x, h, u, rho, phi, smallest_bound_structure, level=0):
        bound_subset = self.bound_subset
        if len(bound_subset) >= 10:
            if potential_mode:
                owner = bound_subset[phi[bound_subset].argmin()]
            else:
                owner = bound_subset[rho[bound_subset].argmax()]
            for i in bound_subset:
                smallest_bound_structure[i] = owner

        for c in self.child1, self.child2:
            if c is not None: c.BoundSubhaloWalk2(m, v, x, h, u, rho, phi, smallest_bound_structure, level=level+1) #, old_bound_subset=bound_subset, phi_old=phi_old, old_group=old_group)
            
    def BoundSubhaloWalk(self, m, v, x, h, u, rho, smallest_bound_structure, level=0, old_bound_subset=None, phi_old=None, old_group=None):
#        phi_old = None
 #       old_group = None
        if old_bound_subset is not None and len(old_bound_subset) and set(old_bound_subset).issubset(self.group):
            bound_subset, phi_old, old_group = old_bound_subset, phi_old, old_group
        else:
            bound_subset, phi_old, old_group = BoundSubset(self.group, m, v, x, h, u, phi_old, old_group)
 
        if len(bound_subset) > 0:
            owner = bound_subset[rho[bound_subset].argmax()]
            for i in bound_subset:
                smallest_bound_structure[i] = owner

        for c in self.child1, self.child2:
            if c is not None: c.BoundSubhaloWalk(m, v, x, h, u, rho, smallest_bound_structure, level=level+1, old_bound_subset=bound_subset, phi_old=phi_old, old_group=old_group)

#@jit
def BoundSubset(group, m, v, x, h, u, phi_old=None, old_group=None):
    g = group
    mg, vg, hg, xg, ug = m[g], v[g], h[g], x[g], u[g]
    t = time()
    phi2 = None
    if phi_old is not None:
        excised = list(set(old_group).difference(g))
        if len(excised) < len(group):
            phi = phi_old[np.in1d(old_group, group)]
            if len(excised) < 10**3:
                # xg and phi must be in the same order
                phi -= BruteForcePotential2(xg, x[excised], m[excised], h=h[excised], G=G)
            else:
                tree = pykdgrav.ConstructKDTree(x[excised], m[excised], h[excised])
                phi -= pykdgrav.Potential(xg, mg, hg, G=G, tree=tree, theta=.7)
#            phi_true =  pykdgrav.Potential(xg, mg, hg, G=G,theta=.5)
#            if len(g) > 100: print(len(g), len(excised), "Error: ", np.percentile(np.abs((phi - phi_true))/np.abs(phi_true),[16,50,84]))
        else:
            if len(g) > 1e4:
                phi = pykdgrav.Potential(xg, mg, hg, G=G,theta=.7)
            else:
                phi = BruteForcePotential(xg, mg, hg, G=G)
#        phi2 = phi.copy()
    else:
        if len(g) > 1e4: 
            phi = pykdgrav.Potential(xg, mg, hg, G=G,theta=.7)
        else:
            phi = BruteForcePotential(xg, mg, hg, G=G)
    if len(g)> 100: print("Initial evaluation: ", time()- t, len(xg)) 
    old_phi = phi.copy()
    old_group = g.copy()
    while(len(g) > 0):
#        print(len(g))
        if len(g) == 1: break
        max_bound = -1
        bound = 0 
        for w in mg*hg**-3, -phi, mg, hg**-6:
         #phi**2)
            v_center = np.average(vg, axis=0, weights=w)
            vSqr = np.sum((vg - v_center)**2,axis=1)
            bound_test = (0.5*vSqr + ug + phi < 0)
            if bound_test.sum() > max_bound:
                bound = bound_test
                max_bound = bound_test.sum()
        
        if bound.sum() == len(g): break
        unbound = np.invert(bound)
        t = time()
        if unbound.sum() < 2000:
            phi[bound] -= BruteForcePotential2(xg[bound], xg[unbound], mg[unbound], h=hg[unbound], G=G)
        else:
            tree = pykdgrav.ConstructKDTree(xg[unbound], mg[unbound], hg[unbound])
            phi[bound] -= pykdgrav.Potential(xg[bound], mg[bound], hg[bound], G=G, tree=tree, theta=1.)
        print(time() - t, unbound.sum())        
        g = np.array(g)[bound]
        phi = phi[bound]
        mg, vg, hg, xg, ug = m[g], v[g], h[g], x[g], u[g]
    if len(g) == 1: g = []
    return g, old_phi, old_group

@jit
def AssignToHalo(largest_assigned_subhalo, halo):
    for l in halo.group:
        largest_assigned_subhalo[l] = halo
#def print_subhalo(subhalo, i):
#    print(i, len(subhalo.group))
#    for c in subhalo.child1, subhalo.child2:
#        if c is not None and len(c.group)>32: print_subhalo(subhalo, i+1)
        
#@jit

def ParticleGroups(x, m, rho, phi, h, u, v, zz, ids, cluster_ngb=32, rmax=1e100):
    if not potential_mode: phi = -rho
    hsml = (cluster_ngb * m * 3/ rho/ (4*np.pi))**(1./3)

    ngbdist, ngb = cKDTree(x).query(x,min(32, len(x)), distance_upper_bound=min(rmax, hsml.max()))

    max_group_size = 0
    assigned_group = -np.ones(len(x),dtype=np.int32)
    
    assigned_bound_group = -np.ones(len(x),dtype=np.int32)
    largest_assigned_subhalo = {} #-np.ones(len(x),dtype=np.int32)

    phig = np.zeros_like(phi)
    nmerger = 0
    for i in range(len(x)): # do it one particle at a time, in decreasing order of density
#        print(phi[i])
        if not i%10000: print(i)
        if np.any(ngb[i] > len(x) -1):
            s = Subhalo()
            s.append(i, x, m, v)
            largest_assigned_subhalo[i] = s
            continue 
        ngbi, ngbdisti = ngb[i][ngbdist[i] < hsml[i]], ngbdist[i][ngbdist[i] < hsml[i]]
        ngbi, ngbdisti = ngbi[1:], ngbdisti[1:]
        lower = phi[ngbi] < phi[i]

        if lower.sum():
            ngb_lower, ngbdist_lower = ngbi[lower], ngbdisti[lower]
            ngb_lower = ngb_lower[ngbdist_lower.argsort()]
            nlower = len(ngb_lower)
        else:
            nlower = 0

        if nlower == 0: # if this is the densest particle in the kernel, let's create our own group with blackjack and hookers
            s = Subhalo()
            s.append(i, x, m, v)
            largest_assigned_subhalo[i] = s
#particles_not_in_tree.append(i)
        elif nlower == 1 or largest_assigned_subhalo[ngb_lower[0]] == largest_assigned_subhalo[ngb_lower[1]]: # if there is only one denser particle, or both of the nearest two denser ones belong to the same group, we belong to that group too
            largest_assigned_subhalo[ngb_lower[0]].append(i, x, m, v)
            largest_assigned_subhalo[i] = largest_assigned_subhalo[ngb_lower[0]]
#            largest_assigned_subhalo[ngb_lower[0]].particles_not_in_tree.append(i)
        else:  #OK, we're at a saddle point, so we need to merge those groups
            nmerger += 1
            S1, S2 = largest_assigned_subhalo[ngb_lower[0]], largest_assigned_subhalo[ngb_lower[1]]

            S1.UpdatePotential(phig,x,m,h)
            S2.UpdatePotential(phig,x,m,h)

            S1.GetBoundSubset(x,m,v,h, u,phig)
            S2.GetBoundSubset(x,m,v,h, u,phig)

            new_group = Subhalo()
            new_group.SetChildren(S1, S2, x, m, h, phig)
            S1.SetMother(new_group)
            S2.SetMother(new_group)
            
            new_group.append(i, x, m, v)

            AssignToHalo(largest_assigned_subhalo, new_group)
#            for i in new_group.group:
#                largest_assigned_subhalo[i] = new_group

#    print("%d mergers performed"%nmerger)
    top_halos = list(set(largest_assigned_subhalo.values()))
    halo_size = np.array([len(T.group) for T in top_halos])
    top_halos = np.array(top_halos)[(-halo_size).argsort()]

    hierarchy = {}

    smallest_bound_structure = {}

    for t in top_halos:
        t.BoundSubhaloWalk2(m, v, x, h, u, rho, phi, smallest_bound_structure)

    groups = {}
    for i, owner in smallest_bound_structure.items():
        if owner in groups.keys(): groups[owner].append(i)
        else: groups[owner] = [i,]

#    for i, g in groups.items():
#        groups[i] = BoundSubset(g, m, v, x, h, u)[0]
#        print(len(g), VirialParameter(g,x, m, h, v, u))
    return groups
#    for g in groups.values():
#        print(m[g].sum())
#        print(m[v].sum())
        
#    Unbinding(top_halos[0], m, v, x, h, u)

#    exit()
#        print(len(t.group))
#    print([len(T.group) for T in top_halos])
#    print(top.group)
#    exit()
#    print(len(top.group), len(top.child1.group), len(top.child2.group))

#    return groups, bound_groups, assigned_group


def ComputeClouds(filepath , options):
    outputfolder = options["--outputfolder"]
    if ".hdf5" in filepath: # we have a lone snapshot, no snapdir
        snapnum = int(filepath.split("_")[-1].split(".hdf5")[0].split(".")[0].replace("/",""))
        snapname = filepath.split("_")[-2].split("/")[-1]
        snapdir = "/".join(filepath.split("/")[:-1])
        if outputfolder == "None": outputfolder = snapdir #getcwd() + "/".join(filepath.split("/")[:-1])
    else: # filepath refers to the directory in which the snapshot's multiple files are stored
        snapnum = int(filepath.split("snapdir_")[-1].replace("/",""))
        print(filepath)
        snapname = glob(filepath+"/*.hdf5")[0].split("_")[-2].split("/")[-1] #"snapshot" #filepath.split("_")[-2].split("/")[-1]
        print(snapname)
        snapdir = filepath.split("snapdir")[0] + "snapdir" + filepath.split("snapdir")[1]
#        print(snapnum, snapname, snapdir, outputfolder)
        if outputfolder == "None": outputfolder = getcwd() + filepath.split(snapdir)[0]

    if outputfolder == "": outputfolder = "."
    if outputfolder is not "None":
        if not path.isdir(outputfolder):
            mkdir(outputfolder)

    hdf5_outfilename = outputfolder + '/'+ "Subfind_%d.hdf5"%(snapnum,) #nmin, alpha_crit)
    dat_outfilename = outputfolder + '/' + "subfind_%d.dat"%(snapnum,)# nmin,alpha_crit)
    if path.isfile(dat_outfilename) and not options["--overwrite"]: return
            
    if not snapdir:
        snapdir = getcwd()
        print('Snapshot directory not specified, using local directory of ', snapdir)

    fname_found, _, _ =load_from_snapshot.check_if_filename_exists(snapdir,snapnum, snapshot_name=snapname)
    if fname_found!='NULL':    
        print('Snapshot ', snapnum, ' found in ', snapdir)
    else:
        print('Snapshot ', snapnum, ' NOT found in ', snapdir, '\n Skipping it...')
        return
    cluster_ngb = int(float(options["--cluster_ngb"]) + 0.5)
    G = float(options["--G"])
    boxsize = options["--boxsize"]
    ptype = int(options["--ptype"])

    #recompute_potential = options["--recompute_potential"]
    softening = float(options["--softening"])
    if boxsize != "None":
        boxsize = float(boxsize)
    else:
        boxsize = None
    fuzz = float(options["--fuzz"])
    npart = load_from_snapshot.load_from_snapshot("NumPart_Total", "Header", snapdir, snapnum, snapshot_name=snapname)[ptype]
    print(npart)
    if npart < cluster_ngb:
        print("Not enough particles for meaningful cluster analysis!")
        return
    
    #Read gas properties
    keys = load_from_snapshot.load_from_snapshot("keys",ptype,snapdir,snapnum, snapshot_name=snapname)

    if keys is 0:
        print("No keys found, noping out!")        
        return

    criteria = np.ones(npart,dtype=np.bool) # now we refine by particle density
    if "Density" in keys:
        rho = load_from_snapshot.load_from_snapshot("Density",ptype,snapdir,snapnum, snapshot_name=snapname, units_to_physical=(not options["--units_already_physical"]))
        if len(rho) < cluster_ngb:
            print("Not enough particles for meaningful cluster analysis!")
            return

    else: # we have to do a kernel density estimate for e.g. dark matter or star particles
        m = load_from_snapshot.load_from_snapshot("Masses",ptype,snapdir,snapnum, snapshot_name=snapname)
        if len(m) < cluster_ngb:
            print("Not enough particles for meaningful cluster analysis!")
            return
        x = load_from_snapshot.load_from_snapshot("Coordinates",ptype,snapdir,snapnum, snapshot_name=snapname)
        print("Computing density...")
        rho = Meshoid(x,m,des_ngb=cluster_ngb).Density()
        print("Density done!")
        criteria = np.arange(len(rho))[(rho*404 > nmin)] # only look at dense gas (>nmin cm^-3)
#        print("%g particles denser than %g cm^-3" %(criteria.size,nmin))  #(np.sum(rho*147.7>nmin), nmin))
#        if not criteria.size:
#            print('No particles dense enough, exiting...')
#            return
#        m = np.take(m, criteria, axis=0)
#        x = np.take(x, criteria, axis=0)
        
    criteria = np.arange(len(rho))[rho*404 > nmin] # only look at dense gas (>nmin cm^-3)
    print("%g particles denser than %g cm^-3" % (criteria.size,nmin))  #(np.sum(rho*147.7>nmin), nmin))
    if not criteria.size > cluster_ngb:
        print('Not enough dense particles, exiting...')
        return
#        print(x, load_from_snapshot.load_from_snapshot("Coordinates",ptype,snapdir,snapnum, snapshot_name=snapname, particle_mask=criteria))
    rho = np.take(rho, criteria, axis=0)
    rho_order = (-rho).argsort()
    rho = rho[rho_order]
    particle_data = {"Density": rho} # now let's store all particle data that satisfies the criteria
    for k in keys:
#        print(k)
        if not k in particle_data.keys():
            particle_data[k] = load_from_snapshot.load_from_snapshot(k,ptype,snapdir,snapnum, snapshot_name=snapname, particle_mask=criteria, units_to_physical=(not options["--units_already_physical"]))[rho_order]
    
    if "StarFormationRate" in keys: print("Total SFR: %g"%particle_data["StarFormationRate"].sum())
    m = particle_data["Masses"]
    x = particle_data["Coordinates"]
    ids = particle_data["ParticleIDs"] #load_from_snapshot.load_from_snapshot("ParticleIDs",ptype,snapdir,snapnum, particle_mask=criteria)
    u = (particle_data["InternalEnergy"] if ptype == 0 else np.zeros_like(m))
    if "MagneticField" in keys:
        energy_density_code_units = np.sum(particle_data["MagneticField"]**2,axis=1) / 8 / np.pi * 5.879e9
        specific_energy = energy_density_code_units / rho
        u += specific_energy
        
    zz = (particle_data["Metallicity"] if "Metallicity" in keys else np.zeros_like(m))
    v = particle_data["Velocities"]
    if "StarFormationRate" in keys: sfr = particle_data["StarFormationRate"]
    else: sfr = np.zeros_like(m)

    if "AGS-Softening" in keys:
        hsml = particle_data["AGS-Softening"]
    elif "SmoothingLength" in keys:
        hsml = particle_data["SmoothingLength"] 
    else:
#        hsml = (cluster_ngb * m * 3/ rho/ (4*np.pi))**(1./3) #
        hsml = np.ones_like(m)*softening

    while len(np.unique(x,axis=0)) < len(x): # make sure no two particles are at the same position
        x *= 1+ np.random.normal(size=x.shape) * 1e-8
        
    if "Potential" in keys: # potential doesn't get used anymore, so this is moot
        phi = particle_data["Potential"] #load_from_snapshot.load_from_snapshot("Potential",ptype,snapdir,snapnum, particle_mask=criteria)
    else:
        print('Potential not available in snapshot, calculating...')
        if potential_mode:
            phi = pykdgrav.Potential(x, m, hsml, G=G, theta=0.5)
        else:
            phi = np.zeros_like(m)
#        print('Potential calculation finished')
    if potential_mode:
        phi_order = phi.argsort()
        x, m, rho, phi, hsml, u, v, zz, ids = np.float64(x)[phi_order], np.float64(m)[phi_order], np.float64(rho)[phi_order], np.float64(phi)[phi_order], np.float64(hsml)[phi_order], np.float64(u)[phi_order], np.float64(v)[phi_order], np.float64(zz)[phi_order], ids[phi_order]
        for k in particle_data.keys():
            particle_data[k] = particle_data[k][phi_order]

    x, m, rho, phi, hsml, u, v, zz = np.float64(x), np.float64(m), np.float64(rho), np.float64(phi), np.float64(hsml), np.float64(u), np.float64(v), np.float64(zz)


    t = time()
    bound_groups = ParticleGroups(x, m, rho, phi, hsml, u, v, zz, ids, cluster_ngb=cluster_ngb, rmax=float(options["--max_linking_length"]))
    t = time() - t
    print("Time: %g"%t)
    print("Done grouping. Computing group properties...")
    groupmass = np.array([m[c].sum() for c in bound_groups.values() if len(c)>3])
    groupsfr = np.array([sfr[c].sum() for c in bound_groups.values() if len(c)>3])
    print("Total SFR in clouds: ",  groupsfr.sum())
    groupid = np.array([c for c in bound_groups.keys() if len(bound_groups[c])>3])
    groupid = groupid[groupmass.argsort()[::-1]]
    bound_groups = OrderedDict(zip(groupid, [bound_groups[i] for i in groupid]))
#    exit()

    # Now we analyze the clouds and dump their properties

    bound_data = OrderedDict()
    bound_data["Mass"] = []
    bound_data["Center"] = []
    bound_data["PrincipalAxes"] = []
    bound_data["Reff"] = []

    bound_data["HalfMassRadius"] = []
    bound_data["ProjHalfMassRadius0"] = []
    bound_data["ProjHalfMassRadius1"] = []
    bound_data["ProjHalfMassRadius2"] = []
    bound_data["NumParticles"] = []
    bound_data["VirialParameter"] = []
#    bound_data["SigmaEff"] = []
    
    
    print(hdf5_outfilename)
    Fout = h5py.File(hdf5_outfilename, 'w')
#    Fout.create_group("Header",data=

    i = 0
#    fids = load_from_snapshot.load_from_snapshot("ParticleIDs",ptype,snapdir,snapnum, particle_mask=criteria)
    fids = ids
    #Store all keys in memory to reduce I/O load
#    print '\t Reading all data for Particle Type ', ptype

#    print '\t Reading done, iterating over clouds...'
    for k,c in bound_groups.items():
 #       print(len(c), len(np.unique(c)))
        bound_data["Mass"].append(m[c].sum())
        bound_data["NumParticles"].append(len(c))
        bound_data["Center"].append(np.average(x[c], weights=m[c], axis=0))
        dx = x[c] - bound_data["Center"][-1]
        eig = np.linalg.eig(np.cov(dx.T))[0]
        bound_data["PrincipalAxes"].append(np.sqrt(eig))
        r = np.sum(dx**2, axis=1)**0.5
        bound_data["HalfMassRadius"].append(np.median(r))
        for dim in range(3):
            r2d = np.sqrt(dx[:,dim]**2 + dx[:,(dim+1)%3]**2)
            bound_data["ProjHalfMassRadius" + str(dim)].append(np.median(r2d))        
        bound_data["Reff"].append(np.sqrt(5./3 * np.average(r**2,weights=m[c])))
#        sigma_eff = meshoid.meshoid(x[c],m[c],hsml[c]).SurfaceDensity(size=4*bound_data["HalfMassRadius"][-1],center=bound_data["Center"][-1], res=400)
        
#        bound_data["SigmaEff"].append(np.average(sigma_eff,weights=sigma_eff)*1e4)
#        print(len(c))
        bound_data["VirialParameter"].append(VirialParameter(c, x, m, hsml, v, u))

        cluster_id = "Cloud"+ ("%d"%i).zfill(int(np.log10(len(bound_groups))+1))

        N = len(c)

        Fout.create_group(cluster_id)
#        idx = np.in1d(fids, fids[c])
        for k in keys: #range(len(keys)):
#            k = keys[j]
            Fout[cluster_id].create_dataset('PartType'+str(ptype)+"/"+k, data = particle_data[k].take(c,axis=0))
        i += 1
#        print "\t \t ",cluster_id

    print("Done grouping bound clusters!")

       
    Fout.close()
    
    #now save the ascii data files
#        datafile_name = "bound_%s_n%g_alpha%g.dat"%(n,nmin,alpha_crit)
#    dat_outfilename = outputfolder + '/' +"bound_%d_n%g_alpha%g.dat"%(snapnum, nmin,alpha_crit)
    SaveArrayDict(dat_outfilename, bound_data)
#    SaveArrayDict(filename.split("snapshot")[0] + "unbound_%s.dat"%n, unbound_data)

nmin = float(docopt(__doc__)["--nmin"])
alpha_crit = float(docopt(__doc__)["--alpha_crit"])
#substructure_mode = docopt(__doc__)["--substructure_mode"]
G = float(docopt(__doc__)["--G"])
overwrite =  docopt(__doc__)["--overwrite"]
potential_mode = docopt(__doc__)["--potential_mode"]
ntree = int(docopt(__doc__)["--ntree"])

def func(path):
    """Necessary for the multiprocessing pickling to work"""
    return ComputeClouds(path, docopt(__doc__))

def main():
    options = docopt(__doc__)
#    print(options)
    nproc=int(options["--np"])
#    snapnum_list = np.array([int(c) for c in options["<snapshots>"][0].split(',')])

    snappaths = [p  for p in options["<snapshots>"]] 
#    snappaths = "snapdir_600",
#    snappaths = ["M4e6_R100_S0_T1_B0.01_Res100_n2_sol0.5_1/output/snapshot_060.hdf5",]
    if nproc==1:
        for f in snappaths:
            print(f)
            ComputeClouds(f, options)
#            cProfile.runctx("ComputeClouds(f, options)", {'ComputeClouds': ComputeClouds, 'f': f, 'options': options}, {})
    else:
        Pool(nproc).map(func, snappaths)
#        Parallel(n_jobs=nproc)(delayed(ComputeClouds)(f,options) for f in snappaths)

if __name__ == "__main__": main()
