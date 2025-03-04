#!/usr/bin/env python

import numpy as np 
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import spatial
from scipy import ndimage
from scipy import signal
import time

import argparse

# correlations.py
# functions for computing spatial auto- and cross- correlation functions between cells in a spatial dataset
# Richard Mebane 2023

##############################################################
# Spatial correlation functions
##############################################################

#print('Imported nextflow branch...')

# rs is r array in coordinate space
# result is an array of xi of length of rs
# total_counts is the total counts of d1 in the full sample
# if total_counts == None, assume this is the full sample
def cf(d1, d2, rs, area, total_counts=None):
	n_avg = len(d2) / area
	distances = spatial.distance.cdist(d1, d2, 'euclidean')
	#print(distances)
	xis = []
	for r in rs:
		if(len(d1) == 0):
			xis.append(0.0)
		else:
			counts = np.count_nonzero(distances <= r) / len(d1)
			if(len(d2) == 0):
				xi = -1.0
			else:
				xi = counts / (np.pi * r**2 * n_avg) - 1.0
			if(total_counts != None):
				xi *= len(d1) / total_counts
			xis.append(xi)
	return xis

# rs is r array in coordinate space
# result is an array of xi of length of rs
# total_counts is the total counts of d1 in the full sample
# if total_counts == None, assume this is the full sample
def cf_and_neighbor(d1, d2, rs, area, total_counts=None):
	n_avg = len(d2) / area
	distances = spatial.distance.cdist(d1, d2, 'euclidean')
	#print(distances)
	xis = []
	for r in rs:
		if(len(d1) == 0):
			xis.append((0.0, 0))
		else:
			counts = np.count_nonzero(distances <= r) / len(d1)
			# get array of counts for each node in d1
			acounts = np.count_nonzero(distances <= r, axis=1)
			# how many detect at least 1? (this is an average value)
			Ndetected = np.count_nonzero(acounts > 0) / len(d1)
			if(len(d2) == 0):
				xi = -1.0
			else:
				xi = counts / (np.pi * r**2 * n_avg) - 1.0
			if(total_counts != None):
				xi *= len(d1) / total_counts
			xis.append((xi, Ndetected))
	return xis

# compute the distribution of node in the fovs of each tissue given
# result is a dictionary mapping biopsy names to a list of counts for each fov in that biopsy
def count_nodes(df, node, tissues=[], type_name = 'node_name', tissue_name = 'biopsy', obs_name = 'Unnamed: 0'):
	if(len(tissues) == 0):
		tissues = df[tissue_name].to_numpy()
		tissues = np.unique(tissues)
	fov_labels = df[obs_name].to_numpy()
	reduced_labels = []
	for l in fov_labels:
		reduced_labels.append(int(np.char.split(l, sep="_").tolist()[0]))
	reduced_labels = np.array(reduced_labels)
	df['fov_number'] = reduced_labels
	# count up the number of unique fovs in each tissue
	label_list = []
	for t in tissues:
		cells = df.loc[df[tissue_name] == t]
		labels = cells['fov_number'].to_numpy().astype(int)
		labels = np.unique(labels)
		label_list.append(labels)
	result = {}
	i = 0
	while(i < len(tissues)):
		t = tissues[i]
		counts = []
		for l in label_list[i]:
			cells = df.loc[(df[tissue_name] == t) & (df[type_name] == node) & (df['fov_number'] == l)]
			counts.append(len(cells))
		result[t] = (np.sum(counts), counts)
		i += 1
	return result

# returns cross correlation functions for each node pair in pairs at each r in rs
# df is data frame of nodes
# area_dict is a dictionary mapping tissues to their sample area in coordinate space
# if fov != None, only look at cells in that fov FOR THE FIRST NODE ONLY (i.e., center of each filter)
def compute_correlation_pairs(df, pairs, rs, area_dict, row_name = 'x', col_name = 'y', tissue_name = 'biopsy', type_name = 'niche', type_name_2 = 'subniche', fov_name = 'fov', tissues = [], fov = None):
	if(len(tissues) == 0):
		tissues = df[tissue_name].to_numpy()
		tissues = np.unique(tissues)
	xis = np.zeros((len(pairs), len(rs) ) )
	i = 0 
	for p in pairs:
		total_counts = len(df.loc[df[type_name] == p[0]])
		for t in tissues:
			if(fov == None):
				cells1 = df.loc[(df[type_name] == p[0]) & (df[tissue_name] == t)]
			else:
				cells1 = df.loc[(df[type_name] == p[0]) & (df[tissue_name] == t) & (df[fov_name] == fov)]
				total_counts = len(cells1)
			cells2 = df.loc[(df[type_name_2] == p[1]) & (df[tissue_name] == t)]
			rows1 = cells1[row_name].to_numpy().astype(int)
			cols1 = cells1[col_name].to_numpy().astype(int)
			rows2 = cells2[row_name].to_numpy().astype(int)
			cols2 = cells2[col_name].to_numpy().astype(int)
			coords1 = np.column_stack((rows1, cols1))
			coords2 = np.column_stack((rows2, cols2))
			area = area_dict[t]
			if(len(coords1) != 0):
				xis[i] += cf(coords1, coords2, rs, area, total_counts = total_counts)
		i += 1
	return(xis) # xis[i] has N members, where N in the number of radii provided

def compute_corrs_fov(df, pairs, radii, obs_name = 'Unnamed: 0', row_name = 'global_coords_1', col_name = 'global_coords_2', tissue_name = 'biopsy', type_name = 'node_name', fov_name = 'fov_number'):
	area = 5472.*3648
	fov_area = 5472.*3648 # these are the dimensions of the cell label mask images
	area_dict = {}
	area_dict['fov_area'] = fov_area
	# base
	area_dict['h02A'] = fov_area * 23.
	area_dict['h12A'] = fov_area * 14.
	area_dict['h30A'] = fov_area * 4.
	area_dict['h43A'] = fov_area * 10.
	area_dict['h47A'] = fov_area * 10.
	area_dict['h16A'] = fov_area * 4.
	# PD1
	area_dict['h02B'] = fov_area * 20.
	area_dict['h12B'] = fov_area * 23.
	area_dict['h15B'] = fov_area * 4.
	area_dict['h16B'] = fov_area * 4.
	area_dict['h17B'] = fov_area * 3.
	area_dict['h43B'] = fov_area * 16.
	area_dict['h47B'] = fov_area * 3.
	# RTPD1
	area_dict['h02C'] = fov_area * 7.
	area_dict['h12C'] = fov_area * 21.
	area_dict['h16C'] = fov_area * 4.
	area_dict['h43C'] = fov_area * 5.
	area_dict['h47C'] = fov_area * 7.
	tissues = df[tissue_name].to_numpy()
	tissues = np.unique(tissues)
	fov_labels = df[obs_name].to_numpy()
	reduced_labels = []
	for l in fov_labels:
		reduced_labels.append(int(np.char.split(l, sep="_").tolist()[0]))
	reduced_labels = np.array(reduced_labels)
	df[fov_name] = reduced_labels
	# count up the number of unique fovs in each tissue
	label_list = []
	for t in tissues:
		cells = df.loc[df[tissue_name] == t]
		labels = cells[fov_name].to_numpy().astype(int)
		labels = np.unique(labels)
		label_list.append(labels)
	results = {} # results[i] contains a tuple with (xis, weights) where xis is the array of xis for each pair in this sample and weights is the total weight compared to the whole sample in tissues
	i = 0
	while(i < len(tissues)):
		t = tissues[i]
		#print('Biopsy: ' + str(t) + ', n fovs: ' + str(len(label_list[i])))
		for l in label_list[i]:
			key = str(t) + '_' + str(l)
			#if(key not in results):
			#	results[key] = []
			results[key] = compute_correlation_pairs(df, pairs, radii, area_dict, tissues = [t], fov=l)
		i += 1
	return results

# take the results from compute_corrs_fov and get the mean xis and errors for all fovs with counts above min_counts
def compute_mean_fov_corr(df, pairs, xis, min_counts = 0, obs_name = 'Unnamed: 0', row_name = 'global_coords_1', col_name = 'global_coords_2', tissue_name = 'biopsy', type_name = 'node_name', fov_name = 'fov_number'):
	Nrs = len(xis[list(xis.keys())[0]][0])
	tissues = df[tissue_name].to_numpy()
	tissues = np.unique(tissues)
	fov_labels = df[obs_name].to_numpy()
	reduced_labels = []
	for l in fov_labels:
		reduced_labels.append(int(np.char.split(l, sep="_").tolist()[0]))
	reduced_labels = np.array(reduced_labels)
	df[fov_name] = reduced_labels
	# count up the number of unique fovs in each tissue
	label_list = []
	for t in tissues:
		cells = df.loc[df[tissue_name] == t]
		labels = cells[fov_name].to_numpy().astype(int)
		labels = np.unique(labels)
		label_list.append(labels)
	res_xi = np.zeros((len(pairs), Nrs))
	res_errors = np.zeros((len(pairs), Nrs))
	totals = np.zeros(len(pairs)) # keep track of the total # of first nodes in fovs which are above min_counts
	j = 0
	for p in pairs:
		#print(p)
		i = 0
		dist = []
		while(i < len(tissues)):
			t = tissues[i]
			for l in label_list[i]:
				N1 = len(df.loc[(df[fov_name] == l) & (df[tissue_name] == t) & (df[type_name] == p[0])])
				N2 = len(df.loc[(df[fov_name] == l) & (df[tissue_name] == t) & (df[type_name] == p[1])])
				if((N1 >= min_counts) & (N2 >= min_counts)):
					totals[j] += N1
					key = str(t) + '_' + str(l)
					results = xis[key][j]
					k = 0
					res_tuple = []
					while(k < Nrs):
						res_xi[j][k] += results[k] * N1 # remember to divide everything by the total at the end
						res_tuple.append(results[k])
						k += 1
					dist.append(res_tuple)
			i += 1
		dist = np.transpose(dist)
		k = 0
		while(k < Nrs):
			if(len(dist > 0)):
				res_errors[j][k] = np.std(dist[k])
			else:
				res_errors[j][k] = -1
			k += 1
		if(totals[j] > 0):
			res_xi[j] /= totals[j]
		j += 1
	return (res_xi, res_errors)


# returns the individual jackknifed correlation functions
# default is to return xi for each fov in df
def get_jk_correlations(df, pairs, rs, area_dict, obs_name = 'Unnamed: 0', row_name = 'x', col_name = 'y', tissue_name = 'biopsy', type_name = 'niche', type_name_2 = 'subniche', tissues = [], add_fov_labels=False):
	if(len(tissues) == 0):
		tissues = df[tissue_name].to_numpy()
		tissues = np.unique(tissues)
	if(add_fov_labels):
		fov_labels = df[obs_name].to_numpy()
		reduced_labels = []
		for l in fov_labels:
			reduced_labels.append(int(np.char.split(l, sep="_").tolist()[0]))
		reduced_labels = np.array(reduced_labels)
		df['fov'] = reduced_labels
	# count up the number of unique fovs in each tissue
	label_list = []
	for t in tissues:
		cells = df.loc[df[tissue_name] == t]
		labels = cells['fov'].to_numpy().astype(int)
		labels = np.unique(labels)
		label_list.append(labels)
	i = 0
	results = [] # results[i] contains a tuple with (xis, weights) where xis is the array of xis for each pair in this sample and weights is the total weight compared to the whole sample in tissues
	while(i < len(tissues)):
		t = tissues[i]
		print('Computing correlations for biopsy ' + str(t))
		for l in label_list[i]:
			cells = df.loc[(df[tissue_name] == t)] # we'll filter out the irrelevant fovs in the next function
			xis = compute_correlation_pairs(cells, pairs, rs, area_dict, row_name = row_name, col_name = col_name, tissue_name = tissue_name, type_name = type_name, type_name_2 = type_name_2, fov_name = 'fov', fov=l)
			weights = []
			for p in pairs:
				total_count = len(df.loc[(df[type_name] == p[0]) & (df[tissue_name].isin(tissues))])
				local_count = len(cells.loc[(cells[type_name] == p[0]) & (cells['fov'] == l)])
				if(total_count == 0):
					weights.append(0.0)
				else:
					weights.append(1.0 * local_count / total_count)
			results.append((xis, weights))
		i += 1
	return results

# computes the cross correlation functions for each pair, keeping track of individual fovs to compute the jacknife errors
def compute_correlations_jk(df, pairs, rs, area_dict, obs_name = 'Unnamed: 0', row_name = 'x', col_name = 'y', tissue_name = 'biopsy', type_name = 'niche', type_name_2 = 'subniche', tissues = [], return_samples=False, add_fov_labels=False, return_full_samples=False):
	ti = time.time() # initial time
	errors = np.zeros((len(pairs), len(rs)))
	results = np.zeros((len(pairs), len(rs)))
	jk_samples = get_jk_correlations(df, pairs, rs, area_dict, obs_name=obs_name, row_name=row_name, col_name=col_name, tissue_name=tissue_name, type_name=type_name, type_name_2 = type_name_2, tissues=tissues, add_fov_labels=add_fov_labels)
	i = 0
	if(return_full_samples):
		return jk_samples
	jk_results = np.zeros((len(jk_samples), len(pairs), len(rs))).tolist()
	# jk_results[i] gives the results from the ith jackknife sample
	# jk_results[i][j] gives the jth pair
	# jk_results[i][j][k] gives the kth radius
	while(i < len(jk_samples)):
		weights = jk_samples[i][1] # these are the weights of the sample we WON'T use
		j = 0
		while(j < len(jk_samples)):
			if(j != i): # compute the average correlation functions including everything but the current sample
				xi_results = np.copy(jk_samples[j][0])
				corrected_weights = jk_samples[j][1] + np.array(weights) / (len(jk_samples) - 1)
				# loop over all pairs and multiply by weights
				k = 0
				while(k < len(xi_results)):
					xi_results[k] *= corrected_weights[k]
					k += 1
				jk_results[i] += xi_results
			j += 1
		i += 1
	i = 0
	while(i < len(pairs)):
		j = 0
		while(j<len(rs)):
			xis = []
			for res in jk_results:
				xis.append(res[i][j])
			errors[i][j] = np.std(xis)
			# also compute the total function from all samples
			k = 0
			while(k < len(jk_samples)):
				results[i][j] += jk_samples[k][0][i][j] * jk_samples[k][1][i]
				k += 1
			j += 1
		i += 1
	print('Total computation time: ' + str(round(time.time() - ti, 2)) + ' seconds')
	# if flag is set, return the full jackknifed sample results
	# default is False since this could take up a lot of space
	if(return_samples):
		return (results, errors, jk_results)
	return (results, errors)

# right now this just assumes that df only has data from one treatment
def create_xi_table(df, niche_pairs, subniche_pairs, cross_pairs, ct_pairs, radii, area_dict, return_samples=False):
	res_dict = {}
	res_dict['r'] = radii
	if(return_samples):
		print('Computing niche-niche correlations...')
		niche_samples = compute_correlations_jk(df, niche_pairs, radii, area_dict, type_name_2='niche', return_full_samples=True)
		print('Computing subniche-subniche correlations...')
		subniche_samples = compute_correlations_jk(df, subniche_pairs, radii, area_dict, type_name='subniche', return_full_samples=True)
		print('Computing niche-subniche correlations...')
		cross_samples = compute_correlations_jk(df, cross_pairs, radii, area_dict, return_full_samples=True)
		print('Computing celltype-celltype correlations...')
		ct_samples = compute_correlations_jk(df, ct_pairs, radii, area_dict, type_name='celltype', type_name_2 ='celltype', return_full_samples=True)
		print('Building result table...')
		results = {}
		i = 0
		for l in niche_pairs:
			xis = []
			weights = []
			j = 0
			while(j < len(niche_samples)):
				xis.append(niche_samples[j][0][i][0])
				weights.append(niche_samples[j][1][i])
				j += 1
			results[l] = (xis, weights)
			i += 1
		i = 0
		for l in subniche_pairs:
			xis = []
			weights = []
			j = 0
			while(j < len(subniche_samples)):
				xis.append(subniche_samples[j][0][i][0])
				weights.append(subniche_samples[j][1][i])
				j += 1
			results[l] = (xis, weights)
			i += 1
		i = 0
		for l in cross_pairs:
			xis = []
			weights = []
			j = 0
			while(j < len(cross_samples)):
				xis.append(cross_samples[j][0][i][0])
				weights.append(cross_samples[j][1][i])
				j += 1
			results[l] = (xis, weights)
			i += 1
		i = 0
		for l in ct_pairs:
			xis = []
			weights = []
			j = 0
			while(j < len(ct_samples)):
				xis.append(ct_samples[j][0][i][0])
				weights.append(ct_samples[j][1][i])
				j += 1
			results[l] = (xis, weights)
			i += 1
		return results


	# Else do the real calculation
	# print('Computing niche-niche correlations...')
	# xi_niche, error_niche = compute_correlations_jk(df, niche_pairs, radii, area_dict, type_name_2='niche')
	# print('Computing subniche-subniche correlations...')
	# xi_subniche, error_subniche = compute_correlations_jk(df, subniche_pairs, radii, area_dict, type_name='subniche')
	# print('Computing niche-subniche correlations...')
	# xi_cross, error_cross = compute_correlations_jk(df, cross_pairs, radii, area_dict)
	print('Computing celltype-celltype correlations...')
	xi_ct, error_ct = compute_correlations_jk(df, ct_pairs, radii, area_dict, type_name='celltype', type_name_2 ='celltype')
	print('Building result table...')
	i = 0
	# for p in niche_pairs:
	# 	#res_dict[str(p)] = np.transpose(np.array((xi_niche[i], error_niche[i])))
	# 	res_dict[str(p)] = list(zip(xi_niche[i], error_niche[i]))
	# 	i += 1
	# i = 0
	# for p in subniche_pairs:
	# 	#res_dict[str(p)] = np.transpose(np.array((xi_subniche[i], error_subniche[i])))
	# 	res_dict[str(p)] = list(zip(xi_subniche[i], error_subniche[i]))
	# 	i += 1
	# i = 0
	# for p in cross_pairs:
	# 	#res_dict[str(p)] = np.transpose(np.array((xi_cross[i], error_cross[i])))
	# 	res_dict[str(p)] = list(zip(xi_cross[i], error_cross[i]))
	# 	i += 1
	i = 0
	for p in ct_pairs:
		#res_dict[str(p)] = np.transpose(np.array((xi_cross[i], error_cross[i])))
		res_dict[str(p)] = list(zip(xi_ct[i], error_ct[i]))
		i += 1
	print('Finished!')
	return pd.DataFrame(data=res_dict)

# takes the output of create_xi_table with return_samples=True
def get_bootstrap_xis(sample_dict, N):
	result = {}
	Nsamples = len(list(sample_dict.values())[0][0])
	for key in sample_dict.keys():
		result[key] = []
	for i in range(0, N):
		bootstrap_samples = np.random.randint(Nsamples, size=Nsamples)
		for key in sample_dict.keys():
			xis = sample_dict[key][0]
			weights = sample_dict[key][1]
			bs_xis = []
			bs_weights = []
			for j in bootstrap_samples:
				bs_xis.append(xis[j])
				bs_weights.append(weights[j])
			bs_weights = bs_weights / np.sum(bs_weights)
			result[key].append(np.sum(bs_xis * bs_weights))
	return result


def get_niche_correlations(adata, radii, treatment='base', bs=[], return_samples=False):
	fov_area = 5472.*3648 # these are the dimensions of the cell label mask images
	area_dict = {}
	area_dict['fov_area'] = fov_area
	# base
	area_dict['h02A'] = fov_area * 23.
	area_dict['h12A'] = fov_area * 14.
	area_dict['h30A'] = fov_area * 4.
	area_dict['h43A'] = fov_area * 10.
	area_dict['h47A'] = fov_area * 10.
	area_dict['h16A'] = fov_area * 4.
	# PD1
	area_dict['h02B'] = fov_area * 20.
	area_dict['h12B'] = fov_area * 23.
	area_dict['h15B'] = fov_area * 4.
	area_dict['h16B'] = fov_area * 4.
	area_dict['h17B'] = fov_area * 3.
	area_dict['h43B'] = fov_area * 16.
	area_dict['h47B'] = fov_area * 3.
	# RTPD1
	area_dict['h02C'] = fov_area * 7.
	area_dict['h12C'] = fov_area * 21.
	area_dict['h16C'] = fov_area * 4.
	area_dict['h43C'] = fov_area * 5.
	area_dict['h47C'] = fov_area * 7.
	df = adata.obs
	subniches = np.unique(adata.obs['subniche'].to_numpy())
	niches = np.unique(adata.obs['niche'].to_numpy())
	biopsies = np.unique(adata.obs['biopsy'].to_numpy())
	celltypes = np.unique(adata.obs['celltype'].to_numpy())
	biopsies_base = []
	biopsies_PD1 = []
	biopsies_RTPD1 = []
	treatments = ['base', 'PD1', 'RTPD1']
	for b in biopsies:
		if(b[3] == 'A'):
			biopsies_base.append(b)
		if(b[3] == 'B'):
			biopsies_PD1.append(b)
		if(b[3] == 'C'):
			biopsies_RTPD1.append(b)
	if(treatment=='base'):
		biopsies = biopsies_base
	elif(treatment=='PD1'):
		biopsies = biopsies_PD1
	elif(treatment=='RTPD1'):
		biopsies = biopsies_RTPD1
	else:
		print('Unknown treatment provided')
		print('Allowed treatments are ' + str(treatments))
		print('Aborting...')
		return -1
	if(bs != []):
		biopsies = bs
	df_tr = df.loc[df['biopsy'].isin(biopsies)]
	cross_pairs = []
	for n in niches:
		for s in subniches:
			cross_pairs.append((n, s))
	niche_pairs = []
	for n in niches:
		for m in niches:
			niche_pairs.append((n, m))
	subniche_pairs = []
	for s in subniches:
		for t in subniches:
			subniche_pairs.append((s, t))
	ct_pairs = []
	for c in celltypes:
		for t in celltypes:
			ct_pairs.append((c,t))
	return create_xi_table(df_tr, niche_pairs, subniche_pairs, cross_pairs, ct_pairs, radii, area_dict, return_samples=return_samples)


def get_n_fovs(data, biopsy):
	db = data.loc[data['biopsy'] == biopsy]
	return len(np.unique(db['fov_number'].to_numpy()))

# return list of biopsies which satisfy the criteria that 
def filter_biopsy_threshold(data, edge, min_threshold):
	# add in fov labels
	fov_labels = data['Unnamed: 0'].to_numpy()
	reduced_labels = []
	for l in fov_labels:
		reduced_labels.append(int(np.char.split(l, sep="_").tolist()[0]))
	reduced_labels = np.array(reduced_labels)
	data['fov_number'] = reduced_labels
	biopsies = np.unique(data['biopsy'].to_numpy())
	counts1 = {}
	counts2 = {}
	max1 = 0
	max2 = 0
	for b in biopsies:
		nfov = get_n_fovs(data, b)
		counts1[b] = len(data.loc[(data['biopsy'] == b) & (data['node_name'] == edge[0])]) / nfov
		counts2[b] = len(data.loc[(data['biopsy'] == b) & (data['node_name'] == edge[1])]) / nfov
		if(counts1[b] > max1):
			max1 = counts1[b]
		if(counts2[b] > max2):
			max2 = counts2[b]
	results = []
	for b in biopsies:
		if((counts1[b] >= min_threshold * max1) & (counts2[b] >= min_threshold * max2)):
			results.append(b)
	return results

# apply the results of filter_biopsy_threshold to a graph to compute the average correlation only in biopsies
# above the threshold
# right now just do this at 1 r
# assume that the individual correlations per biopsy are already in the graph
# this will just add a new column to the graph with the adjusted correlations
def apply_biopsy_threshold(graph, data, min_threshold, r):
	from_nodes = graph["from_node"].to_numpy()
	to_nodes = graph["to_node"].to_numpy()
	pairs = np.column_stack((from_nodes, to_nodes))
	results = []
	bcounts = []
	for p in pairs:
		biopsies = filter_biopsy_threshold(data, p, min_threshold)
		total_counts = len(data.loc[(data['node_name'] == p[0]) & (data['biopsy'].isin(biopsies))])
		res = 0
		if(total_counts != 0):
			for b in biopsies:
				bcount = len(data.loc[(data['node_name'] == p[0]) & (data['biopsy'] == b)])
				column_key = 'xi_' + str(b) + '_' + str(r)
				xi = float(graph.loc[(graph['from_node'] == p[0]) & (graph['to_node'] == p[1])][column_key])
				res += 1.0 * bcount/total_counts * xi
		#print(p)
		#print('Biopsies above threshold: ' + str(biopsies))
		#print('Corrected \\xi: ' + str(res))
		results.append(res)
		bcounts.append(len(biopsies))
	#graph['corrected_xi_' + str(min_threshold) + '_' + str(r)] = results
	graph['xi_' + str(r)] = results
	graph['corrected_biopsies_' + str(min_threshold)] = bcounts

# labels should be the same length as results and errors
# columns will be named xi_<label> and sigma_<label>
# results and errors taken as the output of compute_correlations_jk
def update_graph_table(graph, results, errors, labels):
	resultsT = np.transpose(results)
	errorsT = np.transpose(errors)
	adjusted_resultsT = resultsT-errorsT
	i = 0
	while(i < len(resultsT)):
		graph['xi_' + str(labels[i])] = resultsT[i]
		graph['sigma_' + str(labels[i])] = errorsT[i]
		graph['xi-sigma_' + str(labels[i])] = adjusted_resultsT[i]
		i += 1

	maxT = np.max(adjusted_resultsT,axis=0)
	graph['max_adj_xi'] = maxT


def add_count_columns(graph, pairs, df, min_fov_count):
	biopsies = np.unique(df['biopsy'].to_numpy())
	nfovs = []
	nb = []
	for p in pairs:
		counts1 = count_nodes(df, p[0])
		counts2 = count_nodes(df, p[1])
		fovcount = 0
		bs = []
		for b in biopsies:
			dist1 = counts1[b][1]
			dist2 = counts2[b][1]
			i = 0
			while(i < len(dist1)):
				if((dist1[i] >= min_fov_count) & (dist2[i] >= min_fov_count)):
					bs.append(b)
					fovcount += 1
				i += 1
		nb.append(len(np.unique(bs)))
		nfovs.append(fovcount)
	graph['nfovs'] = nfovs
	graph['nbiopsies'] = nb


# computes correlations for every pair in graph and then updates the table
# must provide either a valid data flag or an area dictionary which corresponds to the biopsies and fovs in 
# the input graph
# if ind_biop, also compute correlations for each individual biopsy
# modes are... 
# full (just use the full sample with jk errors)
# biopsy (compute individual biopsy correlations, only using certain biopsies with a count threshold given by biopsy_count_threshold)
# fov (compute all correlations for every fov, then only use fovs with a certain min count (min_fov_count))
def add_all_correlations(df, graph, radii, area_dict={}, data='pembroRT', mode='full', min_fov_count = 20, biopsy_count_threshold = 0.1):
	allowed_modes = ['full', 'biopsy', 'fov']
	if(mode not in allowed_modes):
		print('Unknown mode given...')
		print('Allowed modes are ' + str(allowed_modes))
		return -1
	# if(data=='pembroRT'):
	# 	fov_area = 5472.*3648 # these are the dimensions of the cell label mask images
	# 	area_dict = {}
	# 	area_dict['fov_area'] = fov_area
	# 	# base
	# 	area_dict['h02A'] = fov_area * 23.
	# 	area_dict['h12A'] = fov_area * 14.
	# 	area_dict['h30A'] = fov_area * 4.
	# 	area_dict['h43A'] = fov_area * 10.
	# 	area_dict['h16A'] = fov_area * 4.
	# 	# PD1
	# 	area_dict['h02B'] = fov_area * 20.
	# 	area_dict['h12B'] = fov_area * 23.
	# 	area_dict['h16B'] = fov_area * 4.
	# 	area_dict['h43B'] = fov_area * 16.
	# 	# RTPD1
	# 	area_dict['h02C'] = fov_area * 7.
	# 	area_dict['h12C'] = fov_area * 21.
	# 	area_dict['h16C'] = fov_area * 4.
	# 	area_dict['h43C'] = fov_area * 5.

	from_nodes = graph["from_node"].to_numpy()
	to_nodes = graph["to_node"].to_numpy()
	pairs = np.column_stack((from_nodes, to_nodes))
	if(mode=='full'):
		xis, errors = compute_correlations_jk(df, pairs, radii, area_dict)
		update_graph_table(graph, xis, errors, radii)
	if(mode=='biopsy'):
		biopsies = np.unique(df['biopsy'].to_numpy())
		print(biopsies)
		for b in biopsies:
			df1c = df.loc[df['biopsy'] == b]
			df1 = df1c.copy()
			print(len(df1))
			xis, errors = compute_correlations_jk(df1, pairs, radii, area_dict)
			labels = []
			for r in radii:
				labels.append(b + '_' + str(r))
			update_graph_table(graph, xis, errors, labels)
		# now compute the mean
		for r in radii:
			apply_biopsy_threshold(graph, df, biopsy_count_threshold, r)
	if(mode=='fov'):
		print('Starting \\xi calculation...')
		xi_dict = compute_corrs_fov(df, pairs, radii)
		print('Computing errors...')
		xis, errors = compute_mean_fov_corr(df, pairs, xi_dict, min_counts = min_fov_count)
		print('Updating table...')
		update_graph_table(graph, xis, errors, radii)
		add_count_columns(graph, pairs, df, min_fov_count)
		print('Finished!')
	return 0


# given an input graph filename and list of radii, saved a new graph with added xi and error columns to outfile
# must provide either a valid data flag or an area dictionary which corresponds to the biopsies and fovs in 
# the input graph
def add_xi_columns(ingraph, indata, outfile, radii, area_dict={}, data='pembroRT', edge_type = None, ind_biop=False):
	if(data=='pembroRT'):
		fov_area = 5472.*3648
		area_dict = {}
		area_dict['fov_area'] = fov_area
		# base
		area_dict['h02A'] = fov_area * 23.
		area_dict['h12A'] = fov_area * 14.
		area_dict['h30A'] = fov_area * 4.
		area_dict['h43A'] = fov_area * 10.
		area_dict['h16A'] = fov_area * 4.
		# PD1
		area_dict['h02B'] = fov_area * 20.
		area_dict['h12B'] = fov_area * 23.
		area_dict['h16B'] = fov_area * 4.
		area_dict['h43B'] = fov_area * 16.
		# RTPD1
		area_dict['h02C'] = fov_area * 7.
		area_dict['h12C'] = fov_area * 21.
		area_dict['h16C'] = fov_area * 4.
		area_dict['h43C'] = fov_area * 5.
	df = pd.read_csv(indata)
	graph = pd.read_csv(ingraph)
	if(edge_type != None):
		graph = graph.loc[graph['edge_type']==edge_type]
	add_all_correlations(df, graph, radii, area_dict=area_dict, data=data, ind_biop=ind_biop)
	graph.to_csv(outfile)

# concatenate graphs to include only shared pressure edges
# input graphs should have xi and sigma columns for each r in radii
# len(graphs) should equal len(labels)
# if min_count > 0, check the counts of each node in data to make sure we have at least min_count
# data is a list of tables
# adding the cut on minimum counts makes this take a bit longer
def concat_graphs(graphs, labels=['base', 'PD1', 'RTPD1'], radii=[100, 150, 200], min_count = -1, data = None, corrected_radii=[]):
	graphs_pressure = []
	for g in graphs:
		gpress = g.loc[g['edge_type'] == 'pressure']
		#from_nodes = g["from_node"].to_numpy()
		#to_nodes = g["to_node"].to_numpy()
		#pairs = np.column_stack((from_nodes, to_nodes))
		#gpress['pair'] = pairs
		graphs_pressure.append(gpress)
	keys = ['from_node', 'to_node']#, 'pair']
	for r in radii:
		keys.append('xi_' + str(r))
		keys.append('sigma_' + str(r))
	dicts = []
	for g in graphs_pressure:
		d = {}
		arr = g[keys].to_numpy()
		for a in arr:
			d[(a[0], a[1])] = a
		dicts.append(d)
	results = []
	for key in dicts[0]:
		c = True
		res = []
		for d in dicts:
			if(key not in d):
				c = False
		if(min_count > 0):
			for d in data:
				counts1 = len(d.loc[d['node_name'] == key[0]])
				counts2 = len(d.loc[d['node_name'] == key[1]])
				if((counts1 < min_count) | (counts2 < min_count)):
					c = False
		if(c):
			res = []
			for d in dicts:
				res.append(d[key])
			results.append(res)
	from_nodes = []
	to_nodes = []
	# xis[i][j] gives xi for the ith label at the jth radius
	xis = np.zeros((len(labels), len(radii), len(results)))
	sigmas = np.zeros((len(labels), len(radii), len(results)))
	# res dict is the dictionary that will give us our final dataframe
	res_dict = {}
	k = 0
	for res in results:
		from_nodes.append(res[0][0])
		to_nodes.append(res[0][1])
		i = 0
		while(i<len(labels)):
			j = 0
			while(j<len(radii)):
				xis[i][j][k] = res[i][2 + j*2]
				sigmas[i][j][k] = res[i][3 + j*2]
				j += 1
			i += 1
		k += 1
	res_dict['from_node'] = from_nodes
	res_dict['to_node'] = to_nodes
	i = 0
	while(i<len(labels)):
		j = 0
		while(j<len(radii)):
			res_dict['xi_' + str(labels[i]) + '_' + str(radii[j])] = xis[i][j]
			res_dict['sigma_' + str(labels[i]) + '_' + str(radii[j])] = sigmas[i][j]
			j += 1
		i += 1
	return pd.DataFrame(data=res_dict)



# removes any edges in the graph where each node has at least n counts in the total data
# remove n, now this just adds new columns to the graph with the counts
def filter_by_number(graph, data, n):
	from_nodes = graph["from_node"].to_numpy()
	to_nodes = graph["to_node"].to_numpy()
	pairs = np.column_stack((from_nodes, to_nodes))
	counts1 = []
	counts2 = []
	for p in pairs:
		counts1.append(len(data.loc[data['node_name'] == p[0]]))
		counts2.append(len(data.loc[data['node_name'] == p[1]]))
	graph['from_node_count'] = counts1
	graph['to_node_count'] = counts2


##############################################################
# Helpful plotting functions
##############################################################

# plots the \xi distribution of the jackknifed samples for each pair in given by the indeces pis
# pairs is the full list of pairs in jk_results just for indexing and labels
# ri is the index of the radius you want to plot
# jk_results is the output of compute_correlations_jk with return_samples=True
def plot_xi_distribution(jk_results, pis, pairs, ri=0, legend=True, output=None, bins='auto'):
	plt.clf()
	for pi in pis:
		dist = []
		for res in jk_results:
			dist.append(res[pi][ri])
		freq, bins = np.histogram(dist, bins=bins)
		plt.stairs(freq, bins, label=pairs[pi])
	plt.xlabel(r'$\xi$')
	plt.ylabel('number of jackknife samples')
	if(legend):
		plt.legend()
	if(output != None):
		plt.savefig(output, format='pdf')

# make a scatter plot with abs(base-PD1) on one axis and abs(PD1-RTPD1) on the other
def plot_xi_difference(base, PD1, RTPD1, r=100, min_count = -1, data = None, labels=False, min_label = -1, xoffset=0, yoffset=0, output=None, title=''):
	plt.clf()
	graph = concat_graphs([base, PD1, RTPD1], min_count=min_count, data=data)
	print('Number of edges satisfying conditions - ' + str(len(graph)))
	base_PD1 = graph.apply(lambda x: x['xi_base_' + str(r)] - x['xi_PD1_' + str(r)], axis=1).to_numpy()
	PD1_RTPD1 = graph.apply(lambda x: x['xi_PD1_' + str(r)] - x['xi_RTPD1_' + str(r)], axis=1).to_numpy()
	plt.scatter(base_PD1, PD1_RTPD1, s=0.5, color='black')
	plt.xlabel(r'$\xi_\mathrm{base} - \xi_\mathrm{PD1}$')
	plt.ylabel(r'$\xi_\mathrm{PD1} - \xi_\mathrm{RTPD1}$')
	plt.title(title + r', $r=$' + str(r))
	if(labels):
		from_nodes = graph["from_node"].to_numpy()
		to_nodes = graph["to_node"].to_numpy()
		pairs = np.column_stack((from_nodes, to_nodes))
		for i, p in enumerate(pairs):
			if((abs(base_PD1[i]) > min_label) | (abs(PD1_RTPD1[i]) > min_label)):
				plt.annotate(str(p), (base_PD1[i] + xoffset, PD1_RTPD1[i] + yoffset), size=7)
	if(output != None):
		plt.savefig(output, format='pdf')


# look at how the correlation of edges changes after treatments, imposing some conditions
# min_threshold is a fraction of the counts in the largest biopsy for each individual node
def plot_xi_variation(graphs, labels=['base', 'PD1', 'RTPD1'], r=100, min_count = -1, data = None, output=None, corrected=False, corrected_r=100):
	plt.clf()
	# make sure there is positive correlation in at least one data set
	if(corrected):
		graph = concat_graphs(graphs, min_count=min_count, data=data, corrected_radii=[corrected_r])
		graph_pos = graph.loc[(graph['xi_' + labels[0] + '_' + str(r)]>0) | (graph['xi_' + labels[1] + '_' + str(r)]>0) | (graph['xi_' + labels[2] + '_' + str(r)]>0)]
	else:
		graph = concat_graphs(graphs, min_count=min_count, data=data)
		graph_pos = graph.loc[(graph['xi_' + labels[0] + '_' + str(r)]>0) | (graph['xi_' + labels[1] + '_' + str(r)]>0) | (graph['xi_' + labels[2] + '_' + str(r)]>0)]
	print('Number of edges satisfying conditions - ' + str(len(graph_pos)))
	fig, ax = plt.subplots()
	xs = [1,2,3]
	xis1 = graph_pos['xi_base_' + str(r)].to_numpy()
	xis2 = graph_pos['xi_PD1_' + str(r)].to_numpy()
	xis3 = graph_pos['xi_RTPD1_' + str(r)].to_numpy()
	from_nodes = graph_pos["from_node"].to_numpy()
	to_nodes = graph_pos["to_node"].to_numpy()
	pairs = np.column_stack((from_nodes, to_nodes))
	color = iter(cm.rainbow(np.linspace(0, 1, len(xis1))))
	i = 0
	ls='dashed'
	while(i < len(xis1)):
		c = next(color)
		if(ls=='dashed'):
			ls='solid'
		else:
			ls='dashed'
		ax.plot(xs, [xis1[i],xis2[i],xis3[i]], lw=1.5, label=pairs[i], c=c, ls=ls)
		i += 1
	ax.set_xlim(1,3)
	ax.set_xticks([1,2,3])
	ax.set_ylabel(r'$\xi$(r=' + str(r) + ')')
	ax.set_xticklabels(['base', 'PD1', 'RTPD1'])
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
	if(output != None):
		plt.savefig(output, format='pdf')

##############################################################
# Additional functions for analyzing data
##############################################################

# compute the distribution of node in the fovs of each tissue given
# result is a dictionary mapping biopsy names to a list of counts for each fov in that biopsy
def count_nodes(df, node, tissues=[], type_name = 'node_name', tissue_name = 'biopsy', obs_name = 'Unnamed: 0'):
	if(len(tissues) == 0):
		tissues = df[tissue_name].to_numpy()
		tissues = np.unique(tissues)
	fov_labels = df[obs_name].to_numpy()
	reduced_labels = []
	for l in fov_labels:
		reduced_labels.append(int(np.char.split(l, sep="_").tolist()[0]))
	reduced_labels = np.array(reduced_labels)
	df['fov_number'] = reduced_labels
	# count up the number of unique fovs in each tissue
	label_list = []
	for t in tissues:
		cells = df.loc[df[tissue_name] == t]
		labels = cells['fov_number'].to_numpy().astype(int)
		labels = np.unique(labels)
		label_list.append(labels)
	result = {}
	i = 0
	while(i < len(tissues)):
		t = tissues[i]
		counts = []
		for l in label_list[i]:
			cells = df.loc[(df[tissue_name] == t) & (df[type_name] == node) & (df['fov_number'] == l)]
			counts.append(len(cells))
		result[t] = (np.sum(counts), counts)
		i += 1
	return result

##############################################################
# Density field filtering functions
##############################################################

# returns a 2d numpy array image where each pixel=1 if there is a cell of the given type
# df should be preprocessed to only include 1 fov and the cells of interest
# fov dimensions for the pembroRT dataset are 5472, 3648
def df_to_image(df, xdim=5472, ydim=3648, row_name = 'x', col_name = 'y'):
	rows = df[row_name].to_numpy().astype(int)
	cols = df[col_name].to_numpy().astype(int)
	rows = df[row_name].to_numpy().astype(int)
	cols = df[col_name].to_numpy().astype(int)
	# shift the image coords to have a min value of 0
	if(np.min(rows) < 0):
		rows += np.abs(np.min(rows))
	if(np.min(rows) > 0):
		rows -= np.min(rows)
	if(np.min(cols) < 0):
		cols += np.abs(np.min(cols))
	if(np.min(cols) > 0):
		cols -= np.min(cols)
	ncells = len(rows)
	sample = np.zeros([xdim, ydim])
	i = 0
	while(i < ncells):
		sample[rows[i]][cols[i]] += 1.0
		i += 1
	return sample

# returns the number of units from the center of a square of length s of pos
def distance_from_center(s, pos):
    dx = abs(pos[0] - s/2.)
    dy = abs(pos[1] - s/2.)
    return np.sqrt(dx**2.0 + dy**2.0)

def gauss_2D(r, sigma):
	return 1.0 / (2 * np.pi * sigma * sigma) * np.exp((-1.0 * r * r) / (2 * sigma * sigma))

def power_2D(r, exp=2):
	# max not including the center is 1.0, so we just set the center to 2.0
	# this might not be right
	if(r==0.0):
		return 2.0
	return 1.0 / (r ** exp)

# return a circular kernel of radius r
# uniform weights
# modes right now are uniform and gaussian
# if mode==gaussian, r is the sigma and we'll go out to a radius of 3sigma
def circular_kernel(r, mode='gaussian', dim=None, offset=0):
    if(mode=='uniform'):
        R = int(np.ceil(r))
        res = np.ones((2*R, 2*R))
        res[R][R] = 0
        res = ndimage.distance_transform_edt(res)
        res = (res < r).astype(int)
        return res
    if(mode=='gaussian'):
        sigma = r
        if(dim==None):
            dim = 2 * int(np.ceil(3*sigma))
        res = np.ones((dim, dim))
        if(offset != 0):
            for angle in np.arange(0, 2 * np.pi, 1/2/offset):
                x = int(dim/2 + offset * np.cos(angle))
                y = int(dim/2 + offset * np.sin(angle))
                res[x][y] = 0
        else:
            res[dim//2][dim//2] = 0
        res = ndimage.distance_transform_edt(res)
        res = 1.0 / (2 * np.pi * sigma * sigma) * np.exp((-1.0 * res * res) / (2 * sigma * sigma))
        return res
    if(mode=='powerlaw'):
        exp = r
        if(dim==None):
            dim = 2 * int(np.ceil(3*sigma))
        res = np.ones((dim, dim))
        res[dim//2][dim//2] = 0
        res = ndimage.distance_transform_edt(res)
        res = 1.0 / (res**exp)
        res[dim//2][dim//2] = 2
        return res
    if(mode=='exp'):
        if(dim==None):
            dim = 2 * int(np.ceil(3*sigma))
        res = np.ones((dim, dim))
        if(offset != 0):
            for angle in np.arange(0, 2 * np.pi, 1/2/offset):
                x = int(dim/2 + offset * np.cos(angle))
                y = int(dim/2 + offset * np.sin(angle))
                res[x][y] = 0
        else:
            res[dim//2][dim//2] = 0
        res = ndimage.distance_transform_edt(res)
        res = np.exp(-1.0 * res / r)
        return res

# perform a 2D convolution of the sample with the given kernel
# kernel is a 2d numpy array
# input df should be preprocessed to only include cells of interest in one fov
# output is the convolved image (total counts in the kernel)
# how do we deal with missing pixels? for now sample should just have zeros in missing pixels
# r is size of kernel (radius for a circle)
def convolve_sample(im, r, xdim=-1, ydim=-1, kernel_type='circle', row_name = 'x', col_name = 'y', mode='same', kernel=[]):
	#sample = df_to_image(df, xdim=xdim, ydim=ydim, row_name=row_name, col_name=col_name)
	# generate the kernel
	if(kernel == []):
		if(kernel_type=='circle'):
			kernel = circular_kernel(r)
		else:
			print("Provided kernel does not match allowed options.")
			print("Defaulting to a circular kernel with uniform weights.")
			kernel = circular_kernel(r)
	#return ndimage.convolve(im, kernel, mode="reflect")
	return signal.fftconvolve(im, kernel, mode=mode)

def plot_overlap(df, biopsy, fov, node1, node2, r, output=None, labels=[1, 2]):
	dffov1 = df.loc[(df['biopsy'] == biopsy) & (df['fov_number'] == fov) & (df['node_name'] == node1)]
	dffov2 = df.loc[(df['biopsy'] == biopsy) & (df['fov_number'] == fov) & (df['node_name'] == node2)]
	im1 = df_to_image(dffov1, 5472, 3648)
	im2 = df_to_image(dffov2, 5472, 3648)
	conv1 = convolve_sample(im1, r)
	conv2 = convolve_sample(im2, r)
	#f = plt.figure()
	f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
	ax1.imshow(conv1)
	ax2.imshow(conv2)
	ax3.imshow(conv1 * conv2)
	ax1.set_title(str(labels[0]))
	ax2.set_title(str(labels[1]))
	ax3.set_title('Overlap')
	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	if(output != None):
		plt.tight_layout()
		plt.savefig(output, format='pdf')

# returns the std of number count distributions of the image of the given cluster of cells convolved at radius r
# might as well return the convolved image since we're computing it
def sigma(df, r, xdim=5472, ydim=3648, kernel_type='circle', row_name = 'x', col_name = 'y'):
	im = df_to_image(df)
	conv_image = convolve_sample(im, r, kernel_type=kernel_type, row_name=row_name, col_name=col_name)
	return (np.std(np.ravel(conv_image)), conv_image)

def sigma_r(df, rmin, rmax, xdim=5472, ydim=3648, Nbins=20, kernel_type='circle', row_name = 'x', col_name = 'y'):
	rs = np.logspace(np.log10(rmin), np.log10(rmax), Nbins)
	#rs = np.linspace(rmin, rmax, Nbins)
	sigmas = np.zeros_like(rs)
	images = []
	i = 0
	while(i < len(rs)):
		res = sigma(df, rs[i], kernel_type=kernel_type, row_name=row_name, col_name=col_name)
		sigmas[i] = res[0]
		images.append(res[1])
		i += 1
	return ((rs, sigmas), images)

def plot_image_sigma(image, rs, sigmas, ri):
	f, (ax1, ax2) = plt.subplots(1, 2)
	ax1.imshow(image)
	ax1.axis('off')
	var = sigmas * sigmas
	ax2.plot(rs, var, color='red', lw=2.0)
	ax2.set_xlim(rs[0], rs[len(rs)-1])
	ax2.set_ylim(var[len(var)-1], var[0])
	xmin,xmax = ax2.get_xlim()
	ymin,ymax = ax2.get_ylim()
	ax2.set_yscale('log')
	ax2.plot([xmin, rs[ri]], [var[ri], var[ri]], color='black', lw=0.5)
	ax2.plot([rs[ri], rs[ri]], [var[ri], ymin], color='black', lw=0.5)
	ax2.set_xlabel(r'r / [coordinate units]')
	ax2.set_ylabel(r'$\sigma^2$')
	plt.tight_layout()


def main(ARGS):
	biopsy_sizes = pd.read_csv(ARGS.biopsy_area_table,
								index_col=0, header=0)
	fov_area = 5472.*3648
	area_dict = biopsy_sizes['n_fovs'].to_dict()
	area_dict = {k: v*fov_area for k,v in area_dict.items()}

	df = pd.read_csv(ARGS.coords_table, index_col=0, header=0)
	graph = pd.read_csv(ARGS.graph_table, index_col=0, header=0)

	# TODO This column is referenced by name all over the place. It's just the index column.
	df['Unnamed: 0'] = df.index.values
	graph['Unnamed: 0'] = graph.index.values

	# from_nodes = graph["from_node"].to_numpy()
	# to_nodes = graph["to_node"].to_numpy()
	# pairs = np.column_stack((from_nodes, to_nodes))

	# xis, errors, jk_results = compute_correlations_jk(df, pairs, ARGS.radii, area_dict, return_samples=True)
	# update_graph_table(graph, xis, errors, ARGS.radii)

	# updates graph in-place
	add_all_correlations(df, graph, ARGS.radii, area_dict, 
		mode='fov',
		min_fov_count=ARGS.min_fov_count,
		biopsy_count_threshold=ARGS.biopsy_count_threshold
	)

	graph.drop('Unnamed: 0', axis=1, inplace=True)

	outf = f'{ARGS.outdir}/graph_edges_xcorr.csv'
	graph.to_csv(outf)
	print('(computeCorrelations) :', graph.shape, outf)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--graph_table')
	parser.add_argument('--coords_table')
	parser.add_argument('--biopsy_area_table')
	parser.add_argument('--outdir')

	parser.add_argument('--radii', nargs='+', default=[100, 150, 200])
	parser.add_argument('--min_fov_count', type=int)
	parser.add_argument('--biopsy_count_threshold', type=float)

	ARGS = parser.parse_args()

	for k,v in ARGS.__dict__.items():
		print('(computeCorrelations) :', k, v)

	main(ARGS)
