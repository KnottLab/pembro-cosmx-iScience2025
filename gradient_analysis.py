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
from correlations import *
from PIL import Image
import warnings
import matplotlib
import os

import argparse

# gradient_analysis.py
# functions for computing gene expression gradients with vector field analysis
# Richard Mebane 2023

##############################################################
# Data functions
##############################################################

# im_prefix is path to image directory
def get_label_image(biopsy, fov, df, im_prefix):
	cells = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov)]
	slide = cells['slide_id'].to_numpy()[0]
	if(fov >= 10):
		fov_label = '0' + str(fov)
	else:
		fov_label = '00' + str(fov)
	imagef = im_prefix + slide + '/CellLabels/CellLabels_F' + fov_label + '.tif'
	print('Opening image at ' + imagef)
	im = Image.open(imagef)
	return np.array(im)


def get_niche_image(label_image, biopsy, fov, df):
	res = np.full((len(label_image), len(label_image[0])), -1)
	cell_ids = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov)]
	cell_ids = cell_ids[['cell_ID', 'niche']].to_numpy()
	for i in cell_ids:
		im_inds = np.where(np.ravel(label_image) == i[0])
		np.put(np.ravel(res), im_inds, int(i[1][5]))
	return res

def get_cluster_image(label_image, biopsy, fov, df, cluster_label='Niche1_CC'):
	res = np.full((len(label_image), len(label_image[0])), -1)
	cell_ids = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov)]
	cell_ids = cell_ids[['cell_ID', cluster_label]].to_numpy()
	for i in cell_ids:
		if(i[1] != 'na'):
			im_inds = np.where(np.ravel(label_image) == i[0])
			cluster_index = int(np.char.split(i[1], sep="_").tolist()[1])
			np.put(np.ravel(res), im_inds, cluster_index)
	return res

def get_binary_cluster_image(label_image, biopsy, fov, df, cluster_label='Niche1_CC'):
	res = np.full((len(label_image), len(label_image[0])), 0.0)
	cell_ids = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov)]
	print(len(cell_ids))
	cell_ids = cell_ids[['cell_ID', cluster_label]].to_numpy()
	for i in cell_ids:
		if(i[1] != 'na'):
			im_inds = np.where(np.ravel(label_image) == i[0])
			cluster_index = int(np.char.split(i[1], sep="_").tolist()[1])
			np.put(np.ravel(res), im_inds, 1.0)
	return res

##############################################################
# Analysis functions
##############################################################

# take a (x,y) vector and normalize it to unit magnitude
def make_unit_vector(x):
	den = np.linalg.norm(x)
	if((den == 0) | np.isnan(den)):
		return (0,0)
	return x / np.linalg.norm(x)

# xs and ys are field of the x and y components of vectors in a field
def make_vector_field_unit(xs, ys):
	mags = np.sqrt(xs**2 + ys**2)
	xs /= mags
	ys /= mags
	return (xs, ys)

# get unit vectors from (x,y), weighted by some function of distance
# weights is of length txs/tys and is a linear multiplier on the distance vector
def get_weighted_unit_vectors(x, y, txs, tys, ex=2, weights=1):
	dxs = txs - x
	dys = tys - y
	magnitudes = np.sqrt(dxs**2 + dys**2)
	scaled_dxs = dxs / magnitudes**(1+ex) # extra 1 to make unit
	scaled_dys = dys / magnitudes**(1+ex)
	scaled_dxs *= weights
	scaled_dys *= weights
	return np.column_stack((scaled_dxs, scaled_dys))

def get_nearest_unit_vector(x, y, txs, tys):
	dxs = txs - x
	dys = tys - y
	magnitudes = np.sqrt(dxs**2 + dys**2)
	mini = np.argmin(magnitudes)
	return(dxs[mini] / magnitudes[mini], dys[mini] / magnitudes[mini])

# x is a list of vectors
# sum them up, then normalize
def get_summed_unit_vector(x):
	res = np.sum(x, axis=0)
	return make_unit_vector(res)

def get_unit_vector(x, y, txs, tys, ex=2, weights=1):
	vectors = get_weighted_unit_vectors(x, y, txs, tys, ex=ex, weights=weights)
	return get_summed_unit_vector(vectors)

# returns a tuple of xs, ys of target cells used to compute the direction vectors
def get_target_cell_locs(adata, biopsy, fov, celltypes = [], niches = [], subniches = [], nocluster=False):
	df = adata.obs
	if(niches == []):
		niches = np.unique(adata.obs['niche'].to_numpy())
	if(subniches == []):
		subniches = np.unique(adata.obs['subniche'].to_numpy())
	if(celltypes == []):
		celltypes = np.unique(adata.obs['celltype'].to_numpy())
	cells_fov = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov)]
	target_cells = cells_fov.loc[(cells_fov['celltype'].isin(celltypes)) & (cells_fov['niche'].isin(niches)) & (cells_fov['subniche'].isin(subniches))]
	if(nocluster):
		target_cells = cells_fov.loc[cells_fov['Niche1_CC'] == 'na']
	target_xs = target_cells['CenterX_local_px'].to_numpy()
	target_ys = target_cells['CenterY_local_px'].to_numpy()
	return(target_xs, target_ys)

# computes the convolved density field of a given gene in an fov and returns the gradient of that field
# input adata should be normalized first!
def get_gene_expression_gradient(adata, gene, biopsy, fov, sigma=100, xlen=3648, ylen=5472, gene_celltypes=[], clusters=[], kernel=[]):
	if(gene_celltypes == []):
		gene_celltypes = np.unique(adata.obs['celltype'].to_numpy())
	if(clusters == []):
		clusters = np.unique(adata.obs['Niche1_CC'].to_numpy())
	exp_image = np.zeros((xlen, ylen))
	gene_subset = adata[adata[: , gene].X > 0.0, :]
	expression_df = gene_subset.to_df()
	gene_cells = gene_subset.obs.loc[(gene_subset.obs['biopsy'] == biopsy) & (gene_subset.obs['fov'] == fov) & (gene_subset.obs['celltype'].isin(gene_celltypes)) & (gene_subset.obs['Niche1_CC'].isin(clusters))]
	gene_xs = gene_cells['CenterX_local_px'].to_numpy()
	gene_ys = gene_cells['CenterY_local_px'].to_numpy()
	cell_names = gene_cells.index.to_numpy()
	i = 0
	while(i < len(cell_names)):
		exp_image[gene_ys[i]][gene_xs[i]] = expression_df.loc[cell_names[i]][gene]
		i += 1
	#print('Number of ' + str(gene_celltypes) + ' cells expressing gene ' + gene + ': ' + str(i))
	if(kernel == []):
		conv_im = convolve_sample(exp_image, sigma)
	else:
		conv_im = convolve_sample(exp_image, sigma, kernel=kernel)
	return (conv_im, np.gradient(conv_im))

def get_cell_density_gradient(adata, biopsy, fov, celltypes = [], niches = [], subniches = [], cluster=None, sigma=100, xlen=3648, ylen=5472, conv=True, kernel=[]):
	cell_im = np.zeros((xlen, ylen))
	df = adata.obs
	if(niches == []):
		niches = np.unique(adata.obs['niche'].to_numpy())
	if(subniches == []):
		subniches = np.unique(adata.obs['subniche'].to_numpy())
	if(celltypes == []):
		celltypes = np.unique(adata.obs['celltype'].to_numpy())
	if(cluster != None):
		cells = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov) & (df['celltype'].isin(celltypes)) & (df['niche'].isin(niches)) & (df['subniche'].isin(subniches)) & (df[cluster] != 'na')]
	else:
		cells = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov) & (df['celltype'].isin(celltypes)) & (df['niche'].isin(niches)) & (df['subniche'].isin(subniches))]
	xs = cells['CenterX_local_px'].to_numpy()
	ys = cells['CenterY_local_px'].to_numpy()
	locs = np.array(list(zip(xs, ys)))
	for p in locs:
		cell_im[p[1]][p[0]] = 1.0
	if(conv):
		if(kernel==[]):
			conv_im = convolve_sample(cell_im, sigma)
		else:
			conv_im = convolve_sample(cell_im, sigma, kernel=kernel)
	else:
		conv_im = cell_im
	#print(np.min(conv_im))
	#print(np.max(conv_im))
	return (conv_im, np.gradient(conv_im))

# this will return the gradient of the given cluster
# remember to multiply by -1 for the negative gradient
# im is the output of get_binary_cluster_image()
def get_cell_density_cluster_gradient(adata, biopsy, fov, im, sigma=100, kernel=[]):
	if(kernel==[]):
		conv_im = convolve_sample(im, sigma)
	else:
		conv_im = convolve_sample(im, sigma, kernel=kernel)
	return (conv_im, np.gradient(conv_im))

# TO DO: something is wrong here... 
# Can you divide two convolved density fields like this?
def get_expression_per_cell_gradient(adata, biopsy, fov, gene, celltypes=['Tcell'], sigma=100, xlen=3648, ylen=5472, kernel=[]):
	gene_celltypes = ['Tcell']
	exp_image = np.zeros((xlen, ylen))
	gene_subset = adata[adata[: , gene].X > 0.0, :]
	expression_df = gene_subset.to_df()
	gene_cells = gene_subset.obs.loc[(gene_subset.obs['biopsy'] == biopsy) & (gene_subset.obs['fov'] == fov) & (gene_subset.obs['celltype'].isin(gene_celltypes))]
	gene_xs = gene_cells['CenterX_local_px'].to_numpy()
	gene_ys = gene_cells['CenterY_local_px'].to_numpy()
	cell_names = gene_cells.index.to_numpy()
	i = 0
	while(i < len(cell_names)):
		exp_image[gene_ys[i]][gene_xs[i]] = expression_df.loc[cell_names[i]][gene]
		i += 1
	cell_im = np.zeros((xlen, ylen))
	df = adata.obs
	cells = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov) & (df['celltype'].isin(celltypes))]
	xs = cells['CenterX_local_px'].to_numpy()
	ys = cells['CenterY_local_px'].to_numpy()
	locs = np.array(list(zip(xs, ys)))
	for p in locs:
		cell_im[p[1]][p[0]] = 1.0
	if(kernel==[]):
		conv_im_cells = convolve_sample(cell_im, sigma)
		conv_im_exp = convolve_sample(exp_image, sigma)
	else:
		conv_im_cells = convolve_sample(cell_im, sigma, kernel=kernel)
		conv_im_exp = convolve_sample(exp_image, sigma, kernel=kernel)
	conv_im_exp /= conv_im_cells
	output = np.nan_to_num(conv_im_exp, posinf=0.0, neginf=0.0, nan=0.0)
	#output = (output-np.min(output))/(np.max(output)-np.min(output))
	#print(np.min(output))
	#print(np.max(output))
	return (output, np.gradient(output))


# computes the dot product of the density gradient of gene1 of a certain celltype on the fov
# adata expressions should be normalized
def gradient_dot(adata, celltypes, gene, biopsy, fov, sigma=100, xlen=3648, ylen=5472, gene_celltypes = ['Tcell'], direction_output=None, direction_input=None):
	output_image = np.zeros((xlen, ylen))
	if(direction_output != None):
		direction_xs = np.zeros((xlen, ylen))
		direction_ys = np.zeros((xlen, ylen))
	if(direction_input != None):
		directions = np.load(direction_input)
	target_xs, target_ys = get_target_cell_locs(adata, biopsy, fov, celltypes=celltypes)
	grad = get_gene_expression_gradient(adata, gene, biopsy, fov, sigma=sigma, gene_celltypes=gene_celltypes)
	x = 0
	while(x < ylen):
		print(x)
		y = 0
		while(y < xlen):
			if(direction_input == None):
				d1 = get_unit_vector(x, y, target_xs, target_ys)
			else:
				d1 = (directions[0][y][x], directions[1][y][x])
			d2 = (grad[1][y][x], grad[0][y][x])
			d2 = make_unit_vector(d2)
			output_image[y][x] = d1[0] * d2[0] + d1[1] * d2[1]
			if(direction_output != None):
				direction_xs[y][x] = d1[0]
				direction_ys[y][x] = d1[1]
			y += 1
		x += 1
	if(direction_output != None):
		output = np.array([direction_xs, direction_ys])
		np.save(direction_output, output)
	return output_image

def get_vector_field(adata, biopsy, fov, celltypes=[], niches=[], subniches=[], xlen=3648, ylen=5472, kernel=[]):
	if(kernel==[]):
		kernel = circular_kernel(0, mode='powerlaw', dim=6000)
	if(niches == []):
		niches = np.unique(adata.obs['niche'].to_numpy())
	if(subniches == []):
		subniches = np.unique(adata.obs['subniche'].to_numpy())
	if(celltypes == []):
		celltypes = np.unique(adata.obs['celltype'].to_numpy())
	res = get_cell_density_gradient(adata, biopsy, fov, celltypes=celltypes, niches=niches, subniches=subniches, kernel=kernel, suffixes = ['A'])
	return res[1]

def output_vector_fields(adata, celltypes=[], niches=[], subniches=[], biopsies=['h02', 'h12', 'h16', 'h43'], xlen=3648, ylen=5472, kernel=[]):
	if(kernel==[]):
		kernel = circular_kernel(0, mode='powerlaw', dim=6000)
	if(niches == []):
		niches = np.unique(adata.obs['niche'].to_numpy())
	if(subniches == []):
		subniches = np.unique(adata.obs['subniche'].to_numpy())
	if(celltypes == []):
		celltypes = np.unique(adata.obs['celltype'].to_numpy())
	outdir='/Users/mebaner/bio/gradient_analysis/vector_fields/' + str(niches[0]) + '/'
	warnings.filterwarnings('ignore', category=RuntimeWarning) # turn off runtime warnings since we will get a div by zero when computing a vector at it's own loaction. this is fine
	for b in biopsies:
		for s in suffixes:
			biopsy = b+s
			df_sub = adata.obs.loc[adata.obs['biopsy'] == biopsy]
			fovs = np.unique(df_sub['fov'].to_numpy())
			for fov in fovs:
				print(biopsy + ' ' + str(fov))
				resx, resy = get_vector_field(adata, biopsy, int(fov), celltypes=celltypes, niches=niches, subniches=subniches, kernel=kernel)
				resx, resy = make_vector_field_unit(resx, resy)
				output = np.array([resx, resy])
				outf = str(biopsy) + '_' + str(fov)
				np.save(outdir + outf, output)
	warnings.resetwarnings()

def output_cluster_vector_field(adata, im_prefix, cluster_label='Niche1_CC', xlen=3648, ylen=5472):
	biopsies = ['h02','h12','h16','h43']
	suffixes = ['A','B','C']
	df = adata.obs
	outdir='/common/mebaner/data/vector_fields/'
	for b in biopsies:
		for s in suffixes:
			biopsy = b+s
			df_sub = adata.obs.loc[adata.obs['biopsy'] == biopsy]
			fovs = np.unique(df_sub['fov'].to_numpy())
			for fov in fovs:
				print(biopsy + ' ' + str(fov))
				#target_xs, target_ys = get_target_cell_locs(adata, biopsy, fov, nocluster=True)
				label_image = get_label_image(biopsy, fov, df, im_prefix)
				cluster_image = np.flipud(get_binary_cluster_image(label_image, biopsy, fov, df, cluster_label=cluster_label))
				conv_im = convolve_sample(cluster_image, 100)
				grad = np.gradient(conv_im)
				# vectors = []
				# points = []
				# x = 0
				# while(x < ylen):
				# 	y = 0
				# 	while(y < xlen):
				# 		if(cluster_image[y][x] != -1):
				# 			d = get_unit_vector(x, y, target_xs, target_ys, ex=10, weights=1)
				# 			vectors.append([d[0], d[1]])
				# 			points.append([x,y])
				# 		y += 1
				# 	x += 1
				# output = np.array([points, vectors])
				#output = get_cell_density_cluster_gradient(adata, biopsy, fov, cluster_image, sigma=100, kernel=[])
				#output = output[1]
				outf = str(biopsy) + '_' + str(fov) + '_' + str(100) + '_' + cluster_label
				np.save(outdir + outf, grad)



'''
def output_vector_fields(adata, celltypes=[], niches=[], subniches=[], biopsies=['h02', 'h12', 'h16', 'h43'], xlen=3648, ylen=5472):
	suffixes = ['A','B','C']
	outdir='/Users/mebaner/bio/gradient_analysis/vector_fields/niche1/'
	warnings.filterwarnings('ignore', category=RuntimeWarning) # turn off runtime warnings since we will get a div by zero when computing a vector at it's own loaction. this is fine
	for b in biopsies:
		for s in suffixes:
			biopsy = b+s
			df_sub = adata.obs.loc[adata.obs['biopsy'] == biopsy]
			fovs = np.unique(df_sub['fov'].to_numpy())
			for fov in fovs:
				out_xs = np.zeros((xlen, ylen))
				out_ys = np.zeros((xlen, ylen))
				print('Computing ' + str(niches) + ' vector field for ' + str(biopsy) + ' ' + str(fov))
				xs, ys = get_target_cell_locs(adata, biopsy, fov, celltypes=celltypes, niches=niches, subniches=subniches)
				outf = str(biopsy) + '_' + str(fov)
				x = 0
				while(x < ylen):
					y = 0
					while(y < xlen):
						d1=get_unit_vector(x, y, xs, ys)
						out_xs[y][x] = d1[0]
						out_ys[y][x] = d1[1]
						y += 1
					x += 1
				output = np.array([out_xs, out_ys])
				np.save(outdir + outf, output)
	warnings.resetwarnings()'''

def output_proximity_maps(adata, r, outdir, im_prefix, celltypes=[], niches = [], subniches = [], biopsies=['h02', 'h12', 'h16', 'h43'], xlen=3648, ylen=5472,suffixes = ['A']):
	df = adata.obs
	if(niches == []):
		niches = np.unique(adata.obs['niche'].to_numpy())
	if(subniches == []):
		subniches = np.unique(adata.obs['subniche'].to_numpy())
	if(celltypes == []):
		celltypes = np.unique(adata.obs['celltype'].to_numpy())
	#outdir='/Users/mebaner/bio/gradient_analysis/proximity_maps/'
	for b in biopsies:
		for s in suffixes:
			biopsy = b+s
			df_sub = adata.obs.loc[adata.obs['biopsy'] == biopsy]
			fovs = np.unique(df_sub['fov'].to_numpy())
			for fov in fovs:
				print('Computing ' + str(celltypes) + ' proximity map for ' + str(biopsy) + ' ' + str(fov) + ' at r = ' + str(r))
				label_image = get_label_image(biopsy, fov, df, im_prefix)
				cells_fov = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov)]
				target_cells = cells_fov.loc[(cells_fov['celltype'].isin(celltypes)) & (cells_fov['subniche'].isin(subniches)) & (cells_fov['niche'].isin(niches))]
				target_xs = target_cells['CenterX_local_px'].to_numpy()
				target_ys = target_cells['CenterY_local_px'].to_numpy()
				target_names = target_cells.index.to_numpy()
				target_labels = []
				for n in target_names:
					sep_label = np.char.split(n, sep="_").tolist()
					sublabel = np.char.split(sep_label[1], sep="-").tolist()
					cell_label = int(sublabel[0])
					target_labels.append(cell_label)
				cell_mask = np.zeros((xlen, ylen))
				for i in target_labels:
					im_inds = np.where(np.ravel(label_image) == i)
					np.put(np.ravel(cell_mask), im_inds, 1)
				# remove areas without cells
				im_inds = np.where(np.ravel(label_image) == 0)
				np.put(np.ravel(cell_mask), im_inds, 1)
				cell_mask = np.flipud(cell_mask)
				prox_map = np.zeros((xlen, ylen))
				x = 0
				while(x < ylen):
					y = 0
					while(y < xlen):
						dxs = target_xs - x
						dys = target_ys - y
						distances = np.sqrt(dxs**2 + dys**2)
						if((np.count_nonzero(distances <= r) > 0) & (cell_mask[y][x] != 1)):
							prox_map[y][x] = 1.0
						y += 1
					x += 1
				outf = niches[0] + '_' + str(biopsy) + '_' + str(fov) + '_r' + str(r)
				np.save(outdir + outf, prox_map)

# returns the cell label mask image
def get_cell_image(adata, biopsy, fov, im_prefix, celltypes=['epithelial_tumor'], niches=[], subniches=[], xlen=3648, ylen=5472):
	df = adata.obs
	if(niches == []):
		niches = np.unique(adata.obs['niche'].to_numpy())
	if(subniches == []):
		subniches = np.unique(adata.obs['subniche'].to_numpy())
	if(celltypes == []):
		celltypes = np.unique(adata.obs['celltype'].to_numpy())
	label_image = get_label_image(biopsy, fov, df, im_prefix)
	cells_fov = df.loc[(df['biopsy'] == biopsy) & (df['fov'] == fov)]
	target_cells = cells_fov.loc[(cells_fov['celltype'].isin(celltypes))]
	target_xs = target_cells['CenterX_local_px'].to_numpy()
	target_ys = target_cells['CenterY_local_px'].to_numpy()
	target_names = target_cells.index.to_numpy()
	target_labels = []
	for n in target_names:
		sep_label = np.char.split(n, sep="_").tolist()
		sublabel = np.char.split(sep_label[1], sep="-").tolist()
		cell_label = int(sublabel[0])
		target_labels.append(cell_label)

	niche_mask = np.zeros((xlen, ylen))
	for i in target_labels:
		im_inds = np.where(np.ravel(label_image) == i)
		np.put(np.ravel(niche_mask), im_inds, 1)
    
	return np.flipud(niche_mask)



