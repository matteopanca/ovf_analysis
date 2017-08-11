import numpy as np
import matplotlib.pyplot as plt
import h5py
import warnings

mu0 = np.pi*4e-7
OOMMFtoOe = np.pi*4e-3
MUMAXtoOe = 1e-4
xv = np.array([1, 0, 0])
yv = np.array([0, 1, 0])
zv = np.array([0, 0, 1])

#-------------------- CLASS Definition --------------------

class OVF_File:

	#----- CONSTRUCTOR -----
	
	def __init__(self, fname, quantity='', mult_coeff=0):
		if type(fname) == str:
			return self.from_ovf(fname, quantity, mult_coeff)
		else:
			return self.from_h5(fname)
	
	def from_ovf(self, fname, quantity, mult_coeff=0):
		type_min = 4
		inType_tuple = (np.float32, np.float64)
		param_tuple = ('valuedim', 'nodes', 'stepsize', 'min', 'max', 'Total simulation')
		
		self.fname = fname
		self.quantity = quantity #'m' for magnetization; 'h' for field
		collect_param = np.zeros(1, dtype=[(param_tuple[0],np.int_),(param_tuple[1],np.int_,3),(param_tuple[2],np.float_,3),(param_tuple[3],np.float_,3),(param_tuple[4],np.float_,3),(param_tuple[5],np.float_)])
		self.nodes = np.zeros(3, dtype=np.int_)
		self.stepsize = np.zeros(3, dtype=np.float_)
		self.mincoord = np.zeros(3, dtype=np.float_)
		self.maxcoord = np.zeros(3, dtype=np.float_)
		self.index = -1 #not always used, but sometimes useful
		self.ok = False
		
		f = open(fname, 'rb') #open the file
		line = f.readline().decode() #OVF version is in first line
		if '1' in line:
			self.ovf_version = 1
			type_tuple = ('>f4', '>f8')
			collect_param['valuedim'][0] = 3 #only vector data for OVF V1 
		elif '2' in line:
			self.ovf_version = 2
			type_tuple = ('<f4', '<f8')
		else:
			f.close() #close the file
			#raise RuntimeError('Not valid OVF version')
			return None
		
		while not('# Begin: Data' in line): #parsing the lines
			line = f.readline().decode()
			found = False
			for str in param_tuple:
				if str in line:
					param = str
					found = True
			
			if found:
				if param == 'valuedim':
					splitted = line.split(' ')
					data = float(splitted[2].strip('\n'))
					collect_param[param][0] = data
				elif param == 'Total simulation':
					splitted = line.split(' ')
					splitted = [el for el in splitted if len(el)!=0]
					data = float(splitted[5])
					collect_param[param][0] = data
				else:
					for i in range(3):
						if i!=0:
							line = f.readline().decode()
						splitted = line.split(' ')
						data = float(splitted[2].strip('\n'))
						collect_param[param][0, 2-i] = data #index order is Z, Y, X
		
		self.valuedim = collect_param['valuedim'][0]
		self.t = collect_param['Total simulation'][0] #in s
		self.nodes = collect_param['nodes'][0, :] #index order is Z, Y, X
		self.stepsize = 1e9*collect_param['stepsize'][0, :] #index order is Z, Y, X - in nm
		self.mincoord = 1e9*collect_param['min'][0, :] #index order is Z, Y, X - in nm
		self.maxcoord = 1e9*collect_param['max'][0, :] #index order is Z, Y, X - in nm
		
		splitted = line.split(' ')
		self.binary_value = int(splitted[4].strip('\n'))
		
		#we are in position for reading the binary data
		tot_data = self.valuedim*np.prod(self.nodes)
		type_index = int(self.binary_value/type_min) - 1
		data_stream = np.fromfile(f, dtype=type_tuple[type_index], count=1+tot_data)
		
		f.close() #close the file
		
		#check byte order
		if self.binary_value == 4:
			if data_stream[0] != 1234567.0:
				raise RuntimeError('Error in reading the file: file corrupted')
				#return None #useful if removing the "raise"
		elif self.binary_value == 8:
			if data_stream[0] != 123456789012345.0:
				raise RuntimeError('Error in reading the file: file corrupted')
				#return None #useful if removing the "raise"
		
		#split the data in the proper arrays
		if mult_coeff != 0:
			data_stream *= mult_coeff
		
		self.z_axis = self.mincoord[0] + self.stepsize[0]*np.arange(0, self.nodes[0], 1, dtype=inType_tuple[type_index])
		self.y_axis = self.mincoord[1] + self.stepsize[1]*np.arange(0, self.nodes[1], 1, dtype=inType_tuple[type_index])
		self.x_axis = self.mincoord[2] + self.stepsize[2]*np.arange(0, self.nodes[2], 1, dtype=inType_tuple[type_index])
		
		#self.z_values = np.zeros(self.nodes, dtype=inType_tuple[type_index]) #index order is Z, Y, X
		#self.y_values = np.zeros(self.nodes, dtype=inType_tuple[type_index]) #index order is Z, Y, X
		#self.x_values = np.zeros(self.nodes, dtype=inType_tuple[type_index]) #index order is Z, Y, X
		if self.valuedim == 3:
			self.x_values = np.reshape(data_stream[1::3], self.nodes, order='C')
			self.y_values = np.reshape(data_stream[2::3], self.nodes, order='C')
			self.z_values = np.reshape(data_stream[3::3], self.nodes, order='C')
			#counter = 1 #first data is the byte order check
			#for i in range(self.nodes[0]):
				#for j in range(self.nodes[1]):
					#for k in range(self.nodes[2]):
						#self.x_values[i, j, k] = data_stream[counter]
						#self.y_values[i, j, k] = data_stream[counter + 1]
						#self.z_values[i, j, k] = data_stream[counter + 2]
						#counter += 3
		elif self.valuedim == 1:
			self.x_values = np.reshape(data_stream[1:], self.nodes, order='C')
			#self.y_values = []
			#self.z_values = []
			#counter = 1 #first data is the byte order check
			#for i in range(self.nodes[0]):
				#for j in range(self.nodes[1]):
					#for k in range(self.nodes[2]):
						#self.x_values[i, j, k] = data_stream[counter] #scalar data will appear in 'x_values'
						#counter += 1
		else:
			raise RuntimeError('Wrong number of components')
			#return None #useful if removing the "raise"
		
		self.ok = True
	
	def from_h5(self, group):
		self.valuedim = group.get('valuedim').value
		self.x_axis = group.get('x_axis').value
		self.y_axis = group.get('y_axis').value
		self.z_axis = group.get('z_axis').value
		self.x_values = group.get('x_values').value
		if self.valuedim == 3:
			self.y_values = group.get('y_values').value
			self.z_values = group.get('z_values').value
		self.fname = group.get('fname').value
		self.quantity = group.get('quantity').value
		self.binary_value = group.get('binary_value').value
		self.t = group.get('t').value
		self.index = group.get('index').value
		self.nodes = group.get('nodes').value
		self.stepsize = group.get('stepsize').value
		self.mincoord = group.get('mincoord').value
		self.maxcoord = group.get('maxcoord').value
		self.ok = True
	
	#----- METHODS -----
	
	def set_index(self, i):
		self.index = i
	
	def mod(self):
		return np.sqrt(self.x_values**2 + self.y_values**2 + self.z_values**2)
	
	def scalar(self, obj):
		return (self.x_values*obj.x_values + self.y_values*obj.y_values + self.z_values*obj.z_values)
	
	def mask(self):
		mod_map = self.mod()
		return np.greater(mod_map, 10*np.finfo(mod_map.dtype).eps) #10 times the "eps" for the considered type
	
	def not_mask(self):
		mod_map = self.mod()
		return np.less(mod_map, 10*np.finfo(mod_map.dtype).eps) #10 times the "eps" for the considered type
	
	def max(self):
		return np.amax(self.mod()[self.mask()])
	
	def min(self):
		return np.amin(self.mod()[self.mask()])
	
	#"dir" have to be a np.array with 3 components; "limits" should be a list (or tuple) of 3 lists (or 3 tuples)
	def avg_comp(self, dir, limits=None):
		norm_dir = dir/np.sqrt(np.sum(dir**2))
		scalar_prod = norm_dir[0]*self.x_values + norm_dir[1]*self.y_values + norm_dir[2]*self.z_values
		useful_mask = self.not_mask()
		scalar_prod = np.ma.masked_array(scalar_prod, useful_mask) #masked array to be considered
		if limits != None:
			sub_range = []
			for i in range(3):
				if len(limits[2-i]) != 0:
					sub_range.append(np.logical_and(self.x_axis >= limits[2-i][0], self.x_axis <= limits[2-i][1]))
				else:
					sub_range.append(np.ones(self.nodes[i], dtype=bool))
			scalar_prod = scalar_prod[sub_range[0], :, :]
			scalar_prod = scalar_prod[:, sub_range[1], :]
			scalar_prod = scalar_prod[:, :, sub_range[2]]
		result = np.mean(scalar_prod)
		if result is np.ma.masked:
			warnings.warn('No valid data to be averaged')
			return 0.
		else:
			return result
	
	def plot(self, comp, slice, cblimits=None, axlimits=None):
		useful_mask = self.not_mask()
		if comp[0:2] == 'xy' or comp[0:2] == 'yx':
			axis_image = True
			x_to_plot = self.x_axis
			x_label = 'x'
			y_to_plot = self.y_axis
			y_label = 'y'
			useful_mask = useful_mask[slice, :, :]
			if comp[2] == 'x':
				values_to_plot = self.x_values[slice, :, :]
			elif comp[2] == 'y':
				values_to_plot = self.y_values[slice, :, :]
			elif comp[2] == 'z':
				values_to_plot = self.z_values[slice, :, :]
			elif comp[2] == 'm':
				values_to_plot = self.mod()[slice, :, :]
		elif comp[0:2] == 'xz' or comp[0:2] == 'zx':
			axis_image = False
			x_to_plot = self.x_axis
			x_label = 'x'
			y_to_plot = self.z_axis
			y_label = 'z'
			useful_mask = useful_mask[:, slice, :]
			if comp[2] == 'x':
				values_to_plot = self.x_values[:, slice, :]
			elif comp[2] == 'y':
				values_to_plot = self.y_values[:, slice, :]
			elif comp[2] == 'z':
				values_to_plot = self.z_values[:, slice, :]
			elif comp[2] == 'm':
				values_to_plot = self.mod()[:, slice, :]
		elif comp[0:2] == 'yz' or comp[0:2] == 'zy':
			axis_image = False
			x_to_plot = self.y_axis
			x_label = 'y'
			y_to_plot = self.z_axis
			y_label = 'z'
			useful_mask = useful_mask[:, :, slice]
			if comp[2] == 'x':
				values_to_plot = self.x_values[:, :, slice]
			elif comp[2] == 'y':
				values_to_plot = self.y_values[:, :, slice]
			elif comp[2] == 'z':
				values_to_plot = self.z_values[:, :, slice]
			elif comp[2] == 'm':
				values_to_plot = self.mod()[:, :, slice]
		
		values_to_plot = np.ma.masked_array(values_to_plot, useful_mask) #masked array to be plotted
		
		if axlimits != None:
			x_subRange = np.logical_and(x_to_plot >= axlimits[0], x_to_plot <= axlimits[1])
			y_subRange = np.logical_and(y_to_plot >= axlimits[2], y_to_plot <= axlimits[3])
			x_to_plot = x_to_plot[x_subRange]
			y_to_plot = y_to_plot[y_subRange]
			values_to_plot = values_to_plot[:, x_subRange]
			values_to_plot = values_to_plot[y_subRange, :]
		
		mySize = 18
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(1,1,1)
		if cblimits == None:
			cmesh = ax.pcolormesh(x_to_plot, y_to_plot, values_to_plot)
			plt.colorbar(cmesh)
		else:
			cmesh = ax.pcolormesh(x_to_plot, y_to_plot, values_to_plot, vmin=cblimits[0], vmax=cblimits[1])
			plt.colorbar(cmesh, extend='both')
		if axis_image:
			ax.axis('image')
		ax.tick_params(axis='both', labelsize=mySize)
		ax.set_title(comp[2]+'-Component - Slice {:d}'.format(slice), fontsize=mySize)
		ax.set_xlabel(x_label+' (nm)', fontsize=mySize)
		ax.set_ylabel(y_label+' (nm)', fontsize=mySize)
		ax.grid(True)
		plt.tight_layout()
		plt.show()

#-------------------- Functions --------------------

#Save multiple data to HDF5 file (not part of the OVF_File class)
#"obj" have to be an object compatible with the "len()" method
#"file_mode" have to be 'w' (default) or 'a'
def save_h5(obj, output_name, file_mode='w'):
	n_maps = len(obj)
	f = h5py.File(output_name+'.hdf5', file_mode)
	start_index = len(f.keys())
	
	for i in range(n_maps):
		basename = 'map{:06d}/'.format(start_index+i) #1000000 groups should be enough
		f.create_dataset(basename+'x_axis', data=obj[i].x_axis)
		f.create_dataset(basename+'y_axis', data=obj[i].y_axis)
		f.create_dataset(basename+'z_axis', data=obj[i].z_axis)
		f.create_dataset(basename+'x_values', data=obj[i].x_values)
		if obj[i].valuedim == 3:
			f.create_dataset(basename+'y_values', data=obj[i].y_values)
			f.create_dataset(basename+'z_values', data=obj[i].z_values)
		f.create_dataset(basename+'fname', data=obj[i].fname)
		f.create_dataset(basename+'quantity', data=obj[i].quantity)
		f.create_dataset(basename+'binary_value', data=obj[i].binary_value)
		f.create_dataset(basename+'valuedim', data=obj[i].valuedim)
		f.create_dataset(basename+'t', data=obj[i].t)
		f.create_dataset(basename+'index', data=obj[i].index)
		f.create_dataset(basename+'nodes', data=obj[i].nodes)
		f.create_dataset(basename+'stepsize', data=obj[i].stepsize)
		f.create_dataset(basename+'mincoord', data=obj[i].mincoord)
		f.create_dataset(basename+'maxcoord', data=obj[i].maxcoord)
		print('Map {:d} saved'.format(i))
	f.close()

#Open HDF5 file (not part of the OVF_File class)
#"read_range" have to be an object compatible with the "len()" method
def open_h5(input_name, read_range=[]):
	f = h5py.File(input_name+'.hdf5', 'r')
	n_maps = len(f.keys())
	if len(read_range) == 0:
		start_index = 0
		stop_index = n_maps
	else:
		start_index = read_range[0]
		stop_index = min(read_range[1]+1, n_maps)
	data = []
	for i in range(start_index, stop_index):
		data.append(OVF_File(f['map{:06d}/'.format(i)]))
	f.close()
	return data