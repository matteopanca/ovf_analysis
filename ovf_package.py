import numpy as np
import matplotlib.pyplot as plt
import h5py

mu0 = np.pi*4e-7
OOMMFtoOe = np.pi*4e-3
MUMAXtoOe = 1e-4

#-------------------- CLASS Definition --------------------

class OVF_File:

	#----- CONSTRUCTOR -----
	
	def __init__(self, fname, quantity, mult_coeff=0):
		type_min = 4
		param_tuple = ('valuedim', 'nodes', 'stepsize', 'min', 'max', 'Total simulation')
		
		self.fname = fname
		self.quantity = quantity #'m' for magnetization; 'h' for field
		collect_param = np.zeros(1, dtype=[(param_tuple[0],np.int_),(param_tuple[1],np.int_,3),(param_tuple[2],np.float_,3),(param_tuple[3],np.float_,3),(param_tuple[4],np.float_,3),(param_tuple[5],np.float_)])
		self.nodes = np.zeros(3, dtype=np.int_)
		self.stepsize = np.zeros(3, dtype=np.float_)
		self.mincoord = np.zeros(3, dtype=np.float_)
		self.maxcoord = np.zeros(3, dtype=np.float_)
		self.index = 0 #not always used, but sometimes useful
		
		f = open(fname, 'rb') #open the file
		line = f.readline().decode() #OVF version is in first line
		if '1' in line:
			self.ovf_version = 1
			type_tuple = ('>f4', '>f8')
			collect_param['valuedim'][0] = 3 #OVF V1 only for vector data
		elif '2' in line:
			self.ovf_version = 2
			type_tuple = ('<f4', '<f8')
		else:
			raise RuntimeError('Not valid OVF version')
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
		type_selected = type_tuple[int(self.binary_value/type_min) - 1]
		data_stream = np.fromfile(f, dtype=type_selected, count=1+tot_data)
		
		f.close() #close the file
		
		#check byte order
		if self.binary_value == 4:
			if data_stream[0] != 1234567.0:
				raise RuntimeError('Error in reading the file: file corrupted')
				return None
		elif self.binary_value == 8:
			if data_stream[0] != 123456789012345.0:
				raise RuntimeError('Error in reading the file: file corrupted')
				return None
		
		#split the data in the proper arrays
		if mult_coeff != 0:
			data_stream *= mult_coeff
		
		self.z_axis = self.mincoord[0] + self.stepsize[0]*np.arange(0, self.nodes[0], 1, dtype=np.float_)
		self.y_axis = self.mincoord[1] + self.stepsize[1]*np.arange(0, self.nodes[1], 1, dtype=np.float_)
		self.x_axis = self.mincoord[2] + self.stepsize[2]*np.arange(0, self.nodes[2], 1, dtype=np.float_)
		
		self.z_values = np.zeros(self.nodes, dtype=np.float_) #index order is Z, Y, X
		self.y_values = np.zeros(self.nodes, dtype=np.float_) #index order is Z, Y, X
		self.x_values = np.zeros(self.nodes, dtype=np.float_) #index order is Z, Y, X
		if self.valuedim == 3:
			counter = 1 #first data is the byte order check
			for i in range(self.nodes[0]):
				for j in range(self.nodes[1]):
					for k in range(self.nodes[2]):
						self.x_values[i, j, k] = data_stream[counter]
						self.y_values[i, j, k] = data_stream[counter + 1]
						self.z_values[i, j, k] = data_stream[counter + 2]
						counter += 3
		elif self.valuedim == 1:
			self.y_values = []
			self.z_values = []
			counter = 1 #first data is the byte order check
			for i in range(self.nodes[0]):
				for j in range(self.nodes[1]):
					for k in range(self.nodes[2]):
						self.x_values[i, j, k] = data_stream[counter] #scalar data will appear in 'x_values'
						counter += 1
		else:
			raise RuntimeError('Wrong number of components')
			return None
	
	#----- METHODS -----
	
	def mod(self):
		return np.sqrt(self.x_values**2 + self.y_values**2 + self.z_values**2)
	
	def scalar(self, obj):
		return (self.x_values*obj.x_values + self.y_values*obj.y_values + self.z_values*obj.z_values)
	
	def mask(self):
		mod_map = self.mod()
		return np.greater(mod_map, np.amax(mod_map)/10.) #is 10 an acceptable value for filtering?
	
	def plot(self, comp, slice, limits=None):
		if comp[0:2] == 'xy' or comp[0:2] == 'yx':
			axis_image = True
			x_to_plot = self.x_axis
			x_label = 'x'
			y_to_plot = self.y_axis
			y_label = 'y'
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
			if comp[2] == 'x':
				values_to_plot = self.x_values[:, :, slice]
			elif comp[2] == 'y':
				values_to_plot = self.y_values[:, :, slice]
			elif comp[2] == 'z':
				values_to_plot = self.z_values[:, :, slice]
			elif comp[2] == 'm':
				values_to_plot = self.mod()[:, :, slice]
				
		mySize = 18
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(1,1,1)
		if limits == None:
			cmesh = ax.pcolormesh(x_to_plot, y_to_plot, values_to_plot)
			plt.colorbar(cmesh)
		else:
			cmesh = ax.pcolormesh(x_to_plot, y_to_plot, values_to_plot, vmin=limits[0], vmax=limits[1])
			plt.colorbar(cmesh, extend='both')
		if axis_image:
			ax.axis('image')
		ax.tick_params(axis='both', labelsize=mySize)
		ax.set_xlabel(x_label+' (nm)', fontsize=mySize)
		ax.set_ylabel(y_label+' (nm)', fontsize=mySize)
		ax.grid(True)
		plt.tight_layout()
		plt.show()

#-------------------- Functions --------------------

#Save data to HDF5 file (not part of the OVF_File class)
def save(obj, output_name):
	f = h5py.File(output_name, 'a')
	f.create_dataset('x_axis', data=obj.x_axis)
	f.create_dataset('y_axis', data=obj.y_axis)
	f.create_dataset('z_axis', data=obj.z_axis)
	f.create_dataset('x_values', data=obj.x_values)
	if obj.valuedim == 3:
		f.create_dataset('y_values', data=obj.y_values)
		f.create_dataset('z_values', data=obj.z_values)
	f.create_dataset('fname', data=obj.fname)
	f.create_dataset('quantity', data=obj.quantity)
	f.create_dataset('binary_value', data=obj.binary_value)
	f.create_dataset('valuedim', data=obj.valuedim)
	f.create_dataset('t', data=obj.t)
	f.create_dataset('index', data=obj.index)
	f.create_dataset('nodes', data=obj.nodes)
	f.create_dataset('stepsize', data=obj.stepsize)
	f.create_dataset('mincoord', data=obj.mincoord)
	f.create_dataset('maxcoord', data=obj.maxcoord)
	f.close()