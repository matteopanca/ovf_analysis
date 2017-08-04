import numpy as np
import matplotlib.pyplot as plt
import h5py

type_tuple = (np.float32, np.float64)
type_min = 4
param_tuple = ('valuedim','nodes','stepsize','min','max')

#-------------------- CLASS Definition --------------------

class OVF_File:

	#CONSTRUCTOR
	
	def __init__(self, fname, quantity):
		self.fname = fname
		self.quantity = quantity #'m' for magnetization; 'h' for field
		collect_param = np.zeros(1, dtype=[(param_tuple[0],np.int_),(param_tuple[1],np.int_,3),(param_tuple[2],np.float_,3),(param_tuple[3],np.float_,3),(param_tuple[4],np.float_,3)])
		self.nodes = np.zeros(3, dtype=np.int_)
		self.stepsize = np.zeros(3, dtype=np.float_)
		self.mincoord = np.zeros(3, dtype=np.float_)
		self.maxcoord = np.zeros(3, dtype=np.float_)
		
		f = open(fname, 'rb') #open the file
		line = f.readline().decode() #OVF version is in first line
		if '1' in line:
			self.ovf_version = 1
		elif '2' in line:
			self.ovf_version = 2
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
				else:
					for i in range(3):
						if i!=0:
							line = f.readline().decode()
						splitted = line.split(' ')
						data = float(splitted[2].strip('\n'))
						collect_param[param][0, 2-i] = data #index order is Z, Y, X
		
		self.valuedim = collect_param['valuedim'][0]
		self.nodes = collect_param['nodes'][0, :] #index order is Z, Y, X
		self.stepsize = collect_param['stepsize'][0, :] #index order is Z, Y, X
		self.mincoord = collect_param['min'][0, :] #index order is Z, Y, X
		self.maxcoord = collect_param['max'][0, :] #index order is Z, Y, X
		
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
		self.z_axis = 1e9*(self.mincoord[0] + self.stepsize[0]*np.arange(0, self.nodes[0], 1, dtype=type_selected)) #in nm
		self.y_axis = 1e9*(self.mincoord[1] + self.stepsize[1]*np.arange(0, self.nodes[1], 1, dtype=type_selected)) #in nm
		self.x_axis = 1e9*(self.mincoord[2] + self.stepsize[2]*np.arange(0, self.nodes[2], 1, dtype=type_selected)) #in nm
		
		self.z_values = np.zeros(self.nodes, dtype=type_selected) #index order is Z, Y, X
		self.y_values = np.zeros(self.nodes, dtype=type_selected) #index order is Z, Y, X
		self.x_values = np.zeros(self.nodes, dtype=type_selected) #index order is Z, Y, X
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
	
	def plotXY(self, comp, z_slice):
		if comp == 'x':
			values_to_plot = self.x_values[z_slice, :, :]
		elif comp == 'y':
			values_to_plot = self.y_values[z_slice, :, :]
		elif comp == 'z':
			values_to_plot = self.z_values[z_slice, :, :]
		elif comp == 'm':
			values_to_plot = self.mod()[z_slice, :, :]
		
		mySize = 18
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(1,1,1)
		cmesh = ax.pcolormesh(self.x_axis, self.y_axis, values_to_plot)
		ax.axis('image')
		plt.colorbar(cmesh)
		ax.tick_params(axis='both', labelsize=mySize)
		ax.set_xlabel('x (nm)', fontsize=mySize)
		ax.set_ylabel('y (nm)', fontsize=mySize)
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
	f.create_dataset('y_values', data=obj.y_values)
	f.create_dataset('z_values', data=obj.z_values)
	f.create_dataset('fname', data=obj.fname)
	f.create_dataset('quantity', data=obj.quantity)
	f.create_dataset('binary_value', data=obj.binary_value)
	f.create_dataset('valuedim', data=obj.valuedim)
	f.create_dataset('nodes', data=obj.nodes)
	f.create_dataset('stepsize', data=obj.stepsize)
	f.create_dataset('mincoord', data=obj.mincoord)
	f.create_dataset('maxcoord', data=obj.maxcoord)
	f.close()