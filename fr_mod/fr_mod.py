import os as _os
import cv2 as _cv2
import numpy as _np
from keras.models import load_model as _load_model

# Make sure to use CPU
_os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
_os.environ["CUDA_VISIBLE_DEVICES"] = ""

class _common_model:
    '''Common base class definition for federal registry image classification models'''
    
    def __init__(self):
                
        self.input_shape = (256,256,3)
        
        #self._home_dir = '/notebook/CJ/fed_detect/modmod/'
        self.sample_dir = './samples/' # self._home_dir+
    
    def predict_from_samples(self):

        # Get list of image files
        test_files = _os.listdir(self.sample_dir)

        # Load images from 'samples' directory into numpy matrix
        x_test = []
        for i,f in enumerate(test_files):
            img = _cv2.imread(_os.path.join(self.sample_dir, f))
            x_test.append(_cv2.resize(img, self.input_shape[:-1]))

        # Normalize values
        x_test  = _np.array(x_test, _np.float32) / 255.

        # Get prediction(s)
        one_hot_pred = self._mod.predict(x_test, verbose=1)

        return(test_files,one_hot_pred)
        
class semantic_model(_common_model):
    '''Class definition for semantic classification model'''
    
    def __init__(self):
        
        _common_model.__init__(self)
        
        # Define sample directory
        self.sample_dir = _os.path.join(self.sample_dir, 'category')
        
        # Load compiled model
        self._mod = _load_model('cat_mod.h5')
        #self._mod = _load_model(_os.path.join(self._home_dir, 'cat_mod.h5'))
        
        # Load category (i.e. 'class') names
        self._class_names = _np.load('cat_class_names.npy')
        self.class_name_dict = dict([[v,k] for k,v in self._class_names])
        
    def get_pred_from_samples(self):
        
        (test_files, one_hot_pred) = self.predict_from_samples()
        prediction = _np.argmax(one_hot_pred, axis=1)
        pred_names = [self.class_name_dict[str(x)] for x in prediction]
        
        return(zip(test_files,pred_names))
        
class rotation_model(_common_model):
    '''Class definition for rotation classification model'''
    
    def __init__(self):
        
        _common_model.__init__(self)
        
        # Define sample directory
        self.sample_dir = _os.path.join(self.sample_dir, 'rotation')
                
        # Load compiled model
        self._mod = _load_model('rot_mod.h5')
    
        # Load category (i.e. 'class') names
        self._class_names = _np.load('rot_class_names.npy')
        self.class_name_dict = dict([[v,k] for k,v in self._class_names])
        
    def get_pred_from_samples(self):
        
        (test_files, one_hot_pred) = self.predict_from_samples()
        prediction = _np.round(one_hot_pred)
        pred_names = [self.class_name_dict[str(int(x))] for x in prediction]
        
        return(zip(test_files,pred_names))