
'''
Usage:
    import os
    os.chdir('/notebook/CJ/fed_detect/fr_mod/')
    import fr_mod

    foo = modmod.semantic_model()
    a = foo.get_pred_from_samples()
    print(a)

    bar = modmod.rotation_model()
    b = bar.get_pred_from_samples()
    print(b)
'''

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
        
        self.sample_dir = './samples/' # self._home_dir+
    
    def predict_from_samples(self):

        # Get list of image files
        test_files = _os.listdir(self.sample_dir)

        test_preds = []
        
        # Load each image in 'samples' directory and classify 
        for i,f in enumerate(test_files):
            
            with open(_os.path.join(self.sample_dir, f), 'rb') as infile:
                buf = infile.read()

            # Use numpy to construct an array from the bytes
            x = _np.fromstring(buf, dtype='uint8')

            raw_img = _cv2.imdecode(x, _cv2.CV_LOAD_IMAGE_COLOR)

            # Decode the array into an image
            img = _cv2.resize(raw_img, self.input_shape[:-1])
            #img = _cv2.imread(_os.path.join(self.sample_dir, f))
            
            # Normalize values
            img = _np.array(img, _np.float32) / 255.
            
            # Get prediction(s)
            one_hot_pred = self._mod.predict(_np.expand_dims(img, axis=0))[0] # verbose=1
            
            test_preds.append(one_hot_pred)

        return(test_files, _np.stack(test_preds, axis=0))
        
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

         # Take max of each prediction, since it's a multi-class categorization
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
        
        # Round output, since it's a binary categorization
        prediction = _np.round(one_hot_pred)
        pred_names = [self.class_name_dict[str(int(x))] for x in prediction]
        
        return(zip(test_files,pred_names))