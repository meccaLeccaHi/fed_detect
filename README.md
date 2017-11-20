# fed_detect

Trains and re-packages 2 image classification models, for the purposes of detecting the semantic category and rotation of figures embedded in the digital archives of the [Federal Register]. Each model consists of a pre-trained convolutional neural network with drop-out, and a partially un-frozen convolutional base, and perform with very high (>98%) accuracy. This tool is intended to improve the overall readability and discoverability of decades of government documents.

* **obj_det_fedreg.ipynb** trains and saves multi-class prediction model for detecting semantic category of an image.
* **rot_det_fedreg.ipynb** trains and saves binary prediction model for detecting rotation of an image.

* **/frmod/frmod.py** loads compiled models and makes predictions using image input.  
Usage:
```python
    import os 
    os.chdir('/notebook/CJ/fed_detect/fr_mod/')  
    import fr_mod  
    foo = modmod.semantic_model()  
    a = foo.get_pred_from_samples()  
    print(a)
    bar = modmod.rotation_model()
    b = bar.get_pred_from_samples()
    print(b)
```

**Semantic model class examples**  
![Semantic model class examples](https://i.imgur.com/ejc6wtY.png?1)  
**Rotation model class examples**  
![Rotation model class examples](https://i.imgur.com/2IFv3XT.png?1)  

**Semantic model prediction examples**  
![Semantic model predictions](https://i.imgur.com/eHeju9Z.png)  
**Rotation model prediction examples**  
![Rotation model predictions](https://i.imgur.com/ziYhnki.png)  

**Semantic model confusion matrix**
![Semantic model confusion matrix](https://i.imgur.com/HH8nZPG.png)

[Federal Register]: https://www.federalregister.gov/
