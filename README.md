# Phishpedia A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages

- This is the official implementation of "Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages" USENIX'21 [link to paper](https://www.usenix.org/system/files/sec21fall-lin.pdf), [link to our website](https://sites.google.com/view/phishpedia-site/home?authuser=0)
- The contributions of our paper:
   - [x] We propose a phishing identification system Phishpedia, which has high identification accuracy and low runtime overhead, outperforming the relevant state-of-the-art identification approaches. 
   - [x] Our system provides explainable annotations which increases users' confidence in model prediction
   - [x] We conduct phishing discovery experiment on emerging domains fed from CertStream and discovered 1,704 real phishing, out of which 1133 are zero-days   

## Framework
    
<img src="phishpedia/big_pic/overview.png" style="width:2000px;height:350px"/>

```Input```: A URL and its screenshot ```Output```: Phish/Benign, Phishing target
- Step 1: Enter <b>Deep Object Detection Model</b>, get predicted logos and inputs (inputs are not used for later prediction, just for explaination)

- Step 2: Enter <b>Deep Siamese Model</b>
    - If Siamese report no target, ```Return  Benign, None```
    - Else Siamese report a target, ```Return Phish, Phishing target``` 
    
## Requirements
The following packages may need to install manually.
- Windows/Linux/Mac machine 
- python=3.7 
- torch=1.6.0 # Make sure that the Pytorch is compatible with your CUDA version.
- torchvision
- Install compatible Detectron2 manually, see the [official installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). If you are using Windows, try this [guide](https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c) instead.


## Use it as a package
First install the requirements, then run
```
 pip install git+https://github.com/lindsey98/Phishpedia.git
```
In python
```python
from phishpedia.phishpedia_main import test
import matplotlib.pyplot as plt
from phishpedia.phishpedia_config import load_config

url = open("phishpedia/datasets/test_sites/accounts.g.cdcde.com/info.txt").read().strip()
screenshot_path = "phishpedia/datasets/test_sites/accounts.g.cdcde.com/shot.png"
cfg_path = None # None means use default config.yaml
ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(cfg_path)

phish_category, pred_target, plotvis, siamese_conf, pred_boxes = test(url, screenshot_path,
                                                                      ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)

print('Phishing (1) or Benign (0) ?', phish_category)
print('What is its targeted brand if it is a phishing ?', pred_target)
print('What is the siamese matching confidence ?', siamese_conf)
print('Where is the predicted logo (in [x_min, y_min, x_max, y_max])?', pred_boxes)
plt.imshow(plotvis[:, :, ::-1])
plt.title("Predicted screenshot with annotations")
plt.show()
```

## Use it as a repository
First install the requirements
Then, run
```
pip install -r requirements.txt
```
Please see detailed instructions in [phishpedia/README.md](phishpedia/README.md)

## Reference 
If you find our work useful in your research, please consider citing our paper by:
```
@inproceedings{lin2021phishpedia,
  title={Phishpedia: A Hybrid Deep Learning Based Approach to Visually Identify Phishing Webpages},
  author={Lin, Yun and Liu, Ruofan and Divakaran, Dinil Mon and Ng, Jun Yang and Chan, Qing Zhou and Lu, Yiwen and Si, Yuxuan and Zhang, Fan and Dong, Jin Song},
  booktitle={30th $\{$USENIX$\}$ Security Symposium ($\{$USENIX$\}$ Security 21)},
  year={2021}
}
```
## Contacts
If you have any issue running our code, you can raise an issue or send an email to liu.ruofan16@u.nus.edu
