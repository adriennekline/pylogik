`PyLogik`
=====

A Python package dedicated to sharing functions and classes for common image processing and statistical tools. This includes Sorensen–Dice coefficient (Dice) score and plotting functions.

---

## Citing this work:

A. Kline, PyLogik, 2022

---

* Integration with Jupyter Lab/Jupyter Notebooks
* Built-in plotting functionality for image comparisons

---

* [Installation](#installation)
* [Package Import](#image-processing-import )
* [Data Import](#data-import)
* [Dice Score Calculation](#dice-score)
* [Impairshow](#impairshow-graphical-output)
* [Conclusion](#conclusion)

---

# Installation

Install the package through pip:

```bash
$ pip install pylogik
```

----

# Image Processing Import 

```python
from pylogik.image import im_analysis
```

----

# Data Import

Options for reading in images:

* Matplotlib - `plt.imread()`

* OpenCV - `cv2.imread()`

* Pillow - `Image.open()`

* scikit-image - `io.imread()`

```python
# read in your data
im1 = cv2.imread('example_im1.png')
im2 = cv2.imread('example_im2.png')
```

----

# Dice Score
The mathematical formalism of the Dice score is denoted by the equation:

$$ DSC = \frac{2*|X \cap Y|}{|X|+ |Y|} $$

where $\cap$ denotes the intersection of two images $X$ and $Y$. 

Performing the calculation using a function in `PyLogik`:

```python
dice = im_analysis.dice_score(pred, true, k=1)
```

**Note:**
* `pred` - array of the predicted segmentation
* `true` - array of the ground truth segmentation
* `k` - value to perform matching on (default = 1)
* Returns: dice score (float)

---

# Impairshow Graphical Output

`imshowpair` returns the array and image associated with a dice score comparison of 2 logical images. Colors are prespecified as magenta and green but can be adjusted by the user. 

```python
im_analysis.imshowpair(pred, true, color1 = (124,252,0), color2 = (255,0,252), show_fig = True):
```

**Note:**
*  `pred` - array of the predicted segmentation
*  `true` - array of the ground truth segmentation
*  `color1` - first color to show unique values from first image
*  `color2` - second color to show unique values from second image
* Returns: array and graphical plot

![dice_demo](https://github.com/adriennekline/pylogik/blob/main/demo/dice_demo.jpg)

---

# Conclusion
This package offers a user friendly dice score calculation and dice score plotting functionality to showcase the intersection and complement of each image relative to the other. This package will be continually built on to incorporate other statistical and image processin functionality 
