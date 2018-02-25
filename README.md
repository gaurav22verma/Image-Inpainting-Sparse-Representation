# Image-Inpainting-Sparse-Representation
Image inpainting using sparse representations (Orthogonal Matching Pursuit (OMP) and Iteratively Reweighted Least Squares Method (IRLS)). This was done as a part of the course work for EE698K at IIT Kanpur. 

# Results
![PatchFillingResults](https://github.com/TheGalileo/Image-Inpainting-Sparse-Representation/blob/master/images/ResultsOMPIRLS.jpg)  
![OMPandIRLS results](https://github.com/TheGalileo/Image-Inpainting-Sparse-Representation/blob/master/images/ResultsPatchFilling.jpg)  
For more details, please refer to `Report.pdf`

# Steps
1. Run the `SparseInpainting.py` using `python SparseInpainting.py`
 * The parameters (like dictionary size, patch size etc.) can be changed from this file itself. Refer to line #8 to line # 14.
 * Don't forget to change line #73, #75, #76, and #113 if you want to run the algorithm on different files (the default is Daniel Radcliff).
 * Comment/uncomment line #183 and #184 to run OMP or IRLS

2. Run `computeNIQE.py` using `python computeNIQE.py` to get the NIQE value
 * Please change the filename in line #5

3. `Report.pdf` is the report for the assignment

# Comments/Suggestions
Please feel free to open an issue, in case you have a question/comment/suggestion. I will try to revert as soon as possible.
