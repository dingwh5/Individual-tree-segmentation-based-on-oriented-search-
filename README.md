This document provides an overview of the code and instructions for its usage. The code is designed for single-tree segmentation based on point clouds. It includes two main processes: **Oriented Search and Clustering** and **Validity Checking and Points Updating**. Below are the detailed descriptions and guidelines for running the code.

### 1. Dependencies  
The following software libraries are required to run the code:  
- **PCL (Point Cloud Library)**: Version 1.11, for point cloud data processing.  
- **OpenCV**: Version 4.2, for visualization and related operations.  

Ensure these libraries are properly installed and configured on your system before running the code.

### 2. Code Structure  
The code folder contains the following files:  
- **Code.cpp**: The main source file containing the implementation of the algorithm.  
- **CMakeLists.txt**: A configuration file for building the project using CMake.  

Key configurable parameters in the source code:  
- Line 558: Set the input folder path for point cloud files (format: .pcd).  
- Line 606: Define the search radius (Radius) for clustering operations.
### **3. Benchmark Data Sources**

This project utilizes two publicly available benchmark datasets:

- **[NEWFOR Dataset](https://www.kaggle.com/datasets/sentinel3734/newfor-tree-detection-benchmark?resource=download)**  
  Published by Eysn et al. (2015), this dataset provides LiDAR-based single tree detection benchmarks using heterogeneous forest data from the Alpine region.

- **[FORinstance Dataset](https://zenodo.org/records/8287792)**  
  Released by Puliti et al. (2023), this UAV laser scanning dataset is designed for semantic and instance segmentation of individual trees.


 **References**  
1. **Puliti, S., Pearse, G., Surový, P., et al. (2023).**  
   *For-instance: A UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees.*  
   arXiv preprint arXiv:2309.01279.  

2. **Eysn, L., Hollaus, M., Lindberg, E., et al. (2015).**  
   *A benchmark of LiDAR-based single tree detection methods using heterogeneous forest data from the Alpine space.*  
   Forests, 6(5), 1721–1747.  

### 4. Generated Output Files  
For each input point cloud, the program generates five output files, which are divided into two main stages:

#### Stage 1: Oriented Search and Clustering  
- **_cha**: Contains the XYZ coordinates and classification of each point.  
- **_cha_point**: Includes XYZ coordinates, classification, and RGB color of each point. Points belonging to the same tree share the same color.  
- **_cha_lei**: Contains the XYZ coordinates of the top point (tree apex) for each tree.

#### Stage 2: Validity Checking and Points Updating  
- **_cha_ge_point**: Similar to _cha_point, it includes the updated XYZ coordinates, classification, and RGB color of each point after validity checks. Points in the same tree retain the same color.  
- **_cha_ge_lei**: Contains the XYZ coordinates of the updated tree apex positions.

### 5. Contact Information  
For questions or issues regarding this code, please contact:  
dingwh5@tongji.edu.cn
