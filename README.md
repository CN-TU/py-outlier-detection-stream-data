# Comparison and Evaluation of Outlier Detection in Streaming Data
## Instructions for experiment replication

**authors:** Felix Iglesias, Alexander Hartl, Tanja Zseby, and Arthur Zimek

**contact:** felix.iglesias@tuwien.ac.at

*May., 2023*

**Warning!!:** this repository includes results, logs, figures and tables files as obtained in the original experiments and published in the paper. Executing the scripts below will overwrite such files.

### 0. Preparing datasets

**Synthetic datasets** are located within the [datasets/synthetic/] folder. **Real datasets** are not included in order to respect original authory and licenses. They must be downloaded, preprocessed and located in their respective folders within the parent [datasets/real/] folder. The [datasets/srcs/] folder contains scripts and further instructions to allow preparing the data as used in our experiments. Please, read the meta-datasets.md document in the [datasets/] folder for additional information.

If you have problems obtaining, accessing or processing third-party datasets (i.e., "real datasets"), please contact the author of the repository.

**Synthetic datasets** are also publicly available in **Mendeley**:
“Data for Evaluation of Stream Data Analysis Algorithms”. Mendeley Data, V1, doi: 10.17632/c43kr4t7h8.1, [https://data.mendeley.com/datasets/c43kr4t7h8/1](https://data.mendeley.com/datasets/c43kr4t7h8/1)

### 1. Replicating experiments

Open a terminal and run experiments with synthetic data and real-application data separately:

> $ pyhton3 run_synthetic.py

> $ pyhton3 run_real.py

Files with results and scores are created in the corresponding folders within [tests/]. **Warning!:** *if executed in a common desktop machine this process can take several days. We recommend using high-performance equipment for this task.*

### 2. 3D scatter plots

Open a terminal from [scatterplots/]. Run:

> $ bash draw_scatterplots.sh

Plots used in the paper are generated in the [scatterplots/paper_plots/] folder.

### 3. Boxplots and critical difference diagrams for synthetic datasets

Open a terminal from [statistics/]. Run:

> $ bash extract_statistics.sh

Plots used in the paper are generated in the [statistics/paper_plots_and_tables/] folder.

Critical Distance diagrams are adapted from the scripts used in: *Ismail Fawaz, H., Forestier, G., Weber, J. et al. Deep learning for time series classification: a review. Data Min Knowl Disc 33, 917--963 (2019). https://doi.org/10.1007/s10618-019-00619-1*

and available for Python in: https://github.com/hfawaz/cd-diagram

### 4. Analysis and plots for the 2D-example

Open a terminal in the current folder. Run:

> $ python3 run_example.py

Plots used in the paper are generated in the [tests/example/] folder.

### 5. Plots for T-sensitivity experiments with real/application data

Open a terminal from [timeSreal/]. Run:

> $ bash plot_aap_tts.sh

Plots used in the paper are generated in the same folder.

### 6. Extracting phi and rho indices from datasets

Open a terminal from [evaldatasets/]. Run:

> $ bash run_evaldata.sh

The script will create a text file ("datasets_indices.txt") with descriptive information of the datasets. To generate the plot in Section 6.4., summarize phi and rho values of dataset collections (synthetic data) by using the median (use your favorite tool here). Create a file like "ind4plot.txt" with the necessary information -- dataset(s), phi, rho -- and run: 

> $ python3 phi_rho_plot.py ind4plot.txt

This creates the "phi_rot.pdf" plot.
