# DSVM
In this study, we construct a prediction model for endometrial cancer grade based on the expression data.  <br>
DSVM is a support vector machine model with sigmoid kernel functions combined with z-score standardization.  <br>

# Dependency:
Python 3.7.12 <br>
Scikit-learn 0.22.1 <br>
matplotlib 3.5.3  <br>
numpy 1.21.6   <br>
pandas 1.3.5   <br>
R 3.2.3 <br>
r-dplyr 0.7.8  <br>
r-survminer 0.4.3    <br>
r-survival 2.42_6   <br>

# Data 
Downloaded from https://github.com/gargyapeter/ucec_ml_grade2021.

ucec_tcga_clinical_data.zip: It contains raw clinical data, and it's the input of R_survival_analysis.Rmd.
uterus_rnaseq_VST1.z01, uterus_rnaseq_VST1.z02, uterus_rnaseq_VST1.zip, uterus_rnaseq_VST_G2.zip: These zip files contain the processed gene expression data, and they are the input of DSVM.py.   <br>


# Usage:
python DSVM.py 

R -e "rmarkdown::render('R_survival_analysis.Rmd',output_file='output.html')"


# Contact
Xindi Yu: yuxindi53@foxmail.com <br>

