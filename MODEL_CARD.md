# Model Card - Random Forest Classifier for Salaries Under Census Data

## Model Details
- Developed by Luis Alfredo León
- Random Forest Classifier
- Trained with census data to predict if, given a census datapoint, an observation will perceive a salary of over/under 50K.

## Intended Use
- The intended use for this is merely educational, however it can depict:
  - The real use case where we want to estimate a person's salary based on the person's demographics.
  - The real use case where based on some salary we can predict some person's demographics.  

## Training Data
- The training data is a subset of 80% on the https://archive.ics.uci.edu/ml/datasets/census+income dataset.

## Evaluation Data  
- A 20% split on the original dataset.

## Metrics
- Test set metrics (fbeta, precision, recall): 0.74, 0.60, 0.66

## Ethical Considerations
- It is worth noting that this is mainly made for educational purposes.
- Even though the model was trained considering some slices of the dataset, we can still violate fairness if we focus in different aspects.
- We can easily obtain certain bias once our classifier gets to work, it won't update on new data.

## Caveats and Recommendations
- The model seems to be skewed towards "white" population. The sliced metrics on the data suggest that underrepresented populations such as "amer-indian" need more leverage to make the model more fair.