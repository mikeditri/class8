# class7
homework for class7

How to run the docker container:

Clone repo to location.

inlcude winpty if running on windows machine

winpty docker run -ti -v /${pwd.local_where_the_files_are}:/app mikeditri/class7:latest

This should create all the plots needed for analysis



The GOAT plot in my opinion is "Proline_vs_OD280 OD315 of diluted wines_size_Flavanoids_scatter_pairs.png" 
because keeping in mind that I am trying to predict the class (or vinyard) where the wine is produced based on some variables, this plot shows clear grouping of the classes based on OD280 and proline values as well as flavanoids.


The following are examples of not selected because it is clear that the class groups overlap which would not help in predicting where the wine comes from.

"Alcalinity of ash_vs_Alcohol_size_Class_scatter_pairs.png"
"Ash_vs_Malic Acid_size_Alcohol_scatter_pairs.png"
"Color intensity_vs_Ash_size_Alcohol_scatter_pairs.png"

