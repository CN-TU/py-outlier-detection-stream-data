
convert -crop 1600x400+200+0 plots/shuttle.arff02_long.png plots/shuttle.arff_long.png
convert -crop 1600x400+200+0 plots/swan.arff02_long.png plots/swan.arff_long.png
convert -crop 1600x400+200+0 plots/yahoo.arff02_long.png plots/yahoo.arff_long.png
convert -crop 1600x400+200+0 plots/optout.arff23_long.png plots/optout.arff_long.png

mkdir paper_plots
montage plots/shuttle.arff_long.png plots/shuttle.arff02_short.png -tile 2x1 -geometry +0+0 paper_plots/shuttle.png 
montage plots/swan.arff_long.png plots/swan.arff02_short.png -tile 2x1 -geometry +0+0 paper_plots/swan.png 
montage plots/yahoo.arff_long.png plots/yahoo.arff02_short.png -tile 2x1 -geometry +0+0 paper_plots/yahoo.png 
montage plots/optout.arff_long.png plots/optout.arff23_short.png -tile 2x1 -geometry +0+0 paper_plots/optout.png 

