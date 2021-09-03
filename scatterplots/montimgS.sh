

convert -crop 1600x400+200+0 plots/base_data_14.arff_long.png plots/base_data.arff_long.png
convert -crop 1600x400+200+0 plots/close_data_12.arff_long.png plots/close_data.arff_long.png
convert -crop 1600x400+200+0 plots/dens-diff_data_18.arff_long.png plots/dens-diff_data.arff_long.png
convert -crop 1600x400+200+0 plots/h-out_data_14.arff_long.png plots/h-out_data.arff_long.png
convert -crop 1600x400+200+0 plots/m-out_data_16.arff_long.png plots/m-out_data.arff_long.png
convert -crop 1600x400+200+0 plots/moving_data_20.arff_long.png plots/moving_data.arff_long.png
convert -crop 1600x400+200+0 plots/nonstat_data_18.arff_long.png plots/nonstat_data.arff_long.png
convert -crop 1600x400+200+0 plots/overlap_data_13.arff_long.png plots/overlap_data.arff_long.png
convert -crop 1600x400+200+0 plots/sequential_data_5.arff_long.png plots/sequential_data.arff_long.png

montage plots/base_data.arff_long.png plots/base_data_14.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/base.png 
montage plots/close_data.arff_long.png plots/close_data_12.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/close.png 
montage plots/dens-diff_data.arff_long.png plots/dens-diff_data_18.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/dens.png 
montage plots/h-out_data.arff_long.png plots/h-out_data_14.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/h-out.png 
montage plots/m-out_data.arff_long.png plots/m-out_data_16.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/m-out.png 
montage plots/moving_data.arff_long.png plots/moving_data_20.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/moving.png 
montage plots/nonstat_data.arff_long.png plots/nonstat_data_18.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/nonstat.png 
montage plots/overlap_data.arff_long.png plots/overlap_data_13.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/overlap.png 
montage plots/sequential_data.arff_long.png plots/sequential_data_5.arff_short.png -tile 2x1 -geometry +0+0 paper_plots/seq.png 

