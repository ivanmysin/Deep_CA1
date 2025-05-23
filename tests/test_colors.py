import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

plotting_colors = {
    "neuron_colors" : {
        "pyr deep" : "#FF0000", #(1.0, 0.0, 0.0), # red
        "pyr sup": "#8B0000", # (0.8, 0.2, 0.0),  #

        "pvbas": (0.0, 0.0, 1.0), # blue
        "olm": (0.0, 0.0, 0.5), #
        "cckbas": (0.0, 1.0, 0.0), # green
        "ivy": (0.0, 0.5, 0.5), #
        "ngf": (0.5, 0.5, 0.5), #
        "bis": "#00FFFF", #(0.1, 0.0, 0.5), #
        "aac": "#7B68EE", #(1.0, 0.0, 0.5), #
        "sca": '#32CD32', #(0.0, 1.0, 0.5), #

        "ISI R-O": (0.9, 0.7, 0.0),  #
        "ISI RO-O": '#FFD700', # (0.0, 0.8, 0.5),  #
        "tril": "#ADD8E6", #(0.0, 0.5, 0.9),  #

        "ca3": "#FFA07A", #(0.5, 0.5, 0.0), #
        "ec3": "#F08080", # (0.5, 1.0, 0.0), #


    },

    "neurons_order" : ["pvbas", "olm", "cckbas", "bis", "aac",  "ivy", "ngf"],
    "generators_order" : ["ec3", "ca3pyr", "ca1pyr"],
}


fig, axes = plt.subplots(ncols = len(plotting_colors['neuron_colors'].values()), figsize=(15, 5) )


for color_idx, (color_name, color_rgb) in enumerate(plotting_colors['neuron_colors'].items()):
    axes[color_idx].axis('off')

    axes[color_idx].set_xlim(0, 1)
    axes[color_idx].set_ylim(0, 1)



    axes[color_idx].add_patch(mpatch.Rectangle((0, 0), 1, 1, color=color_rgb))

    axes[color_idx].set_title(color_name)



plt.show()