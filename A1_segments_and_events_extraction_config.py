from fittes import smooth


# parameters: choose the right parameters that work for the data

expected_open_current = None  # None, pA

cutoff = (5000, 20000) # define two cutoffs for current filter: first one for event detection, second one for extraction of current features

sigma_tol = 6.0  # threshold to start an event ( x*std(OPC) )

sigma_tol_out = 3.0  # threshold to end an event ( x*std(OPC) )

filter_fct = smooth   # None,smooth





# more parameters that need adjustment in specific cases
sigma_heart = 1 # threshold to cut the heart of the event ( x*std(gradient within detected event) )
voltage_segmentation = True  # flag to enable/disable segmentation based on constant voltage
residual_thr = 0.2  # ratio
resolution = 1.0  # pA
swap_channel = False
n_skip = 1  # event detection speed tradeoff with minimum event length
min_segment_duration = 1.0  # s
dt_stab = 0.0  # ms
rescale_current_factor = 1000 # does nothing