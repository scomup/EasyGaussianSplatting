# More information about real spherical harmonics can be obtained from:
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
# https://github.com/NVlabs/tiny-cuda-nn/blob/master/scripts/gen_sh.py

SH_C0_0 = 0.28209479177387814  # Y0,0:  1/2*sqrt(1/pi)       plus

SH_C1_0 = -0.4886025119029199  # Y1,-1: sqrt(3/(4*pi))       minus
SH_C1_1 = 0.4886025119029199   # Y1,0:  sqrt(3/(4*pi))       plus
SH_C1_2 = -0.4886025119029199  # Y1,1:  sqrt(3/(4*pi))       minus

SH_C2_0 = 1.0925484305920792   # Y2,-2: 1/2 * sqrt(15/pi)    plus
SH_C2_1 = -1.0925484305920792  # Y2,-1: 1/2 * sqrt(15/pi)    minus
SH_C2_2 = 0.31539156525252005  # Y2,0:  1/4*sqrt(5/pi)       plus
SH_C2_3 = -1.0925484305920792  # Y2,1:  1/2*sqrt(15/pi)      minus
SH_C2_4 = 0.5462742152960396   # Y2,2:  1/4*sqrt(15/pi)      plus

SH_C3_0 = -0.5900435899266435  # Y3,-3: 1/4*sqrt(35/(2*pi))  minus
SH_C3_1 = 2.890611442640554    # Y3,-2: 1/2*sqrt(105/pi)     plus
SH_C3_2 = -0.4570457994644658  # Y3,-1: 1/4*sqrt(21/(2*pi))  minus
SH_C3_3 = 0.3731763325901154   # Y3,0:  1/4*sqrt(7/pi)       plus
SH_C3_4 = -0.4570457994644658  # Y3,1:  1/4*sqrt(21/(2*pi))  minus
SH_C3_5 = 1.445305721320277    # Y3,2:  1/4*sqrt(105/pi)     plus
SH_C3_6 = -0.5900435899266435  # Y3,3:  1/4*sqrt(35/(2*pi))  minus

SH_C4_0 = 2.5033429417967046  # Y4,-4:  3/4*sqrt(35/pi)       plus
SH_C4_1 = -1.7701307697799304  # Y4,-3:  3/4*sqrt(35/(2*pi))  minus
SH_C4_2 = 0.9461746957575601  # Y4,-2:  3/4*sqrt(5/pi)        plus
SH_C4_3 = -0.6690465435572892  # Y4,-1:  3/4*sqrt(5/(2*pi))   minus
SH_C4_4 = 0.10578554691520431  # Y4,0:  3/16*sqrt(1/pi)       plus
SH_C4_5 = -0.6690465435572892  # Y4,1:  3/4*sqrt(5/(2*pi))    minus
SH_C4_6 = 0.47308734787878004  # Y4,2:  3/8*sqrt(5/pi)        plus
SH_C4_7 = -1.7701307697799304  # Y4,3:  3/4*sqrt(35/(2*pi))   minus
SH_C4_8 = 0.6258357354491761  # Y4,4:  3/16*sqrt(35/pi)       plus

SH_C5_0 = -0.65638205684017015
SH_C5_1 = 8.3026492595241645
SH_C5_2 = -0.48923829943525038
SH_C5_3 = 4.7935367849733241
SH_C5_4 = -0.45294665119569694
SH_C5_5 = 0.1169503224534236
SH_C5_6 = -0.45294665119569694
SH_C5_7 = 2.3967683924866621
SH_C5_8 = -0.48923829943525038
SH_C5_9 = 2.0756623148810411
SH_C5_10 = -0.65638205684017015
