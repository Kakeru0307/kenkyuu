import os

indices1 = []
indices2 = []
indices3 = []
for i in range(1000):
    fpath1 = f"uncond_1000/{i}/ours.mid"
    fpath2 = f"uncond_1000/{i}/mmt.mid"
    fpath3 = f"uncond_1000/{i}/atc.mid"
    if not os.path.exists(fpath1):
        indices1.append(fpath1)
    if not os.path.exists(fpath2):
        indices2.append(fpath2)
    if not os.path.exists(fpath3):
        indices3.append(fpath3)
print(indices1, indices2, indices3)

indices1 = []
indices2 = []
indices3 = []
for i in range(1000):
    fpath1 = f"control_exp_random/{i}/output_wo_drum.mid"
    fpath2 = f"control_exp_random/{i}/qna_good_control.mid"
    fpath3 = f"control_exp_random/{i}/qna_simple_control.mid"
    if not os.path.exists(fpath1):
        indices1.append(fpath1)
    if not os.path.exists(fpath2):
        indices2.append(fpath2)
    if not os.path.exists(fpath3):
        indices3.append(fpath3)
print(indices1, indices2, indices3)