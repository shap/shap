

def standard_combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = [(mult[l]*(orig_inp[l] - bg_data[l])).mean(0)
                 for l in range(len(orig_inp))]
    return to_return
