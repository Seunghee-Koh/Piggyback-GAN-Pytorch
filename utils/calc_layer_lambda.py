import pdb
import math
def calc_layer_lambda(lamb_type='fixed', average_val=0.25):
    out = []
    if lamb_type == 'fixed':
        for _ in range(15):
            out.append(average_val)
        out.append(1.)
    elif lamb_type == 'exponential':
        val = 1.0
        append_val = val
        for i in range(16):
            out.append(append_val)
            val *= 0.8
            append_val = math.ceil(val*64)/64 
        out.reverse()
    elif lamb_type == 'linear':
        raise NotImplementedError
    return out

if __name__ == '__main__':
    llist = calc_layer_lambda('exponential', 0.25)
    # for i in range(16):
    #     llist[i] = llist[i]*64
    print(llist, len(llist))