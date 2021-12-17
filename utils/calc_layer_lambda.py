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
    elif lamb_type == 'exponential2':
        val = 1.0
        append_val = val
        out.append(append_val)
        minimum = 1/8
        for i in range(15):
            val *= 0.8
            append_val = math.ceil(val*64)/64 
            out.append(max(append_val, minimum))
        out.reverse()
    elif lamb_type == 'exponential3':
        val = 1.0
        append_val = val
        out.append(append_val)
        minimum = 1/8
        for i in range(15):
            val *= 0.75
            append_val = math.ceil(val*64)/64 
            out.append(max(append_val, minimum))
        out.reverse()
    elif lamb_type == 'exponential4':
        val = 1.0
        append_val = val
        out.append(append_val)
        minimum = 1/8
        for i in range(15):
            val *= 0.7
            append_val = math.ceil(val*64)/64 
            out.append(max(append_val, minimum))
        out.reverse()
    elif lamb_type == 'exponential5':
        val = 1.0
        append_val = val
        out.append(append_val)
        minimum = 1/8
        for i in range(15):
            val *= 0.78
            append_val = math.ceil(val*64)/64 
            out.append(max(append_val, minimum))
        out.reverse()
    elif lamb_type == 'd_exponential1':
        val = 0.8
        append_val = val
        out.append(append_val)
        minimum = 1/8
        for i in range(7):
            val *= 0.78
            append_val = math.ceil(val*64)/64 
            out.append(max(append_val, minimum))
        for i in reversed(range(8)):
            out.append(out[i])
    elif lamb_type == 'd_exponential2':
        val = 0.8
        append_val = val
        out.append(append_val)
        minimum = 1/8
        for i in range(7):
            val *= 0.7
            append_val = math.ceil(val*64)/64 
            out.append(max(append_val, minimum))
        for i in reversed(range(8)):
            out.append(out[i])
    elif lamb_type == 'd_exponential3':
        val = 0.75
        append_val = val
        out.append(append_val)
        minimum = 1/8
        for i in range(7):
            val *= 0.78
            append_val = math.ceil(val*64)/64 
            out.append(max(append_val, minimum))
        for i in reversed(range(8)):
            out.append(out[i])
    return out

if __name__ == '__main__':
    llist = calc_layer_lambda('d_exponential3', 0.25)
    out_list = [64,128,256,512,512,512,512,512,512,512,512,512,256,128,64,3]
    in_list = [3,64,128,256,512,512,512,512,1024,1024,1024,1024,1024,512,256,128]

    sum_param = 0
    for i in range(16):
        sum_param += llist[i]*out_list[i]*in_list[i]

    print(f'sum of parameters is {sum_param}, and is {sum_param/915888:.2f} times bigger than fixed')
    print(llist)
    for i in range(16):
        llist[i] = llist[i]*64
    print(llist)

