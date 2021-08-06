import numpy as np

def squarify(M,val):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val), padding

def remove_padding(M,padding):
    top,bottom=padding[0]
    left,right=padding[1]
    
    index_top = 0+top
    index_bottom= M.shape[0]-bottom
    index_left = 0+left
    index_right = M.shape[1]-right
    
    return M[index_top:index_bottom, index_left:index_right]

