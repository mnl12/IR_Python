def sgn(x):
    if x>0:
        return 'Positive'
    elif x<0:
        return 'Negetive'
    else:
        return 'Zero'


for x in [-5,0,4]:
    print(sgn(x))
