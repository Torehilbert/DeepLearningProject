
A = 10

if(hasattr(A, '__len__')):
    print(len(A))
    print('Yes')
else:
    print(A)
    print('No')