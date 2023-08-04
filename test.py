z0=complex(1.0, 1.4)
z = 0
n = 0
while abs(z) <= 2\
        and n < 300:
    n = n + 1
    z = z*z + z0
print(n)