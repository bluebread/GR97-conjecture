a = -1 
b = 2
c = -3

for i in range(1000):
    a = a + b
    print(hex(a))
    
for i in range(1000):
    a = a + c
    print(hex(a))
    
def int_to_twos_complement(num, bits):
    if num >= 0:
        return bin(num)[2:].zfill(bits)
    return bin((1 << bits) + num)[2:]
# Convert integer to two's complement with 8 bits
print(int_to_twos_complement(-5, 256))