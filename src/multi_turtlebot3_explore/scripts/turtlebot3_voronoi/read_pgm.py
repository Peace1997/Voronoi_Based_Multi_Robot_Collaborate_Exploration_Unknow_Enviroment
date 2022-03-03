# read pgm 

f = open('true.pgm', 'rb')
type = f.readline()
note = f.readline()
(width, height) = [int(i) for i in f.readline().split()]
depth = int(f.readline())
print(type,note,width,height,depth)
raster = []
num_0 = 0
num_254 = 0 
num_205 = 0
for y in range(height):
    row = []
    for y in range(width):
        data = ord(f.read(1))
        if(data == 0):
            num_0 += 1

        if(data == 254):
            num_254 += 1

        if(data == 205):
            num_205 +=1
        row.append(data)

    raster.append(row)
print(num_0,num_254,num_205)