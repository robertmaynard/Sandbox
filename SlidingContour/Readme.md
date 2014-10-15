
Sliding Marching Cubes demo.

Invoke with:
```
./slidingTBB --file=data.nhdr  --contour=0.075 --slice=256
```

the slice parameter must be divisible by the z extent of the dataset

Performance Results for 1024^3 data on 2.6 GHZ Intel i7 ( 4core ) laptop:
```
vtk single core: 33sec
32  slices: 19.5772sec
64  slices: 12.194sec
128 slices: 9.30068sec
256 slices: 10.5784sec
512 slices: 8.81249sec
```
