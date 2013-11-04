A simple program that compares different way to resolve duplicate points.
We compare vtk's vtkMergePoints, vtkIncrementalOctreePointLocator, and
using a std::vector that is sorted, uniqueified and than scan with lower_bounds
to find the new values.

Not unsurprising to me, using a std::vector and the sort, unique, lower_bounds
is the best for performance.