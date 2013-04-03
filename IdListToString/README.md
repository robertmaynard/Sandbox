Example of converting vtkIdLists or vtkIdType* to a string key that can be used
for unique comparison. This string that is generated will be based on the sorted
order of the ids.

```
  std::string key = create_key(vtkIdList* ids);
  std::string key = create_key(vtkIdType* ids, vtkIdType size);
```

Remember since we sort the ids, the ids 1,2,4,5 will be equal to 5,4,2,1.
Note we will NOT remove duplicate ids, so 1,2,2,2,4,5 is NOT equal to 1,2,4,5



