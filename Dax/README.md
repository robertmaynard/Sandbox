Dax based Sandcastles that I am keeping around for historical reasons


## ConceptualExecutive  ##
An attempt to create a delayed execution pipeline that could be used
by Dax/VTK-m.


## GenericVariadicParser ##
An in progress attempt at being able to strip and pack function arguments
into a storage mechanism so that we can extract them latter. The main feature
is that subclasses can state how many arguments they want not packed in the
opaque container.

So basically you have parser that takes 10 paramenters. Than
derived parser can state it wants only the first two. It than has
the ability to add parameters or remove the ones it explicitly asked for.

Example:
```
void operator(Functor f, T param1, O param2, RestOfParameters rp)
{
params::invoke(f,param1,myInsertedParam,param2,rp);
}
```


## LHMC ##
An example of looking at using Low High tables to speed up operations such
as Marching Cubes


## Low Level Dax ##
An example of writing vis algorithms using everything but the Dax Scheduler
infrastructure.


## PortConcept ##
More brainstorming about filters and worklets in DAX.


## SlidingContour ##
Slides through chunks of a volume along the Z axis performing Marching
Cubes.

