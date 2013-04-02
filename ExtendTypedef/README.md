Meta template utilities to modify and extend a typedef that represents
a function call. For example it can take something like:
    void Sig(_1,_2,_3)

and convert it to
    void Sig(_1,_2,_4,_3)

Classes that are provided:

    ReplaceAndExtendSignatures<Functor, ArgumentToMatch, ArgumentToReplaceItWith>;
    BuildSignature<FunctionSignature>;
    ExtendedFunctor<Functor,NewContSig,NewExecSig>;
    ConvertToBoost<RealFunctor> BoostExtendFunctor;

all classes expect Functor to represent a class with the typedef to modify
be named ControlSignature or ExecutionSignature.

This is all research code to be used in DAX, and this should be only
used as a draft example of the final implementation