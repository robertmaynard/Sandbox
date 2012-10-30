#ifndef __arg_
#define __arg_

namespace arg
{
class Field {};

template <int> class Arg {};

namespace placeholders {
 typedef Arg<1> _1;
 typedef Arg<2> _2;
 typedef Arg<3> _3;
 typedef Arg<4> _4;
 typedef Arg<5> _5;
 typedef Arg<6> _6;
 typedef Arg<7> _7;
 typedef Arg<8> _8;
 typedef Arg<9> _9;

}

}

#endif
