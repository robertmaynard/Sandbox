extern "C" { struct foo { int x,y,z; }; }

float test_foo(foo *f)
{
  return f->x * f->y;
}

foo* make_foo()
{
  return new foo();
}
