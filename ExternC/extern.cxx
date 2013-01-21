
  struct foo
  {
    int x;
    float y, z;
  };


extern "C" {

  foo f;

  float test_foo(foo *f)
  {
    return f->x * f->y;
  }

  foo* make_foo()
  {
  return new foo();
  }


}

foo* make_foo2()
{
  return new foo();
}


