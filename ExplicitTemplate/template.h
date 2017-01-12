

class TemplateTester
{
public:
  TemplateTester();
  ~TemplateTester();

  float Do();

  template<typename T> void DoWith(T &t);
};
