

void checker()
{
  using vec_float = std::vector<float>;
  using short_list = list<float, vec_float, vec_float, long>;
  using long_list = list<
      float, vec_float, vec_float, vec_float, float, vec_float, vec_float,
      vec_float, float, double, double, vec_float, float, vec_float, vec_float,
      vec_float, float, vec_float, long, long, float, short, short, vec_float,
      float, vec_float, vec_float, vec_float, float, vec_float, int, int, float,
      vec_float, vec_float, float, double, double, vec_float, float, vec_float,
      vec_float, vec_float, float, vec_float, long, long, float, short, short,
      vec_float, float, vec_float, vec_float, vec_float, float, vec_float, int,
      int, float, vec_float, vec_float, float, double, double, vec_float, float,
      vec_float, vec_float, vec_float, float, vec_float, long, long, float,
      short, short, vec_float, float, vec_float, vec_float, vec_float, float,
      vec_float, int, int, float, vec_float, vec_float, float, double, double,
      vec_float, float, vec_float, vec_float, vec_float, float, vec_float, long,
      long, float, short, short, vec_float, float, vec_float, vec_float,
      vec_float, float, vec_float, int, int, float, vec_float, vec_float, float,
      double, double, vec_float, float, vec_float, vec_float, vec_float, float,
      vec_float, long, long, float, short, short, vec_float, float, vec_float,
      vec_float, vec_float, float, vec_float, int, int, float, vec_float,
      vec_float, float, double, double, vec_float, float, vec_float, vec_float,
      vec_float, float, vec_float, long, long, float, short, short, vec_float,
      float, vec_float, vec_float, vec_float, float, vec_float, int, int, float,
      vec_float, vec_float, float, double, double, vec_float, float, vec_float,
      vec_float, vec_float, float, vec_float, long, long, float, short, short,
      vec_float, float, vec_float, vec_float, vec_float, float, vec_float, int,
      int, float, vec_float, vec_float, float, double, double, vec_float, float,
      vec_float, vec_float, vec_float, float, vec_float, long, long, float,
      short, short, vec_float, float, vec_float, vec_float, vec_float, float,
      vec_float, int, int, float, vec_float, vec_float, vec_float>;

  using front_short_type = at_c<short_list, 0UL>;
  using back_short_type = at_c<short_list, 3UL>;

  using long_type_0 = at_c<long_list, 0UL>;
  using long_type_10 = at_c<long_list, 10UL>;
  using long_type_15 = at_c<long_list, 15UL>;
  using long_type_20 = at_c<long_list, 20UL>;
  using long_type_25 = at_c<long_list, 25UL>;
  using long_type_30 = at_c<long_list, 30UL>;

  using long_type_70 = at_c<long_list,70UL>;
  using long_type_80 = at_c<long_list,80UL>;
  using long_type_90 = at_c<long_list,90UL>;
  using long_type_100 = at_c<long_list,100UL>;
  using long_type_110 = at_c<long_list,110UL>;
  using long_type_120 = at_c<long_list,120UL>;
  using long_type_130 = at_c<long_list,130UL>;
  using long_type_140 = at_c<long_list,140UL>;
  using long_type_150 = at_c<long_list,150UL>;
  using long_type_160 = at_c<long_list,160UL>;
  using long_type_170 = at_c<long_list,170UL>;
  using long_type_180 = at_c<long_list,180UL>;
  using long_type_190 = at_c<long_list,190UL>;

  using back_long_type = at_c<long_list, 224UL>;

  //should print out long type
  std::cout << "short" << std::endl;
  std::cout << typeid(front_short_type{}).name() << std::endl;
  std::cout << typeid(back_short_type{}).name() << std::endl;

  std::cout << "long" << std::endl;
  std::cout << typeid(long_type_0{}).name() << std::endl;
  std::cout << typeid(long_type_10{}).name() << std::endl;
  std::cout << typeid(long_type_15{}).name() << std::endl;
  std::cout << typeid(long_type_20{}).name() << std::endl;
  std::cout << typeid(long_type_130{}).name() << std::endl;
  std::cout << typeid(long_type_190{}).name() << std::endl;
  std::cout << typeid(back_long_type{}).name() << std::endl;
}
