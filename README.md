<div align="center">
  lan_py_common
  <p>Working in progress Common python building blocks for effective mathematics and data exploration</p>
</div>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg"
      alt="License: MIT" />
  </a>
  <a href="https://buymeacoffee.com/lan22h">
    <img src="https://img.shields.io/static/v1?label=Buy me a coffee&message=%E2%9D%A4&logo=BuyMeACoffee&link=&color=greygreen"
      alt="Buy me a Coffee" />
  </a>
</p>

<!-- <p align="center">

  <a href="https://github.com/sponsors/jeffreytse">
    <img src="https://img.shields.io/static/v1?label=sponsor&message=%E2%9D%A4&logo=GitHub&link=&color=greygreen"
      alt="Donate (GitHub Sponsor)" />
  </a>

  <a href="https://github.com/jeffreytse/zsh-vi-mode/releases">
    <img src="https://img.shields.io/github/v/release/jeffreytse/zsh-vi-mode?color=brightgreen"
      alt="Release Version" />
  </a>

  <a href="https://liberapay.com/jeffreytse">
    <img src="http://img.shields.io/liberapay/goal/jeffreytse.svg?logo=liberapay"
      alt="Donate (Liberapay)" />
  </a>

  <a href="https://patreon.com/jeffreytse">
    <img src="https://img.shields.io/badge/support-patreon-F96854.svg?style=flat-square"
      alt="Donate (Patreon)" />
  </a>

  <a href="https://ko-fi.com/jeffreytse">
    <img height="20" src="https://www.ko-fi.com/img/githubbutton_sm.svg"
      alt="Donate (Ko-fi)" />
  </a>

</p> -->

<div align="center">
  <sub>Built with ❤︎ by Mohammed Alzakariya
  <!-- <a href="https://jeffreytse.net">jeffreytse</a> and
  <a href="https://github.com/jeffreytse/zsh-vi-mode/graphs/contributors">contributors </a> -->
</div>
<br>

<!-- <img alt="TTM Demo" src="https://user-images.githubusercontent.com/9413602/105746868-f3734a00-5f7a-11eb-8db5-22fcf50a171b.gif" /> TODO -->

- [What is this?](#what-is-this)
- [Contributing](#contributing)
- [License](#license)


# What is this?
This is an ad-hoc library for type-safe data exploration in python and other common python needs. It includes schema-aware dataframes as well as some common plotting functions and mathematical types for sympy. Independent libraries can split from this given significant development or when the need arises. 

Here is an example for minimal plotting over an Iris dataset in a jupyter notebook:

```py
from lan_py_common.lib import *

df = Df.from_schema_and_csv(
    DfJsonSchema.from_dict({
        'ty': 'Iris',
        'Id': 'int',
        'SepalLengthCm': 'float',
        'SepalWidthCm': 'float',
        'PetalLengthCm': 'float',
        'PetalWidthCm': 'float',
        'Species': 'str',
    }).unwrap(),
    '~/data/kaggle/datasets/uciml/iris/Iris.csv'
).unwrap()

plot_df = to_scatter2_df(df, "PetalWidthCm", "PetalLengthCm").unwrap()
mpl_scatter2(plot_df, groupby_method="sum")
```

The dataframe schema also allows null values within the columns if one specifies Optional[int] instead of int for example.

# Contributing

All contributions are welcome! I would appreciate feedback on improving the library and optimizing for use cases I haven't thought of yet! Please feel free to contact me by opening an issue ticket or emailing lanhikarixx@gmail.com if you want to chat.

# License

This theme is licensed under the [MIT license](https://opensource.org/licenses/mit-license.php) © 2025 Mohammed Alzakariya.
