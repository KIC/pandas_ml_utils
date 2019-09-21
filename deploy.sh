#!/bin/bash

sed -i 's/(.\/images\//(https:\/\/raw.githubusercontent.com\/KIC\/pandas_ml_utils\/master\/images\//' README.md
flit --repository testpypi publish
git checkout README.md
