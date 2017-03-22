# Copyright and license

SchellingSegregation: Exercise on the conventional Schelling segregation model
Copyright (C) 2017 Fabio Correa <facorread@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Computer requirements

This software has been developed using [Visual Basic Community 2015](https://www.visualstudio.com/vs/community/) with [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-toolkit). There is no guarantee it will work on any other setup.

For `Windows` you might want to install [MinGW](http://www.mingw.org).

# Developer notes

This source code is managed using [git](http://git-scm.org); the repository is hosted in both [GitHub](https://github.com/facorread/segregation) and [GitLab](https://gitlab.com/facorread/segregation).

If all warnings are enabled in Visual Studio (`-Wall`), it will be necessary to add the following flags to the host compiler, at Project Property Pages > Configuration Properties > CUDA C / C++ > Host > Additional Compiler Options:

`/wd4001 /wd4668 /wd4514 /wd4820 /wd4515 /wd4571 /wd4365  /wd4626 /wd5027 /wd4324 /wd4623 /wd4100 /wd4191 /wd4710`

The additional compiler option `-Za` for strict `C++11` compliance cannot be used because there are noncompliance issues with precompiled headers that were provided with the CUDA Toolkit 8.0.

Because `-Za` cannot be used, `constexpr` is unavailable for class member functions.

NVidia recommends 64-bit compilation using `--machine 64`. Memory in CUDA devices is addressed using 64 bits, and size_t is 64 bits long, but registers are 32 bits long; shared memory alignment and banks are 32 bits too. In order to optimize memory bandwidth, prefer arrays of `unsigned char` (8 bits), `unsigned short` (16 bits) or `unsigned int` (32 bits) indices over arrays of pointers.

`cuRAND` is better and more complete than `thrust` for random numbers.